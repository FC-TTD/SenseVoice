# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum
from io import BytesIO
import logging
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import gradio as gr
from pydantic import BaseModel, Field
import soundfile as sf
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
import torch
import torch.multiprocessing as mp
import torchaudio
from ttd_fastapi_utils import setup_cuda_health
from typing_extensions import Annotated
from uuid import uuid4

from model import SenseVoiceSmall
from utils.asd_service import clear_model_cache as clear_asd_model_cache
from utils.asd_service import get_model_cache_stats as get_asd_model_cache_stats
from utils.asd_service import infer as asd_infer
from utils.asd_service import load_model as load_asd_model
from utils.lazy_model_manager import LazyModelManager
from utils.pri import PriFile
from utils.vec import Wav2Vec2VAD

# 添加日志过滤器，用于过滤健康检查和文档请求的日志
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage() and "/docs" not in record.getMessage()

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 添加标准输出处理器，确保日志输出到控制台
    ]
)

# 创建自定义日志记录器
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

# 确保日志输出到标准输出
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 应用日志过滤器
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

"""并发模型：单进程 + 动态线程池 + 延迟加载模型。
适用于低频高并发批量处理场景，空闲时释放内存。
"""
# 获取系统 CPU 核心数
cpu_count = os.cpu_count() or 1
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # 可能已经设置过启动方法
    pass

# 配置线程池
MAX_WORKERS = int(os.getenv("MAX_CONCURRENT_REQUESTS", min(32, cpu_count * 2)))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
logger.info(f"线程池配置: max_workers={MAX_WORKERS}, cpu_count={cpu_count}")

# 配置并发限流
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_REQUESTS", min(32, cpu_count * 2)))
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# 配置模型空闲超时（秒）
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", 300))  # 默认 5 分钟
ENABLE_AUTO_UNLOAD = os.getenv("ENABLE_AUTO_UNLOAD", "true").lower() == "true"

# Pydantic 模型定义
class ASRItem(BaseModel):
    text: str = Field(description="处理后的文本")
    raw_text: str = Field(description="原始文本")
    clean_text: str = Field(description="清理后的文本")
    key: str = Field(description="音频文件名")

class ASRResponse(BaseModel):
    result: List[ASRItem] = Field(description="ASR 识别结果列表")

class ASDRequestMeta(BaseModel):
    request_id: str = Field(description="请求唯一标识")
    key: str = Field(description="音频文件名")
    requested_language: str = Field(description="请求传入的语言参数")
    resolved_language: str = Field(description="实际执行时使用的语言参数")
    requested_model: Optional[str] = Field(default=None, description="请求传入的模型参数")
    resolved_model: str = Field(description="实际执行时使用的模型类型")

class ASDAudioMeta(BaseModel):
    sample_rate: int = Field(description="音频采样率")
    duration_seconds: float = Field(description="音频时长（秒）")
    size_bytes: int = Field(description="上传音频字节大小")

class ASDTimingMeta(BaseModel):
    processing_ms: int = Field(description="处理耗时（毫秒）")

class ASDSegmentItem(BaseModel):
    start: int = Field(description="分段起始时间（毫秒）")
    end: int = Field(description="分段结束时间（毫秒）")
    text: str = Field(description="处理后的文本")
    raw_text: str = Field(description="原始文本")
    clean_text: str = Field(description="清理后的文本")
    speaker: Optional[str] = Field(default=None, description="说话人标识")
    key: str = Field(description="音频文件名")

class ASDResultData(BaseModel):
    segments: List[ASDSegmentItem] = Field(description="结构化识别结果列表")
    text: str = Field(description="最终展示文本")
    raw_text: str = Field(description="原始文本")
    clean_text: str = Field(description="清洗后的文本")
    srt: str = Field(description="字幕结果")

class ASDData(BaseModel):
    request: ASDRequestMeta = Field(description="请求与路由信息")
    audio: ASDAudioMeta = Field(description="音频元信息")
    timing: ASDTimingMeta = Field(description="耗时信息")
    result: ASDResultData = Field(description="识别结果")

class ASDEnvelope(BaseModel):
    code: int = Field(description="业务状态码，0 表示成功")
    message: str = Field(description="响应消息")
    data: Optional[ASDData] = Field(default=None, description="响应数据")


def _asd_error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": message,
            "data": None,
        },
    )

class VADData(BaseModel):
    v: float = Field(description="Valence 值", default=0)
    a: float = Field(description="Arousal 值", default=0)
    d: float = Field(description="Dominance 值", default=0)
    raw: List[float] = Field(description="原始数据", default_factory=list)

class VADResponse(BaseModel):
    result: VADData = Field(description="VAD 分析结果")

class LoudnessData(BaseModel):
    itgr: float = Field(description="综合响度值")
    max: float = Field(description="最大响度值")

class PitchData(BaseModel):
    mean: float = Field(description="基频均值 (Hz)")
    max: float = Field(description="基频最大值 (Hz)")

class PRIData(BaseModel):
    mean_pri: str = Field(description="平均 PRI 值")
    max_pri: str = Field(description="最大 PRI 值")
    rate: float = Field(description="语速 (WPM)")
    loundness: LoudnessData = Field(description="响度数据")
    pitches: PitchData = Field(description="音高数据")

class PRIResponse(BaseModel):
    result: PRIData = Field(description="PRI 分析结果")

TARGET_FS = 16000
LONG_AUDIO_THRESHOLD_SECONDS = int(os.getenv("ASR_LONG_AUDIO_THRESHOLD_SECONDS", "120"))

class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"

# 全局变量用于存储模型管理器
model_manager_asr = None
model_manager_vad = None
model_manager_pri = None

# 超时中间件，处理长时间运行的请求
class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            # 设置请求超时时间（秒）
            timeout = int(os.getenv("SENSEVOICE_REQUEST_TIMEOUT", "600"))
            # 使用 asyncio.wait_for 设置超时
            response = await asyncio.wait_for(call_next(request), timeout=timeout)
            return response
        except asyncio.TimeoutError:
            # 请求超时，返回 503 Service Unavailable
            logger.error(f"请求处理超时: {request.url.path}")
            if request.url.path == "/api/v1/asd":
                return _asd_error_response(503, "请求处理超时")
            return JSONResponse(status_code=503, content={"detail": "请求处理超时"})
        except Exception as e:
            # 其他异常，记录错误并返回 500 Internal Server Error
            logger.error(f"请求处理异常: {str(e)}\n{traceback.format_exc()}")
            if request.url.path == "/api/v1/asd":
                return _asd_error_response(500, "服务器内部错误")
            return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})

# 模型加载函数（同步）
def load_sensevoice_model():
    """加载 SenseVoice ASR 模型"""
    model_dir = "iic/SenseVoiceSmall"
    device = os.getenv("SENSEVOICE_DEVICE", "cpu")
    model, model_kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
    model.eval()
    return (model, model_kwargs)

def load_vad_model():
    """加载 VAD 模型"""
    return Wav2Vec2VAD()

def load_pri_model():
    """加载 PRI 模型"""
    device = os.getenv("SENSEVOICE_DEVICE", "cpu")
    return AutoModel(model="paraformer-zh", device=device)

def _normalize_asd_language(language):
    if isinstance(language, Language):
        return language.value
    if language is None:
        return "auto"
    return str(language).strip() or "auto"

def _resolve_asd_model(language, model):
    normalized_language = _normalize_asd_language(language)
    normalized_model = (model or "").strip().lower()

    if normalized_language in {"", "auto"}:
        if normalized_model not in {"paraformer", "whisperx"}:
            raise HTTPException(
                status_code=422,
                detail="当 language 为 auto 或未传时，必须显式指定 model=paraformer 或 model=whisperx",
            )
        return normalized_model, "zh" if normalized_model == "paraformer" else "auto"

    if normalized_language == "zh":
        return "paraformer", "zh"

    return "whisperx", normalized_language

def process_asd(file_data, key, language, model):
    """ASD 处理函数：按语言路由到 paraformer 或 whisperx。"""
    try:
        start_time = time.time()
        request_id = uuid4().hex
        requested_language = _normalize_asd_language(language)
        requested_model = (model or "").strip().lower() or None
        backend, effective_language = _resolve_asd_model(language, model)

        with BytesIO(file_data) as file_io:
            audio, sr = sf.read(file_io)

        srt_text, full_text, segments = asd_infer(
            (sr, audio),
            effective_language,
            model_type=backend,
            return_segments=True,
        )

        cleaned_text = re.sub(regex, "", full_text, 0, re.MULTILINE)
        normalized_segments = []
        for item in segments or []:
            if not isinstance(item, dict):
                continue
            segment = dict(item)
            segment["key"] = key
            normalized_segments.append(segment)

        logger.info(
            "ASD 处理完成: key=%s, language=%s, model=%s, segments=%s, 耗时=%.2f秒",
            key,
            effective_language,
            backend,
            len(normalized_segments),
            time.time() - start_time,
        )
        return {
            "code": 0,
            "message": "识别成功",
            "data": {
                "request": {
                    "request_id": request_id,
                    "key": key,
                    "requested_language": requested_language,
                    "resolved_language": effective_language,
                    "requested_model": requested_model,
                    "resolved_model": backend,
                },
                "audio": {
                    "sample_rate": sr,
                    "duration_seconds": len(audio) / sr,
                    "size_bytes": len(file_data),
                },
                "timing": {
                    "processing_ms": int((time.time() - start_time) * 1000),
                },
                "result": {
                    "segments": normalized_segments,
                    "text": full_text,
                    "raw_text": full_text,
                    "clean_text": cleaned_text,
                    "srt": srt_text,
                },
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"ASD 处理时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ASD 处理失败: {str(e)}")

# 使用 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行
    global model_manager_asr, model_manager_vad, model_manager_pri
    
    try:
        pid = os.getpid()
        logger.info(f"Worker {pid}: 初始化模型管理器（延迟加载模式）")
        logger.info(f"Worker {pid}: CPU 核心数: {cpu_count}, 最大并发: {MAX_CONCURRENT}")
        logger.info(f"Worker {pid}: 模型空闲超时: {MODEL_IDLE_TIMEOUT}s, 自动卸载: {ENABLE_AUTO_UNLOAD}")
        
        # 创建模型管理器（不立即加载模型）
        model_manager_asr = LazyModelManager(
            name="SenseVoice-ASR",
            load_func=load_sensevoice_model,
            idle_timeout=MODEL_IDLE_TIMEOUT,
            enable_auto_unload=False # ASR 太常用，禁用自动卸载
        )
        
        model_manager_vad = LazyModelManager(
            name="Wav2Vec2-VAD",
            load_func=load_vad_model,
            idle_timeout=MODEL_IDLE_TIMEOUT,
            enable_auto_unload=ENABLE_AUTO_UNLOAD
        )
        
        model_manager_pri = LazyModelManager(
            name="FunASR-PRI",
            load_func=load_pri_model,
            idle_timeout=MODEL_IDLE_TIMEOUT,
            enable_auto_unload=ENABLE_AUTO_UNLOAD
        )
        
        logger.info(f"Worker {pid}: 模型管理器初始化完成（模型将在首次请求时加载）")
        
        yield  # 这里是应用运行期间
    except Exception as e:
        logger.exception(f"应用启动异常: {str(e)}")
        raise
    finally:
        # 应用关闭时执行的清理代码
        logger.info(f"Worker {pid}: 正在清理资源...")
        if model_manager_asr:
            await model_manager_asr.force_unload()
        if model_manager_vad:
            await model_manager_vad.force_unload()
        if model_manager_pri:
            await model_manager_pri.force_unload()
        executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)
# 添加超时中间件
app.add_middleware(TimeoutMiddleware)

# 使用私有 CUDA 健康检查插件，统一 /health 行为
# 使用路径前缀跟踪 /api/v1/* 推理接口；/health 与 /docs 访问日志将由插件抑制
setup_cuda_health(
    app,
    path="/health",
    ready_predicate=lambda: True,  # 延迟加载模式下，服务始终就绪
)

regex = r"<\|.*\|>"

# 使用 PyTorch JIT 优化音频处理函数
@torch.jit.script
def preprocess_audio(audio_data: torch.Tensor) -> torch.Tensor:
    # 对音频数据进行预处理
    return audio_data.mean(0) if audio_data.dim() > 1 else audio_data

# 处理单个音频文件的函数
def process_audio(file_data):
    try:
        start_time = time.time()

        with BytesIO(file_data) as file_io:
            data_or_path_or_list, fs = torchaudio.load(file_io)

        # 使用 JIT 优化的预处理函数
        data_or_path_or_list = preprocess_audio(data_or_path_or_list).to(torch.float32)
        duration_seconds = (
            float(data_or_path_or_list.numel()) / float(fs) if fs > 0 else 0.0
        )

        if fs != TARGET_FS:
            if duration_seconds >= LONG_AUDIO_THRESHOLD_SECONDS:
                logger.info(
                    "长音频预处理降级: duration=%.2fs, sample_rate=%s -> %s",
                    duration_seconds,
                    fs,
                    TARGET_FS,
                )
            else:
                logger.info("音频重采样: sample_rate=%s -> %s", fs, TARGET_FS)
            data_or_path_or_list = resample_audio(data_or_path_or_list, fs, TARGET_FS)
            fs = TARGET_FS

        logger.debug(f"音频处理完成，耗时: {time.time() - start_time:.2f}秒")
        return data_or_path_or_list, fs
    except Exception as e:
        logger.error(f"处理音频文件时出错: {str(e)}\n{traceback.format_exc()}")
        return None, 0

def background_process_asr(model_tuple, file_data, key, lang):
    """ASR 处理函数（在线程池中执行）"""
    try:
        model, model_kwargs = model_tuple
        audio, audio_fs = process_audio(file_data)
        if audio is None:
            return {"result": []}
        
        # 使用模型进行推理
        with torch.set_grad_enabled(False):  # 禁用梯度计算提高性能
            res = model.inference(
                data_in=[audio],
                language=lang,
                use_itn=True,
                ban_emo_unk=True,
                key=[key],
                fs=audio_fs,
                **model_kwargs,
            )
        
        if len(res) == 0:
            return {"result": []}
        
        # 后处理结果
        for it in res[0]:
            it["raw_text"] = it["text"]
            it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
            it["text"] = rich_transcription_postprocess(it["text"])
        
        return {"result": res[0]}
    except Exception as e:
        logger.exception(f"ASR 处理时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ASR 处理失败: {str(e)}")

@app.post("/api/v1/asr", response_model=ASRResponse)
async def turn_audio_to_text(
    files: Annotated[bytes, File(description="wav or mp3 audio in 16KHz")],
    key: Annotated[str, Form(description="name of audio file")] = "wav_file_tmp_name",
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
) -> Dict[str, List[Dict[str, Any]]]:
    global model_manager_asr
    
    if lang == "":
        lang = "auto"
    if key == "":
        key = "wav_file_tmp_name"
    
    logger.debug(f"收到 ASR 请求: key={key}, lang={lang}, 文件大小={len(files)} 字节")
    start_time = time.time()
    
    try:
        # 使用并发限流
        async with semaphore:
            # 获取模型（首次调用时加载）
            model_tuple = await model_manager_asr.get_model()
            
            # 在线程池中执行推理
            result = await asyncio.to_thread(
                background_process_asr, model_tuple, files, key, lang
            )
        
        process_time = time.time() - start_time
        logger.info(f"ASR 请求处理完成: key={key}, 耗时={process_time:.2f}秒")
        return result
    except Exception as e:
        logger.exception(f"ASR 处理异常: key={key}, 错误={str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")


@app.post("/api/v1/asd", response_model=ASDEnvelope)
async def turn_audio_to_speaker(
    files: Annotated[bytes, File(description="wav or mp3 audio in 16KHz")],
    key: Annotated[str, Form(description="name of audio file")] = "wav_file_tmp_name",
    lang: Annotated[str, Form(description="language of audio content")] = "auto",
    model: Annotated[str | None, Form(description="model backend: paraformer or whisperx")] = None,
) -> Dict[str, Any]:
    global semaphore

    if lang == "":
        lang = "auto"
    if key == "":
        key = "wav_file_tmp_name"

    logger.debug(
        "收到 ASD 请求: key=%s, lang=%s, model=%s, 文件大小=%s 字节",
        key,
        lang,
        model,
        len(files),
    )
    start_time = time.time()

    try:
        async with semaphore:
            result = await asyncio.to_thread(
                process_asd,
                files,
                key,
                lang,
                model,
            )

        process_time = time.time() - start_time
        logger.info(f"ASD 请求处理完成: key={key}, 耗时={process_time:.2f}秒")
        return result
    except HTTPException as e:
        logger.exception(f"ASD 处理异常: key={key}, 错误={str(e.detail)}")
        return _asd_error_response(e.status_code, str(e.detail))
    except Exception as e:
        logger.exception(f"ASD 处理异常: key={key}, 错误={str(e)}")
        return _asd_error_response(500, f"处理请求失败: {str(e)}")

# 使用 PyTorch JIT 优化音频重采样函数
@torch.jit.script
def resample_audio(data: torch.Tensor, orig_sr: int, target_sr: int = 16000) -> torch.Tensor:
    if orig_sr == target_sr:
        return data
    # 将 numpy 数组转换为 torch.Tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    # 计算新的采样点数
    number_of_samples = int(len(data) * float(target_sr) / orig_sr)
    # 使用 PyTorch 的重采样函数
    resampler = torch.nn.functional.interpolate
    # 添加批次维度和通道维度
    data = data.view(1, 1, -1)
    # 重采样
    data = resampler(data, size=number_of_samples, mode='linear', align_corners=False)
    # 移除批次维度和通道维度
    return data.view(-1)

# VAD 处理函数，用于线程池
def process_vad(vad_processor, file_data):
    """VAD 处理函数（在线程池中执行）"""
    try:
        start_time = time.time()
        
        data, sr = sf.read(BytesIO(file_data))
        # 转换为 PyTorch Tensor 并使用 JIT 优化的重采样函数
        data_tensor = torch.tensor(data, dtype=torch.float32)
        if len(data_tensor.shape) > 1:
            data_tensor = data_tensor.mean(dim=1)
        
        # 使用优化的重采样函数
        resampled_data = resample_audio(data_tensor, sr)
        # 转回 numpy 数组
        resampled_data = resampled_data.numpy()
        
        # 使用 VAD 处理器
        with torch.set_grad_enabled(False):  # 禁用梯度计算提高性能
            vad_data = vad_processor.process(resampled_data, raw=True)
        
        logger.debug(f"VAD 处理完成，耗时: {time.time() - start_time:.2f}秒")
        return vad_data
    except Exception as e:
        logger.exception(f"VAD 处理时出错: {str(e)}")
        return None

@app.post("/api/v1/vad", response_model=VADResponse)
async def get_vad_from_file(
    file: Annotated[bytes, File(description="wav or mp3 audios")],
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    global model_manager_vad
    
    logger.debug(f"收到 VAD 请求: 文件大小={len(file)} 字节")
    start_time = time.time()
    
    try:
        # 使用并发限流
        async with semaphore:
            # 获取模型（首次调用时加载）
            vad_processor = await model_manager_vad.get_model()
            
            # 在线程池中执行推理
            vad_data = await asyncio.to_thread(process_vad, vad_processor, file)
        
        if vad_data is None:
            raise HTTPException(status_code=500, detail="VAD 处理失败")

        process_time = time.time() - start_time
        logger.info(f"VAD 请求处理完成: 大小={len(file)} 耗时={process_time:.2f}秒")
        
    except Exception as e:
        logger.exception(f"VAD 处理异常: 错误={str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")

    return {
        "result": {
            "v": vad_data.get("Valence", 0),
            "a": vad_data.get("Arousal", 0),
            "d": vad_data.get("Dominance", 0),
            "raw": vad_data.get("raw", []),
        }
    }

# PRI 处理函数
def process_pri(pri_model, file_data):
    """PRI 处理函数（在线程池中执行）"""
    try:
        start_time = time.time()
        
        # 使用 with 语句确保资源正确释放
        with BytesIO(file_data) as file_io:
            audio, sr = sf.read(file_io)
        
        # 使用 torch.no_grad() 上下文管理器禁用梯度计算
        with torch.no_grad():
            # 创建 PriFile 实例，传入模型
            pri_data = PriFile((audio, sr), model=pri_model)
        
        logger.debug(f"PRI 处理完成，大小={len(file_data)} 耗时: {time.time() - start_time:.2f}秒")
        return {
            "mean_pri": pri_data.mean_measure(),
            "max_pri": pri_data.max_measure(),
            "rate": pri_data.rate,
            "loundness": {
                "itgr": pri_data.loundness["itgr"],
                "max": pri_data.loundness["max"],
            },
            "pitches": {
                "mean": pri_data.pitches["mean"],
                "max": pri_data.pitches["max"],
            },
        }
    except Exception as e:
        logger.exception(f"PRI 处理时出错: {str(e)}")
        return None

@app.post("/api/v1/pri", response_model=PRIResponse)
async def get_pri_from_file(
    file: Annotated[bytes, File(description="wav or mp3 audios")],
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    global model_manager_pri
    
    logger.debug(f"收到 PRI 请求: 文件大小={len(file)} 字节")
    start_time = time.time()
    
    try:
        # 使用并发限流
        async with semaphore:
            # 获取模型（首次调用时加载）
            pri_model = await model_manager_pri.get_model()
            
            # 在线程池中执行推理
            pri_data = await asyncio.to_thread(process_pri, pri_model, file)
        
        if pri_data is None:
            raise HTTPException(status_code=500, detail="PRI 处理失败")

        process_time = time.time() - start_time
        logger.info(f"PRI 请求处理完成: 大小={len(file)} 耗时={process_time:.2f}秒")
        
    except Exception as e:
        logger.exception(f"PRI 处理异常: 错误={str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")

    return {
        "result": pri_data
    }

# ========== 管理端点 ==========

@app.post("/api/v1/warmup")
async def warmup():
    """
    预热端点：提前加载所有模型
    建议在批量任务开始前调用，避免首次请求的冷启动延迟
    """
    global model_manager_asr, model_manager_vad, model_manager_pri
    
    logger.info("🔥 开始预热所有模型...")
    start_time = time.time()
    
    try:
        # 并行加载所有模型
        results = await asyncio.gather(
            model_manager_asr.warmup(),
            model_manager_vad.warmup(),
            model_manager_pri.warmup(),
            asyncio.to_thread(load_asd_model, "zh", model_type="paraformer"),
            asyncio.to_thread(load_asd_model, "en", model_type="whisperx"),
            return_exceptions=True
        )
        
        # 检查是否有加载失败
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            logger.error(f"❌ 预热失败: {errors}")
            raise HTTPException(status_code=500, detail=f"模型预热失败: {errors[0]}")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ 所有模型预热完成 (耗时: {elapsed:.1f}s)")
        
        return {
            "status": "ready",
            "elapsed": elapsed,
            "models": {
                "asr": model_manager_asr.is_loaded(),
                "vad": model_manager_vad.is_loaded(),
                "pri": model_manager_pri.is_loaded(),
                "asd": bool(get_asd_model_cache_stats()["paraformer"] or get_asd_model_cache_stats()["whisperx"]),
            }
        }
    except Exception as e:
        logger.exception(f"❌ 预热异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预热失败: {str(e)}")


@app.post("/api/v1/unload")
async def force_unload():
    """
    强制卸载端点：立即卸载所有模型释放内存
    """
    global model_manager_asr, model_manager_vad, model_manager_pri
    
    logger.info("🗑️ 开始强制卸载所有模型...")
    start_time = time.time()
    
    try:
        await asyncio.gather(
            model_manager_asr.force_unload(),
            model_manager_vad.force_unload(),
            model_manager_pri.force_unload(),
        )
        clear_asd_model_cache()
        
        elapsed = time.time() - start_time
        logger.info(f"✅ 所有模型卸载完成 (耗时: {elapsed:.1f}s)")
        
        return {
            "status": "unloaded",
            "elapsed": elapsed,
        }
    except Exception as e:
        logger.exception(f"❌ 卸载异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"卸载失败: {str(e)}")


@app.get("/api/v1/status")
async def get_status():
    """
    状态端点：查看模型加载状态和统计信息
    """
    global model_manager_asr, model_manager_vad, model_manager_pri
    
    return {
        "models": {
            "asr": model_manager_asr.get_stats() if model_manager_asr else None,
            "vad": model_manager_vad.get_stats() if model_manager_vad else None,
            "pri": model_manager_pri.get_stats() if model_manager_pri else None,
            "asd": get_asd_model_cache_stats(),
        },
        "config": {
            "max_concurrent": MAX_CONCURRENT,
            "max_workers": MAX_WORKERS,
            "model_idle_timeout": MODEL_IDLE_TIMEOUT,
            "auto_unload_enabled": ENABLE_AUTO_UNLOAD,
            "cpu_count": cpu_count,
        }
    }

# 将 Gradio 应用挂载到根路径，放在最后以确保 /api/* 等更具体路由优先生效
# gr.mount_gradio_app(app, create_gradio_app(default_language="auto"), path="/")
