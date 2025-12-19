# coding=utf-8


import logging

import os

import gradio as gr
import librosa
import numpy as np
from funasr import AutoModel

from utils.subtitle_utils import generate_srt

# 全局缓存 FunASR 模型（按语言和类型缓存），避免重复加载
_funasr_models = {}

def load_model(language: str = "zh", model_type: str = "paraformer"):
    """按语言懒加载 FunASR 模型，并尽可能利用多核 CPU。
    
    Args:
        language: 语言代码 "zh" 或 "en" (对于 Whisper 仅作为缓存键的一部分)
        model_type: 模型类型 "paraformer" (默认) 或 "whisper"
    """
    lang_key = "en" if language == "en" else "zh"
    cache_key = f"{model_type}_{lang_key}"
    
    if cache_key in _funasr_models:
        return _funasr_models[cache_key]

    cpu_count = os.cpu_count() or 1
    
    if model_type == "whisper":
        # Whisper 模型加载 (使用 openai-whisper)
        # 注意：这里使用 turbo 版本以获得较好的速度/精度平衡
        logging.info(f"Loading Whisper model for language={lang_key}...")
        try:
            model = AutoModel(
                model="Whisper-large-v3-turbo",
                hub="openai",
                device=os.getenv("SENSEVOICE_DEVICE", "cpu"),
                ncpu=cpu_count,
            )
        except Exception as e:
            logging.error(f"Failed to load Whisper model. Ensure 'openai-whisper' is installed. Error: {e}")
            raise
    else:
        # 默认 Paraformer 模型加载
        if lang_key == "zh":
            model = AutoModel(
                model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                device=os.getenv("SENSEVOICE_DEVICE", "cpu"),
                trust_remote_code=True,
                ncpu=cpu_count,
            )
        else:
            model = AutoModel(
                model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                device=os.getenv("SENSEVOICE_DEVICE", "cpu"),
                trust_remote_code=True,
                ncpu=cpu_count,
            )
            
    _funasr_models[cache_key] = model
    logging.info(f"FunASR model loaded with ncpu={cpu_count}, language={lang_key}, type={model_type}")
    return _funasr_models[cache_key]


def preload_models(langs):
    """预加载指定语言的模型列表。仅在独立运行（launch）场景使用。"""
    if not isinstance(langs, (list, tuple)):
        langs = [langs]
    loaded = []
    for lg in langs:
        try:
            # 默认预加载 Paraformer，如果需要预加载 Whisper 可以扩展环境变量解析逻辑
            m = load_model("en" if lg == "en" else "zh", model_type="paraformer")
            loaded.append(lg)
        except Exception as e:
            logging.exception(f"Preload model failed for language={lg}: {e}")
    if loaded:
        logging.info(f"Preloaded UI models for languages: {','.join(loaded)}")


def convert_pcm_to_float(data: np.ndarray):
    """将常见 PCM 类型统一转换为 float64 [-1, 1] 区间。"""
    if data.dtype == np.float64:
        return data
    if data.dtype == np.float32:
        return data.astype(np.float64)
    if data.dtype == np.int16:
        return data.astype(np.float64) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float64) / 2147483648.0
    if data.dtype == np.int8:
        # 8-bit PCM 通常是无符号 [0,255]，中心在 128
        data = data.astype(np.float64) - 128.0
        return data / 128.0
    raise ValueError(f"Unsupported data type: {data.dtype}")


def infer(audio_data, language, model_type="paraformer"):
    sr, data = audio_data

    # 统一为 float64
    data = convert_pcm_to_float(data)

    # 多通道下混为单通道
    if data.ndim == 2:
        try:
            ch_axis = int(np.argmin(data.shape))
            logging.warning("Input wav shape: %s, downmix along axis %d to mono.", str(data.shape), ch_axis)
            data = data.mean(axis=ch_axis)
        except Exception as e:
            logging.exception("Downmix failed with shape %s: %s", str(data.shape), e)
            if data.shape[-1] >= 1:
                data = data[..., 0]
            else:
                raise

    # 重采样至 16k
    if sr != 16000:
        logging.warning("Resampling from %d Hz to 16000 Hz", sr)
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000

    logging.info(f"Input audio ready. length: {len(data) / sr:.2f} seconds. Model: {model_type}")

    m = load_model("en" if language == "en" else "zh", model_type=model_type)
    
    if model_type == "whisper":
        # Whisper 推理逻辑
        decoding_options = {
            "task": "transcribe",
            "language": None if language == "auto" else language, # Whisper auto detection uses None
            "beam_size": 5,
            "fp16": False, # CPU 可能会警告 fp16，视环境而定
            "without_timestamps": False,
        }
        
        # Whisper expect input as array or path. 
        # FunASR AutoModel wrapper for Whisper passes args to model.generate
        try:
            # 注意：FunASR 对 Whisper 的封装可能有些不同，通常它需要 batch_size_s=0 来处理整段音频
            rec_result = m.generate(
                input=data, 
                batch_size_s=0, 
                DecodingOptions=decoding_options
            )
            
            # 适配 Whisper 结果到 generate_srt 格式
            # 假设 rec_result 包含 OpenAI Whisper 的原生输出结构或 FunASR 包装结构
            # 通常 FunASR Whisper wrapper 返回的是 list of dict 或 dict
            # 这里我们需要 defensive coding
            
            sentence_info = []
            full_text = ""
            
            # 检查结果结构
            results = rec_result if isinstance(rec_result, list) else [rec_result]
            
            for res in results:
                if isinstance(res, dict):
                    # 尝试提取 text
                    text = res.get("text", "")
                    full_text += text
                    
                    # 尝试提取 segments 用于 SRT
                    segments = res.get("segments", [])
                    # 如果 segments 存在，转换为 sentence_info 格式
                    # sentence_info element: {'text': str, 'timestamp': [[start_ms, end_ms]]}
                    for seg in segments:
                        seg_text = seg.get("text", "")
                        start = seg.get("start", 0.0) * 1000 # convert to ms
                        end = seg.get("end", 0.0) * 1000
                        sentence_info.append({
                            "text": seg_text,
                            "timestamp": [[start, end]] 
                        })
            
            if not sentence_info and full_text:
                # 如果没有 segments 但有文本（极少情况），造一个假的 timestamp
                duration_ms = (len(data) / sr) * 1000
                sentence_info.append({
                    "text": full_text,
                    "timestamp": [[0, duration_ms]]
                })
                
            res_srt = generate_srt(sentence_info)
            return res_srt, full_text
            
        except Exception as e:
            logging.exception(f"Whisper inference failed: {e}")
            raise
            
    else:
        # Paraformer 推理逻辑
        rec_result = m.generate(
            data,
            return_spk_res=True,
            return_raw_text=True,
            is_final=True,
            pred_timestamp=language == "en",
            en_post_proc=language == "en",
            cache={},
        )

        res_srt = generate_srt(rec_result[0]["sentence_info"])
        asr_result = rec_result[0]["text"]
        return res_srt, asr_result


def create_gradio_app(default_language: str = "auto") -> gr.Blocks:
    """创建并返回 Gradio Blocks 应用，用于挂载到 FastAPI。"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column():
                audio_inputs = gr.Audio(label="上传音频或使用麦克风")
            with gr.Column():
                with gr.Accordion("配置"):
                    model_type_input = gr.Dropdown(
                        choices=["paraformer", "whisper"],
                        value="paraformer",
                        label="模型类型 (Paraformer: 速度快/中文强; Whisper: 多语言强)"
                    )
                    language_inputs = gr.Dropdown(
                        choices=[
                            "auto",
                            "zh",
                            "en",
                            "yue",
                            "ja",
                            "ko",
                            "nospeech",
                        ],
                        value=default_language,
                        label="识别语言（Paraformer主要支持中英，Whisper支持更多）",
                    )
                fn_button = gr.Button("开始识别", variant="primary")
        with gr.Row():
            asr_outputs = gr.Textbox(label="ASR 结果", lines=30, show_copy_button=True)
            srt_outputs = gr.Textbox(label="SRT 结果", lines=30, show_copy_button=True)

        fn_button.click(
            infer,
            inputs=[audio_inputs, language_inputs, model_type_input],
            outputs=[srt_outputs, asr_outputs],
        )

    return demo
def launch():
    """独立运行时：预加载模型后启动本地 Gradio 服务。"""
    preload = os.getenv("SENSEVOICE_UI_PRELOAD_LANGS", "zh")
    langs = [x.strip() for x in preload.split(",") if x.strip()]
    preload_models(langs)
    demo = create_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=8000, share=False)


if __name__ == "__main__":
    launch()
