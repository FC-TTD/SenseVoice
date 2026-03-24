# coding=utf-8

import logging
import os
from pathlib import Path
import subprocess
import numpy as np
import librosa
import soundfile as sf
from io import BytesIO

import gradio as gr
import requests


logger = logging.getLogger("asrsrt")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


API_BASE_URL = os.getenv("SENSEVOICE_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ASD_ENDPOINT = f"{API_BASE_URL}/api/v1/asd"
REQUEST_TIMEOUT = int(os.getenv("SENSEVOICE_API_TIMEOUT", "600"))
TARGET_SAMPLE_RATE = 16000

LANGUAGE_CHOICES = [
    "auto",
    "zh",
    "en",
    "yue",
    "ja",
    "ko",
    "de",
    "fr",
    "es",
    "ru",
    "it",
    "pt",
    "tr",
    "nl",
    "pl",
    "uk",
    "vi",
    "th",
    "id",
    "ms",
    "hi",
    "ar",
    "fa",
    "he",
    "cs",
    "sv",
    "da",
    "no",
    "fi",
    "hu",
    "ro",
]


def _normalize_language(language: str):
    normalized = (language or "auto").strip()
    return normalized or "auto"


def _normalize_model(model_type: str, language: str):
    normalized_model = (model_type or "").strip().lower()
    normalized_language = _normalize_language(language)
    if normalized_language == "auto" and normalized_model not in {"paraformer", "whisperx"}:
        return None
    return normalized_model or None


def _normalize_audio_array(sample_rate: int, audio_data):
    audio_array = np.asarray(audio_data)
    if audio_array.ndim == 0:
        raise ValueError("音频数据为空")

    if audio_array.ndim == 2:
        if audio_array.shape[0] <= 8 and audio_array.shape[0] < audio_array.shape[1]:
            audio_array = audio_array.mean(axis=0)
        else:
            audio_array = audio_array.mean(axis=1)
    elif audio_array.ndim > 2:
        raise ValueError(f"暂不支持当前音频维度: {audio_array.shape}")

    if np.issubdtype(audio_array.dtype, np.integer):
        dtype_info = np.iinfo(audio_array.dtype)
        scale = max(abs(dtype_info.min), dtype_info.max)
        audio_array = audio_array.astype(np.float32) / float(scale)
    else:
        audio_array = audio_array.astype(np.float32)

    if int(sample_rate) != TARGET_SAMPLE_RATE:
        audio_array = librosa.resample(audio_array, orig_sr=int(sample_rate), target_sr=TARGET_SAMPLE_RATE)
    return np.ascontiguousarray(audio_array, dtype=np.float32), TARGET_SAMPLE_RATE


def _encode_wav_bytes(sample_rate: int, audio_data, filename: str):
    audio_array, normalized_sr = _normalize_audio_array(sample_rate, audio_data)
    buffer = BytesIO()
    sf.write(buffer, audio_array, int(normalized_sr), format="WAV", subtype="PCM_16")
    return buffer.getvalue(), filename


def _ffmpeg_decode_to_wav_bytes(path: Path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            path.as_posix(),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-f",
            "wav",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        error_text = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(error_text or "ffmpeg decode failed")
    return result.stdout, path.name


def _read_audio_path_bytes(path_value):
    path = Path(path_value)
    if not path.exists():
        raise ValueError(f"音频临时文件不存在: {path}")
    try:
        return _ffmpeg_decode_to_wav_bytes(path)
    except Exception as exc:
        logger.warning("ffmpeg 解码失败，回退 librosa: path=%s error=%s", path, exc)
        audio_data, sample_rate = librosa.load(path.as_posix(), sr=TARGET_SAMPLE_RATE, mono=True)
        return _encode_wav_bytes(sample_rate, audio_data, path.name)


def _read_audio_bytes(audio_input):
    if audio_input is None:
        raise ValueError("请先上传音频")

    if isinstance(audio_input, str):
        return _read_audio_path_bytes(audio_input)

    if isinstance(audio_input, (tuple, list)) and len(audio_input) == 2:
        sample_rate, audio_data = audio_input
        return _encode_wav_bytes(sample_rate, audio_data, "audio.wav")

    if isinstance(audio_input, dict):
        path = audio_input.get("path")
        if path:
            return _read_audio_path_bytes(path)
        if "sample_rate" in audio_input and "data" in audio_input:
            return _encode_wav_bytes(audio_input.get("sample_rate"), audio_input.get("data"), "audio.wav")

    raise ValueError("暂不支持当前音频输入格式，请重新上传文件")


def infer(audio_input, language, model_type="paraformer", diarize=None):
    del diarize

    audio_bytes, default_key = _read_audio_bytes(audio_input)
    normalized_language = _normalize_language(language)
    normalized_model = _normalize_model(model_type, normalized_language)

    data = {
        "key": default_key,
        "lang": normalized_language,
    }
    if normalized_model:
        data["model"] = normalized_model

    files = {
        "files": (default_key, audio_bytes, "audio/wav"),
    }

    logger.info(
        "Forward ASD request to API: endpoint=%s, language=%s, model=%s, key=%s",
        ASD_ENDPOINT,
        normalized_language,
        normalized_model,
        default_key,
    )
    response = requests.post(ASD_ENDPOINT, data=data, files=files, timeout=REQUEST_TIMEOUT)
    try:
        payload = response.json()
    except ValueError as e:
        logger.exception("ASD API returned non-JSON response")
        raise RuntimeError(f"ASD API 返回了非 JSON 响应: {response.text[:300]}") from e

    if not isinstance(payload, dict):
        raise RuntimeError(f"ASD API 返回格式错误: {payload}")

    if not response.ok or payload.get("code") not in {0, None}:
        detail = payload.get("message") or payload.get("detail") or payload
        raise RuntimeError(f"ASD API 调用失败: HTTP {response.status_code} - {detail}")

    data_payload = payload.get("data") or {}
    result_payload = data_payload.get("result") or {}
    result_items = result_payload.get("segments") or []
    text = result_payload.get("text") or "\n".join(
        item.get("text", "") for item in result_items if item.get("text")
    ).strip()
    return result_payload.get("srt", ""), text


def create_gradio_app(default_language: str = "auto") -> gr.Blocks:
    """创建并返回 Gradio Blocks 应用，用于挂载到 FastAPI。"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column():
                audio_inputs = gr.Audio(label="上传音频或使用麦克风", type="numpy")
            with gr.Column():
                with gr.Accordion("配置"):
                    model_type_input = gr.Dropdown(
                        choices=["paraformer", "whisperx"],
                        value="paraformer",
                        label="模型类型（由 API 实际执行）",
                    )
                    language_inputs = gr.Dropdown(
                        choices=LANGUAGE_CHOICES,
                        value=default_language,
                        allow_custom_value=True,
                        label="识别语言（可自由输入语言代码；中文走 Paraformer，其他语言通常走 WhisperX；auto 需指定模型）",
                    )
                    diarize_input = gr.Checkbox(
                        value=True,
                        label="识别说话人（由 API 侧配置决定，这里仅保留界面兼容）",
                    )
                fn_button = gr.Button("开始识别", variant="primary")
        with gr.Row():
            asr_outputs = gr.Textbox(label="ASR 结果", lines=30, show_copy_button=True)
            srt_outputs = gr.Textbox(label="SRT 结果", lines=30, show_copy_button=True)

        fn_button.click(
            infer,
            inputs=[audio_inputs, language_inputs, model_type_input, diarize_input],
            outputs=[srt_outputs, asr_outputs],
        )

    return demo


def launch():
    """独立运行时：仅启动薄客户端 Gradio 服务。"""
    logger.info("Launch asrsrt thin client, forwarding to %s", ASD_ENDPOINT)
    demo = create_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=8000, share=False)


if __name__ == "__main__":
    launch()
