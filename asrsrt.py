# coding=utf-8


import logging

import os

import gradio as gr
import librosa
import numpy as np
from funasr import AutoModel

from utils.subtitle_utils import generate_srt

# 全局缓存模型，避免重复加载
_funasr_models = {}
_whisper_models = {}


def _coerce_ms(value):
    """将秒/毫秒时间统一为毫秒整数。"""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    # Whisper segments 常见为秒级浮点；FunASR timestamp 常见为毫秒整数。
    if abs(numeric) < 1000:
        numeric *= 1000
    return int(round(numeric))


def _segments_to_sentence_info(segments):
    sentence_info = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        start = _coerce_ms(seg.get("start"))
        end = _coerce_ms(seg.get("end"))
        if start is None or end is None or end <= start:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        sentence_info.append({
            "text": text,
            "timestamp": [[start, end]],
        })
    return sentence_info


def _normalize_timestamp_list(timestamp):
    normalized = []
    for item in timestamp or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = _coerce_ms(item[0])
        end = _coerce_ms(item[1])
        if start is None or end is None or end <= start:
            continue
        normalized.append([start, end])
    return normalized


def _collect_whisper_output(rec_result, audio_duration_ms):
    results = rec_result if isinstance(rec_result, list) else [rec_result]
    sentence_info = []
    texts = []

    for res in results:
        if not isinstance(res, dict):
            continue

        text = (res.get("text") or "").strip()
        if text:
            texts.append(text)

        if isinstance(res.get("sentence_info"), list):
            for sent in res["sentence_info"]:
                if not isinstance(sent, dict):
                    continue
                sent_text = (sent.get("text") or sent.get("sentence") or "").strip()
                timestamp = _normalize_timestamp_list(sent.get("timestamp"))
                if sent_text and timestamp:
                    sentence_info.append({"text": sent_text, "timestamp": timestamp})
            continue

        segments_sentence_info = _segments_to_sentence_info(res.get("segments"))
        if segments_sentence_info:
            sentence_info.extend(segments_sentence_info)
            continue

        timestamp = _normalize_timestamp_list(res.get("timestamp"))
        if text and timestamp:
            sentence_info.append({"text": text, "timestamp": timestamp})

    full_text = "\n".join(texts).strip()
    if not sentence_info and full_text:
        sentence_info.append({
            "text": full_text,
            "timestamp": [[0, int(round(audio_duration_ms))]],
        })
    return sentence_info, full_text


def _normalize_device(device: str):
    if not device:
        return "cpu"
    device = device.lower()
    if device.startswith("cuda"):
        return "cuda"
    return device


def _load_whisper_model(language: str = "zh"):
    lang_key = "en" if language == "en" else "zh"
    device = _normalize_device(os.getenv("SENSEVOICE_DEVICE", "cpu"))
    whisper_model_name = os.getenv("SENSEVOICE_WHISPER_MODEL", "turbo")
    cache_key = f"whisper_{lang_key}_{device}_{whisper_model_name}"
    if cache_key in _whisper_models:
        return _whisper_models[cache_key]

    logging.info("Loading official openai-whisper model=%s device=%s", whisper_model_name, device)
    try:
        import whisper
    except Exception as e:
        logging.error("Failed to import openai-whisper. Error: %s", e)
        raise

    model = whisper.load_model(whisper_model_name, device=device)
    _whisper_models[cache_key] = model
    return model


def _load_paraformer_model(language: str = "zh"):
    lang_key = "en" if language == "en" else "zh"
    cache_key = f"paraformer_{lang_key}"
    if cache_key in _funasr_models:
        return _funasr_models[cache_key]

    cpu_count = os.cpu_count() or 1
    device = os.getenv("SENSEVOICE_DEVICE", "cpu")
    if lang_key == "zh":
        model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            device=device,
            trust_remote_code=True,
            ncpu=cpu_count,
        )
    else:
        model = AutoModel(
            model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            device=device,
            trust_remote_code=True,
            ncpu=cpu_count,
        )

    _funasr_models[cache_key] = model
    logging.info("FunASR model loaded with ncpu=%s language=%s type=paraformer", cpu_count, lang_key)
    return model


def load_model(language: str = "zh", model_type: str = "paraformer"):
    """按语言懒加载模型。"""
    if model_type == "whisper":
        return _load_whisper_model(language)
    return _load_paraformer_model(language)


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


def _prepare_audio(audio_data):
    sr, data = audio_data

    data = convert_pcm_to_float(data)

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

    if sr != 16000:
        logging.warning("Resampling from %d Hz to 16000 Hz", sr)
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000

    return sr, data


def _resolve_whisper_language(language: str):
    if language in {"auto", "nospeech", None}:
        return None
    return language


def _infer_whisper(model, data: np.ndarray, sr: int, language: str):
    device = _normalize_device(os.getenv("SENSEVOICE_DEVICE", "cpu"))
    rec_result = model.transcribe(
        data.astype(np.float32),
        task="transcribe",
        language=_resolve_whisper_language(language),
        fp16=device == "cuda",
        word_timestamps=False,
        verbose=False,
    )

    audio_duration_ms = len(data) / sr * 1000
    sentence_info, full_text = _collect_whisper_output(rec_result, audio_duration_ms)
    logging.info(
        "Whisper parsed result: segments=%d, sentence_info=%d",
        len(rec_result.get("segments", [])) if isinstance(rec_result, dict) else 0,
        len(sentence_info),
    )
    return generate_srt(sentence_info), full_text


def infer(audio_data, language, model_type="paraformer"):
    sr, data = _prepare_audio(audio_data)

    logging.info(f"Input audio ready. length: {len(data) / sr:.2f} seconds. Model: {model_type}")

    m = load_model("en" if language == "en" else "zh", model_type=model_type)
    
    if model_type == "whisper":
        try:
            return _infer_whisper(m, data, sr, language)
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
