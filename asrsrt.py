# coding=utf-8


import logging

import os

import gradio as gr
import librosa
import numpy as np
from funasr import AutoModel

from utils.subtitle_utils import generate_srt

# 全局缓存 FunASR 模型（按语言缓存），避免重复加载
_funasr_models = {}

def load_model(language: str = "zh"):
    """按语言懒加载 FunASR 模型，并尽可能利用多核 CPU。"""
    lang_key = "en" if language == "en" else "zh"
    if lang_key in _funasr_models:
        return _funasr_models[lang_key]

    cpu_count = os.cpu_count() or 1
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
    _funasr_models[lang_key] = model
    logging.info(f"FunASR model loaded with ncpu={cpu_count}, language={lang_key}")
    return _funasr_models[lang_key]


def preload_models(langs):
    """预加载指定语言的模型列表。仅在独立运行（launch）场景使用。"""
    if not isinstance(langs, (list, tuple)):
        langs = [langs]
    loaded = []
    for lg in langs:
        try:
            m = load_model("en" if lg == "en" else "zh")
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


def infer(audio_data, language):
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

    logging.info("Input audio ready. length: %.2f seconds.", len(data) / sr)

    m = load_model("en" if language == "en" else "zh")
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
                        label="识别语言（似乎只支持中文和英文）",
                    )
                fn_button = gr.Button("开始识别", variant="primary")
        with gr.Row():
            asr_outputs = gr.Textbox(label="ASR 结果", lines=30, show_copy_button=True)
            srt_outputs = gr.Textbox(label="SRT 结果", lines=30, show_copy_button=True)

        fn_button.click(
            infer,
            inputs=[audio_inputs, language_inputs],
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
