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
 
 
def convert_pcm_to_float(data: np.ndarray):
    if data.dtype == np.float64:
        return data
    elif data.dtype == np.float32:
        return data.astype(np.float64)
    elif data.dtype == np.int16:
        bit_depth = 16
    elif data.dtype == np.int32:
        bit_depth = 32
    elif data.dtype == np.int8:
        bit_depth = 8
    else:
        raise ValueError("Unsupported audio data type")

    max_int_value = float(2 ** (bit_depth - 1))
    if bit_depth == 8:
        data = data - 128
    return data.astype(np.float64) / max_int_value

def infer(audio_data, language):
    sr, data = audio_data

    # Convert to float64 consistently (includes data type checking)
    data = convert_pcm_to_float(data)

    # Handle multi-channel audio BEFORE resampling: downmix to mono (A + downmix)
    if data.ndim == 2:
        try:
            # Heuristic: treat the smaller dimension as channels for downmix
            ch_axis = int(np.argmin(data.shape))
            logging.warning(
                "Input wav shape: {}, downmix along axis {} to mono.".format(
                    data.shape, ch_axis
                )
            )
            data = data.mean(axis=ch_axis)
        except Exception as e:
            logging.exception(
                "Downmix failed with shape {}: {}".format(data.shape, e)
            )
            # Fallback to keep the first channel if possible (assume (T, C))
            if data.shape[-1] >= 1:
                data = data[..., 0]
            else:
                raise

    # Resample to 16k after ensuring mono 1-D data
    # assert sr == 16000, "16kHz sample rate required, {} given.".format(sr)
    if sr != 16000:
        logging.warning("Resampling from {} Hz to 16000 Hz".format(sr))
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000

    logging.info("Input audio ready. length: {} seconds.".format(len(data) / sr))

    m = load_model("en" if language == "en" else "zh")
    rec_result = m.generate(
        data,
        return_spk_res=True,
        return_raw_text=True,
        is_final=True,
        # output_dir=output_dir,
        # hotword=hotwords,
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
                audio_inputs = gr.Audio(
                    label="Upload audio or use the microphone"
                )
            with gr.Column():
                with gr.Accordion("Configuration"):
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
                        label="Language",
                    )
                fn_button = gr.Button("Start", variant="primary")
        with gr.Row():
            srt_outputs = gr.Textbox(label="SRT Results", lines=20)
            asr_outputs = gr.Textbox(label="ASR Results", lines=20)
 
        fn_button.click(
            infer,
            inputs=[audio_inputs, language_inputs],
            outputs=[srt_outputs, asr_outputs],
        )
 
    return demo
 
 
def launch():
    demo = create_gradio_app()
    demo.launch()


if __name__ == "__main__":
    launch()
