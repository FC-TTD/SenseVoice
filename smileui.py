# coding=utf-8
import gradio as gr
import numpy as np
import soundfile as sf
import librosa
from utils.os import AcousticVADAnalyzer
from utils.vec import Wav2Vec2VAD


def resample_audio(audio_path, target_sample_rate=16000):
    """重采样音频文件"""
    y, sr = librosa.load(audio_path, sr=None)
    audio = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)
    sf.write(audio_path, audio, target_sample_rate)


def extract_features_from_audio(audio_path):
    feature_extractor = AcousticVADAnalyzer()
    features = feature_extractor.extract_features(audio_path)
    if features is None:
        return "特征提取失败"
    result = feature_extractor.print_analysis(features)
    return result


def extract_vec_from_audio(audio_path):
    # print("Loading audio: ", audio_path)
    vec_extractor = Wav2Vec2VAD()
    audio, sr = sf.read(audio_path)
    vec = vec_extractor.process(audio)
    return vec


def process_audio(audio_path):
    if not audio_path:
        return None
    resample_audio(audio_path)
    vec = extract_vec_from_audio(audio_path)
    result = extract_features_from_audio(audio_path)
    return f"\n声学特征分析结果:\n{vec}\n\n{result}"


def launch():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Tab("声学特征分析"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(type="filepath", label="输入")
                    analyze_btn = gr.Button("分析特征")
                with gr.Column():
                    output_text = gr.Textbox(label="分析结果", lines=20)

            analyze_btn.click(fn=process_audio, inputs=[audio_input], outputs=[output_text])
            audio_input.change(fn=process_audio, inputs=[audio_input], outputs=[output_text])
			
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch()
