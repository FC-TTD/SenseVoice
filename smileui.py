# coding=utf-8
import gradio as gr
import numpy as np
from utils.os import AcousticVADAnalyzer

def extract_features_from_audio(audio_path):
    feature_extractor = AcousticVADAnalyzer()
    features = feature_extractor.extract_features(audio_path)
    if features is None:
        return "特征提取失败"
    
    # 格式化输出结果
    result = "1. VAD情感维度分析:\n"
    result += f"   情感效价 (Valence): {int((features['valence'] + 1) * 49.5)}  {_get_valence_description(features['valence'])}\n"
    result += f"   唤醒度 (Arousal): {int((features['arousal'] + 1) * 49.5)}  {_get_arousal_description(features['arousal'])}\n"
    result += f"   支配度 (Dominance): {int((features['dominance'] + 1) * 49.5)}  {_get_dominance_description(features['dominance'])}\n\n"
    
    result += "2. 基础声学特征:\n"
    result += "   RMS能量:\n"
    result += f"   - 均值: {features['rms_mean']:.2f}\n"
    result += f"   - 标准差: {features['rms_std']:.2f}\n\n"
    
    result += "   音高特征:\n"
    result += f"   - 均值: {features['f0_mean']:.2f} Hz\n"
    result += f"   - 标准差: {features['f0_std']:.2f}\n"
    result += f"   - 范围: {features['f0_range']:.2f}\n\n"
    
    result += "   频谱特征:\n"
    result += f"   - 频谱斜率: {features['spectral_slope_mean']:.2f}\n"
    result += f"   - 频谱通量: {features['spectral_flux_mean']:.2f}\n"
    
    return result

def _get_valence_description(value):
    if value > 0.5: return "积极/愉悦"
    elif value > 0: return "轻度积极"
    elif value > -0.5: return "轻度消极"
    else: return "消极/不悦"

def _get_arousal_description(value):
    if value > 0.5: return "高度唤醒"
    elif value > 0: return "轻度唤醒"
    elif value > -0.5: return "轻度平静"
    else: return "低度唤醒"

def _get_dominance_description(value):
    if value > 0.5: return "强势/主导"
    elif value > 0: return "轻度主导"
    elif value > -0.5: return "轻度顺从"
    else: return "顺从/被动"


def launch():
	with gr.Blocks(theme=gr.themes.Soft()) as demo:
		with gr.Tab("声学特征分析"):
			with gr.Row():
				with gr.Column():
					audio_input = gr.Audio(type="filepath", label="输入")
					analyze_btn = gr.Button("分析特征")
				with gr.Column():
					output_text = gr.Textbox(label="分析结果", lines=20)
			
			analyze_btn.click(
				fn=extract_features_from_audio,
				inputs=[audio_input],
				outputs=[output_text]
			)
			
			audio_input.change(
				fn=extract_features_from_audio,
				inputs=[audio_input],
				outputs=[output_text]
			)
	demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
	launch()
