import opensmile
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class AcousticVADAnalyzer:
    def __init__(self):

        # 使用 eGeMAPSv02 特征集
        self.smile_gemaps = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract_features(self, audio_file):
        """提取音频特征"""
        try:
            # 提取两种特征集
            gemaps_features = self.smile_gemaps.process_file(audio_file)
            # print("\neGeMaPS特征集可用特征:")
            # print(pd.DataFrame(gemaps_features).columns.tolist())

            # 整合特征
            features = self._process_features(gemaps_features)
            return features
        except Exception as e:
            print(f"特征提取出错: {(e)}")
            return None

    def _process_features(self, gemaps_features):
        """处理和整合特征"""
        features = {}

        # 基础声学特征
        features.update(
            {
                # RMS/响度特征
                "rms_mean": gemaps_features["loudness_sma3_amean"].iloc[0],
                "rms_std": gemaps_features["loudness_sma3_stddevNorm"].iloc[0],  # 修改为正确的特征名
                # 音高特征
                "f0_mean": gemaps_features["F0semitoneFrom27.5Hz_sma3nz_amean"].iloc[0],
                "f0_std": gemaps_features["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"].iloc[0],  # 修改为正确的特征名
                "f0_range": gemaps_features["F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"].iloc[0],  # 修改为正确的特征名
                # 频谱特征
                "spectral_flux_mean": gemaps_features["spectralFlux_sma3_amean"].iloc[0],
                "spectral_slope_mean": gemaps_features["slopeV0-500_sma3nz_amean"].iloc[0],  # 使用这个作为谱斜率的替代
            }
        )

        return features

    def print_analysis(self, features):
        """打印分析结果"""
        result = "\n2. 声学特征:"
        result += "\n   RMS能量:"
        result += f"\n   - 均值: {features['rms_mean']:.2f}"
        result += f"\n   - 标准差: {features['rms_std']:.2f}"

        result += "\n   音高特征:"
        result += f"\n   - 均值: {features['f0_mean']:.2f} Hz"
        result += f"\n   - 标准差: {features['f0_std']:.2f}"
        result += f"\n   - 范围: {features['f0_range']:.2f}"

        result += "\n   频谱特征:"
        result += f"\n   - 频谱斜率: {features['spectral_slope_mean']:.2f}"
        result += f"\n   - 频谱通量: {features['spectral_flux_mean']:.2f}"
        return result


def main():
    print("音频VAD情感维度与声学特征分析工具")
    print("支持格式: WAV, MP3等常见音频格式")

    analyzer = AcousticVADAnalyzer()

    while True:
        audio_file = input("\n请输入音频文件路径 (输入 'q' 退出): ")
        if audio_file.lower() == "q":
            break

        print("\n正在分析音频...")
        features = analyzer.extract_features(audio_file)

        if features:
            analyzer.print_analysis(features)
        else:
            print("分析失败，请检查文件是否存在且格式正确")


if __name__ == "__main__":
    main()
