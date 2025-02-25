import opensmile
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AcousticVADAnalyzer:
    def __init__(self):
        # 初始化特征提取器
        # 使用 ComParE_2016 特征集
        self.smile_compare = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        # 使用 eGeMAPSv02 特征集
        self.smile_gemaps = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract_features(self, audio_file):
        """提取音频特征"""
        try:
            # 提取两种特征集
            compare_features = self.smile_compare.process_file(audio_file)
            gemaps_features = self.smile_gemaps.process_file(audio_file)
            

            # print("\nComParE特征集可用特征:")
            # print(pd.DataFrame(compare_features).columns.tolist())
            
            # print("\neGeMaPS特征集可用特征:")
            # print(pd.DataFrame(gemaps_features).columns.tolist())

            # 整合特征
            features = self._process_features(compare_features, gemaps_features)
            return features
        except Exception as e:
            print(f"特征提取出错: {(e)}")
            return None

    def _process_features(self, compare_features, gemaps_features):
        """处理和整合特征"""
        features = {}
        
        # 基础声学特征
        features.update({
            # RMS/响度特征
            'rms_mean': gemaps_features['loudness_sma3_amean'].iloc[0],
            'rms_std': gemaps_features['loudness_sma3_stddevNorm'].iloc[0],  # 修改为正确的特征名
            
            # 音高特征
            'f0_mean': gemaps_features['F0semitoneFrom27.5Hz_sma3nz_amean'].iloc[0],
            'f0_std': gemaps_features['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'].iloc[0],  # 修改为正确的特征名
            'f0_range': gemaps_features['F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2'].iloc[0],  # 修改为正确的特征名
            
            # 频谱特征
            'spectral_flux_mean': gemaps_features['spectralFlux_sma3_amean'].iloc[0],
            'spectral_slope_mean': gemaps_features['slopeV0-500_sma3nz_amean'].iloc[0],  # 使用这个作为谱斜率的替代
        })
        
        # 计算VAD情感维度
        features.update(self._calculate_vad(compare_features, gemaps_features))
        
        return features

    def _calculate_vad(self, compare_features, gemaps_features):
        """计算VAD情感维度"""
        # Valence (情感效价)
        valence_features = [
            gemaps_features['F0semitoneFrom27.5Hz_sma3nz_amean'].iloc[0],
            gemaps_features['jitterLocal_sma3nz_amean'].iloc[0],
            gemaps_features['spectralFlux_sma3_amean'].iloc[0]
        ]
        
        # Arousal (唤醒度)
        arousal_features = [
            gemaps_features['loudness_sma3_amean'].iloc[0],
            gemaps_features['shimmerLocaldB_sma3nz_amean'].iloc[0],  # 修改为正确的特征名
            gemaps_features['spectralFlux_sma3_amean'].iloc[0]
        ]
        
        # Dominance (支配度)
        dominance_features = [
            gemaps_features['F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2'].iloc[0],
            gemaps_features['loudness_sma3_pctlrange0-2'].iloc[0],
            gemaps_features['slopeV0-500_sma3nz_amean'].iloc[0]
        ]
        
        return {
            'valence': float(np.mean(self._normalize_array(valence_features))),
            'arousal': float(np.mean(self._normalize_array(arousal_features))),
            'dominance': float(np.mean(self._normalize_array(dominance_features)))
        }


    def _normalize_array(self, array):
        """特征归一化到[-1, 1]范围"""
        array = np.array(array)
        if np.max(array) == np.min(array):
            return np.zeros_like(array)
        return (array - np.min(array)) / (np.max(array) - np.min(array)) * 2 - 1

    def print_analysis(self, features):
        """打印分析结果"""
        print("\n=== 音频分析报告 ===")
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n1. VAD情感维度:")
        print(f"   情感效价 (Valence): {(features['valence']*100+50)}  ", 
              self._get_valence_description(features['valence']))
        print(f"   唤醒度 (Arousal): {(features['arousal']*100+50)}  ",
              self._get_arousal_description(features['arousal']))
        print(f"   支配度 (Dominance): {(features['dominance']*100+50)}  ",
              self._get_dominance_description(features['dominance']))
        
        print("\n2. 声学特征:")
        print(f"   RMS能量:")
        print(f"   - 均值: {features['rms_mean']:.2f}")
        print(f"   - 标准差: {features['rms_std']:.2f}")
        
        print(f"\n   音高特征:")
        print(f"   - 均值: {features['f0_mean']:.2f} Hz")
        print(f"   - 标准差: {features['f0_std']:.2f}")
        print(f"   - 范围: {features['f0_range']:.2f}")
        
        print(f"\n   频谱特征:")
        print(f"   - 频谱斜率: {features['spectral_slope_mean']:.2f}")
        print(f"   - 频谱通量: {features['spectral_flux_mean']:.2f}")

    def _get_valence_description(self, value):
        if value > 0.5: return "积极/愉悦"
        elif value > 0: return "轻度积极"
        elif value > -0.5: return "轻度消极"
        else: return "消极/不愉悦"
    
    def _get_arousal_description(self, value):
        if value > 0.5: return "高度唤醒/激动"
        elif value > 0: return "轻度唤醒"
        elif value > -0.5: return "平和"
        else: return "平静/冷淡"
    
    def _get_dominance_description(self, value):
        if value > 0.5: return "强势/支配"
        elif value > 0: return "轻度支配"
        elif value > -0.5: return "轻度顺从"
        else: return "顺从/被动"

def main():
    print("音频VAD情感维度与声学特征分析工具")
    print("支持格式: WAV, MP3等常见音频格式")
    
    analyzer = AcousticVADAnalyzer()
    
    while True:
        audio_file = input("\n请输入音频文件路径 (输入 'q' 退出): ")
        if audio_file.lower() == 'q':
            break
            
        print("\n正在分析音频...")
        features = analyzer.extract_features(audio_file)
        
        if features:
            analyzer.print_analysis(features)
        else:
            print("分析失败，请检查文件是否存在且格式正确")

if __name__ == "__main__":
    main()