import re

import librosa
import numpy as np
import pyloudnorm as pyln

# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from funasr import AutoModel


class PriFile:
    def __init__(self, args, model=None):
        audio, sr = args
        # self.file = file_path
        self.audio = audio
        self.sr = sr
        self.model = model  
        self.loundness = self.calculate_loudness()
        self.rate = self.funasr_speechspeed_measure()
        self.pitches = self.pitch_measure()

    def calculate_loudness(self) -> dict({"mean": int, "max": int}):
        """
        计算音频文件的综合响度（LUFS）和最大响度（短期或瞬态响度）
        :param data: 音频数据
        :param rate: 采样率
        :return: 综合响度值（LUFS），最大响度（LUFS）
        """
        data = self.audio
        rate = self.sr
        # 如果音频是多通道，转为单通道（取平均值）
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        # 创建响度测量器
        meter = pyln.Meter(rate)
        # 计算综合响度
        itgr_loudness = meter.integrated_loudness(data)
        # 分帧计算瞬态响度（400ms窗口）
        max_loudness = float("-inf")
        frame_size = int(rate * 0.4)  # 400ms 的帧大小
        for i in range(0, len(data), frame_size):
            frame = data[i : i + frame_size]
            if len(frame) < frame_size:  # 跳过不足一个窗口的片段
                continue
            loudness = meter.integrated_loudness(frame)
            max_loudness = max(max_loudness, loudness)
        return {"itgr": itgr_loudness, "max": max_loudness}

    def funasr_speechspeed_measure(self):
        """
        使用FunASR模型计算音频文件的语速（WPM）
        :param audio_path: 音频文件路径
        :return: 语速（WPM）
        """
        # inference_pipeline = pipeline(
        #     task=Tasks.auto_speech_recognition,
        #     model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        #     model_revision="v2.0.4",
        # )

        # rec_result = inference_pipeline(audio_path)
        if self.model is None:
            # 如果没有传入模型，则创建一个临时模型
            temp_model = AutoModel(model="paraformer-zh")
            rec_result = temp_model.generate(self.audio)
        else:
            # 使用传入的模型
            rec_result = self.model.generate(self.audio)
        # 提取时间戳
        timestamps = rec_result[0]["timestamp"]
        # print(timestamps)
        # 计算语音活动总时间（秒）
        total_duration_ms = sum(end - start for start, end in timestamps)
        total_duration_seconds = total_duration_ms / 1000.0
        # 计算总字数
        content = re.findall(r"[\u4e00-\u9fff]", rec_result[0]["text"])
        total_words = len(content)
        # print("total_words:", total_words)
        # print("total_duration_seconds", total_duration_seconds)
        # 计算语速 (WPM)
        wpm = (total_words / total_duration_seconds) * 60
        return wpm

    def pitch_measure(self) -> dict({"mean": int, "max": int}):
        """
        提取音频文件的基频
        :param data: 音频数据
        :param rate: 采样率
        :return: 基频均值（Hz），基频最大值（Hz）
        """
        data = self.audio
        rate = self.sr
        # 提取基频
        f0, voiced_flag, voiced_probs = librosa.pyin(data, fmin=50, fmax=500, sr=rate)
        mean_pitch = np.mean(f0[~np.isnan(f0)])
        max_pitch = np.nanmax(f0)
        return {"mean": mean_pitch, "max": max_pitch}

    def mean_measure(self):  # 方案一
        mean_pitch = "{:03d}".format(round(self.pitches["mean"]))
        wpm = "{:03d}".format(round(self.rate))
        average_loudness = "{:03d}".format(round(self.loundness["itgr"]))
        return str(mean_pitch) + str(wpm) + str(average_loudness)

    def max_measure(self):  # 方案一
        max_pitch = "{:03d}".format(round(self.pitches["max"]))
        wpm = "{:03d}".format(round(self.rate))
        max_loudness = "{:03d}".format(round(self.loundness["max"]))
        return str(max_pitch) + str(wpm) + str(max_loudness)
