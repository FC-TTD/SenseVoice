import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
        self,
        input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


class Wav2Vec2VAD:
    def __init__(self):
        self.device = device
        self.model = model
        self.processor = processor

    def process(
        self,
        input: np.ndarray,
        embeddings: bool = False,
        raw: bool = False,
    ) -> str | dict[str, int]:
        r"""Predict emotions or extract embeddings from raw audio signal."""

        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        y = self.processor(input, sampling_rate=16000)
        y = y["input_values"][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(self.device)

        # run through model
        with torch.no_grad():
            y = self.model(y)[0 if embeddings else 1]

        # convert to numpy
        y = y.detach().cpu().numpy().flatten().tolist()

        result = self.format_vad(y)

        # pretty print
        result_text = f"Valence: {result['Valence']} Arousal: {result['Arousal']} Dominance: {result['Dominance']} \n ORI(A,D,V): {y}"

        return raw and result or result_text
    
    def format_vad(self, vad: np.array) -> dict[str, int]:
        r"""曲线特征：
        x ≤ 60	        y = x 严格线性	        (30,30) (60,60)
        60 < x ≤ 90	    平缓衰减，差距增长较慢	    (75,73.79)
        90 < x <110	    加速衰减，差距快速增大	    (100,93.5)
        x ≥ 110	        恒定值 99	            (110,99)"""
        
        def fixed_custom_curve(x):
            if x <= 60:
                return x
            elif 60 < x <= 90:
                a = 6 / (30**1.5)  # 保证x=90时差距为6
                return x - a * (x - 60)**1.5
            elif 90 < x < 110:
                return x - (6 + 0.0125 * (x - 90)**2)
            else:
                return 99

        return {
            "Valence": int(fixed_custom_curve(vad[2]*100)),
            "Arousal": int(fixed_custom_curve(vad[0]*100)),
            "Dominance": int(fixed_custom_curve(vad[1]*100)),
            "raw": vad,
        }


device = "cpu"
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)
# print(process_func(signal, sampling_rate))
#  Arousal    dominance valence
# [[0.5460754  0.6062266  0.40431657]]

# print(process_func(signal, sampling_rate, embeddings=True))
# Pooled hidden states of last transformer layer
# [[-0.00752167  0.0065819  -0.00746342 ...  0.00663632  0.00848748
#    0.00599211]]
