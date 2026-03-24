import gc
import logging
import os
import re
from typing import Any, Dict, List

import librosa
import numpy as np
from funasr import AutoModel

from utils.subtitle_utils import generate_srt

_funasr_models = {}
_whisperx_models = {}
_whisperx_align_models = {}
_whisperx_diarize_models = {}


def _parse_timestamp_number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _should_scale_timestamp_values(values: List[float]) -> bool:
    numeric_values = [abs(v) for v in values if v is not None]
    if not numeric_values:
        return False
    if any(not float(v).is_integer() for v in numeric_values):
        return True
    return max(numeric_values) <= 30


def _coerce_ms(value, scale_to_ms: bool = False):
    numeric = _parse_timestamp_number(value)
    if numeric is None:
        return None
    if scale_to_ms:
        numeric *= 1000
    return int(round(numeric))


def _segments_to_sentence_info(segments):
    sentence_info = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        scale_to_ms = _should_scale_timestamp_values(
            [
                _parse_timestamp_number(seg.get("start")),
                _parse_timestamp_number(seg.get("end")),
            ]
        )
        start = _coerce_ms(seg.get("start"), scale_to_ms=scale_to_ms)
        end = _coerce_ms(seg.get("end"), scale_to_ms=scale_to_ms)
        if start is None or end is None or end <= start:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        item = {
            "text": text,
            "timestamp": [[start, end]],
        }
        if seg.get("speaker"):
            item["speaker"] = _normalize_speaker_label(seg.get("speaker"))
        sentence_info.append(item)
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


def _normalize_speaker_label(value):
    if value is None:
        return "SPEAKER_UNKNOWN"
    speaker = str(value).strip()
    if not speaker:
        return "SPEAKER_UNKNOWN"

    normalized = speaker.upper()
    if normalized.startswith("SPEAKER_"):
        suffix = normalized.split("_", 1)[1]
        if suffix.isdigit():
            return f"SPEAKER_{int(suffix):02d}"
        return normalized

    match = re.search(r"(\d+)", speaker)
    if match:
        return f"SPEAKER_{int(match.group(1)):02d}"

    return f"SPEAKER_{normalized}"


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
                speaker = _normalize_speaker_label(sent.get("speaker")) if sent.get("speaker") else None
                if sent_text and timestamp:
                    item = {"text": sent_text, "timestamp": timestamp}
                    if speaker:
                        item["speaker"] = speaker
                    sentence_info.append(item)
            continue

        segments_sentence_info = _segments_to_sentence_info(res.get("segments"))
        if segments_sentence_info:
            sentence_info.extend(segments_sentence_info)
            continue

        timestamp = _normalize_timestamp_list(res.get("timestamp"))
        if text and timestamp:
            sentence_info.append({"text": text, "timestamp": timestamp})

    full_text = "\n".join(texts).strip()
    if not full_text and sentence_info:
        full_text = " ".join(
            sent.get("text", "").strip()
            for sent in sentence_info
            if sent.get("text")
        ).strip()
    if not sentence_info and full_text:
        sentence_info.append(
            {
                "text": full_text,
                "timestamp": [[0, int(round(audio_duration_ms))]],
            }
        )
    return sentence_info, full_text


def _normalize_device(device: str):
    if not device:
        return "cpu"
    device = device.lower()
    if device.startswith("cuda"):
        return "cuda"
    return device


def _resolve_whisperx_language(language: str):
    if language in {"auto", "nospeech", None}:
        return None
    if language == "yue":
        return "zh"
    return language


def _resolve_whisperx_threads():
    raw_value = os.getenv("SENSEVOICE_WHISPERX_THREADS")
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            logging.warning("Invalid SENSEVOICE_WHISPERX_THREADS=%s, fallback to cpu count", raw_value)
    return max(1, os.cpu_count() or 1)


def _resolve_whisperx_compute_type():
    device = _normalize_device(os.getenv("SENSEVOICE_DEVICE", "cpu"))
    if device == "cuda":
        return os.getenv("SENSEVOICE_WHISPERX_COMPUTE_TYPE", "float16")
    return os.getenv("SENSEVOICE_WHISPERX_COMPUTE_TYPE", "float32")


def _resolve_whisperx_hf_token():
    token = os.getenv("SENSEVOICE_WHISPERX_HF_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        return token.strip()
    token_file = os.getenv("SENSEVOICE_WHISPERX_HF_TOKEN_FILE") or os.getenv("HF_TOKEN_FILE")
    if token_file:
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                token = f.read().strip()
        except OSError as e:
            logging.warning("Failed to read WhisperX HF token file %s: %s", token_file, e)
            return None
        if token:
            return token
    return None


def _resolve_whisperx_diarize():
    raw_value = os.getenv("SENSEVOICE_WHISPERX_DIARIZE", "true")
    return raw_value.lower() not in {"0", "false", "no"}


def _load_whisperx_model(language: str = "zh"):
    lang_key = "en" if language == "en" else "zh"
    device = _normalize_device(os.getenv("SENSEVOICE_DEVICE", "cpu"))
    model_name = os.getenv("SENSEVOICE_WHISPERX_MODEL", "small")
    compute_type = _resolve_whisperx_compute_type()
    vad_method = os.getenv("SENSEVOICE_WHISPERX_VAD_METHOD", "silero")
    hf_token = _resolve_whisperx_hf_token()
    cache_key = f"whisperx_{lang_key}_{device}_{model_name}_{compute_type}_{vad_method}"
    if cache_key in _whisperx_models:
        return _whisperx_models[cache_key]

    logging.info(
        "Loading WhisperX model=%s device=%s compute_type=%s vad_method=%s",
        model_name,
        device,
        compute_type,
        vad_method,
    )
    try:
        import whisperx
    except Exception as e:
        logging.error("Failed to import whisperx. Error: %s", e)
        raise

    load_kwargs = dict(
        whisper_arch=model_name,
        device=device,
        compute_type=compute_type,
        language=_resolve_whisperx_language(language),
        vad_method=vad_method,
        threads=_resolve_whisperx_threads(),
        use_auth_token=hf_token,
    )
    try:
        model = whisperx.load_model(**load_kwargs)
    except Exception as e:
        error_text = str(e).lower()
        if vad_method == "silero" and hf_token and "rate limit" in error_text:
            logging.warning("Silero VAD hit GitHub rate limit, fallback to pyannote VAD")
            load_kwargs["vad_method"] = "pyannote"
            model = whisperx.load_model(**load_kwargs)
        else:
            raise
    _whisperx_models[cache_key] = model
    return model


def _load_whisperx_diarize_model():
    device = _normalize_device(os.getenv("SENSEVOICE_DEVICE", "cpu"))
    hf_token = _resolve_whisperx_hf_token()
    cache_key = f"diarize_{device}"
    if cache_key in _whisperx_diarize_models:
        return _whisperx_diarize_models[cache_key]

    if not hf_token:
        raise RuntimeError("WhisperX diarization requires HF token")

    try:
        from whisperx.diarize import DiarizationPipeline
    except Exception as e:
        logging.error("Failed to import whisperx for diarization. Error: %s", e)
        raise

    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    _whisperx_diarize_models[cache_key] = diarize_model
    logging.info("WhisperX diarization model loaded. device=%s", device)
    return diarize_model


def _load_whisperx_align_model(language: str):
    resolved_language = _resolve_whisperx_language(language) or os.getenv(
        "SENSEVOICE_WHISPERX_ALIGN_FALLBACK_LANGUAGE", "en"
    )
    device = _normalize_device(os.getenv("SENSEVOICE_DEVICE", "cpu"))
    cache_key = f"align_{resolved_language}_{device}"
    if cache_key in _whisperx_align_models:
        return _whisperx_align_models[cache_key]

    try:
        import whisperx
    except Exception as e:
        logging.error("Failed to import whisperx for alignment. Error: %s", e)
        raise

    model, metadata = whisperx.load_align_model(language_code=resolved_language, device=device)
    _whisperx_align_models[cache_key] = (model, metadata)
    logging.info("WhisperX align model loaded. language=%s device=%s", resolved_language, device)
    return model, metadata


def _sentence_info_to_labeled_text(sentence_info):
    lines = []
    for sent in sentence_info:
        text = (sent.get("text") or "").strip()
        if not text:
            continue
        speaker = None
        if sent.get("speaker"):
            speaker = _normalize_speaker_label(sent.get("speaker"))
        elif sent.get("spk") is not None:
            speaker = _normalize_speaker_label(sent.get("spk"))
        if speaker:
            lines.append(f"[{speaker}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines).strip()


def _normalize_sentence_speakers(sentence_info):
    normalized = []
    for sent in sentence_info or []:
        item = dict(sent)
        if item.get("speaker"):
            item["speaker"] = _normalize_speaker_label(item.get("speaker"))
        elif item.get("spk") is not None:
            item["speaker"] = _normalize_speaker_label(item.get("spk"))
            item.pop("spk", None)
        normalized.append(item)
    return normalized


def _sentence_info_to_api_segments(sentence_info) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for sent in sentence_info or []:
        if not isinstance(sent, dict):
            continue
        text = (sent.get("text") or "").strip()
        if not text:
            continue

        timestamp = _normalize_timestamp_list(sent.get("timestamp"))
        if timestamp:
            start = timestamp[0][0]
            end = timestamp[-1][1]
        else:
            start = _coerce_ms(sent.get("start"))
            end = _coerce_ms(sent.get("end"))
        if start is None or end is None or end <= start:
            continue

        raw_text = str(sent.get("raw_text") or text).strip()
        clean_text = re.sub(r"<\|.*?\|>", "", raw_text, 0, re.MULTILINE).strip() or text

        item: Dict[str, Any] = {
            "start": int(start),
            "end": int(end),
            "text": text,
            "raw_text": raw_text,
            "clean_text": clean_text,
        }

        speaker = None
        if sent.get("speaker"):
            speaker = _normalize_speaker_label(sent.get("speaker"))
        elif sent.get("spk") is not None:
            speaker = _normalize_speaker_label(sent.get("spk"))
        if speaker:
            item["speaker"] = speaker

        segments.append(item)

    return segments


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
    if model_type == "whisperx":
        return _load_whisperx_model(language)
    return _load_paraformer_model(language)


def convert_pcm_to_float(data: np.ndarray):
    if data.dtype == np.float64:
        return data
    if data.dtype == np.float32:
        return data.astype(np.float64)
    if data.dtype == np.int16:
        return data.astype(np.float64) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float64) / 2147483648.0
    if data.dtype == np.int8:
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


def _infer_whisperx(model, data: np.ndarray, sr: int, language: str, diarize: bool):
    try:
        import whisperx
        from whisperx.diarize import assign_word_speakers
    except Exception as e:
        logging.error("Failed to import whisperx during inference. Error: %s", e)
        raise

    device = _normalize_device(os.getenv("SENSEVOICE_DEVICE", "cpu"))
    batch_size = int(os.getenv("SENSEVOICE_WHISPERX_BATCH_SIZE", "8"))
    resolved_language = _resolve_whisperx_language(language)
    align_enabled = os.getenv("SENSEVOICE_WHISPERX_ALIGN", "true").lower() not in {"0", "false", "no"}

    rec_result = model.transcribe(
        data.astype(np.float32),
        batch_size=batch_size,
        language=resolved_language,
    )

    if align_enabled and rec_result.get("segments"):
        align_model, align_metadata = _load_whisperx_align_model(
            rec_result.get("language") or resolved_language or "en"
        )
        rec_result = whisperx.align(
            rec_result["segments"],
            align_model,
            align_metadata,
            data.astype(np.float32),
            device,
            return_char_alignments=False,
        )

    if diarize:
        diarize_model = _load_whisperx_diarize_model()
        diarize_segments = diarize_model(data.astype(np.float32))
        rec_result = assign_word_speakers(diarize_segments, rec_result)

    audio_duration_ms = len(data) / sr * 1000
    sentence_info, full_text = _collect_whisper_output(rec_result, audio_duration_ms)
    if diarize:
        full_text = _sentence_info_to_labeled_text(sentence_info) or full_text
    logging.info(
        "WhisperX parsed result: language=%s, segments=%d, sentence_info=%d, diarize=%s",
        rec_result.get("language"),
        len(rec_result.get("segments", [])) if isinstance(rec_result, dict) else 0,
        len(sentence_info),
        diarize,
    )
    return generate_srt(sentence_info), full_text, sentence_info


def _infer_paraformer(model, data: np.ndarray, language: str, speaker_enabled: bool):
    rec_result = model.generate(
        data,
        return_spk_res=speaker_enabled,
        return_raw_text=True,
        is_final=True,
        pred_timestamp=language == "en",
        en_post_proc=language == "en",
        cache={},
    )
    sentence_info = _normalize_sentence_speakers(rec_result[0]["sentence_info"])
    res_srt = generate_srt(sentence_info)
    asr_result = _sentence_info_to_labeled_text(sentence_info) if speaker_enabled else rec_result[0]["text"]
    return res_srt, asr_result, sentence_info


def infer(audio_data, language, model_type="paraformer", diarize=None, return_segments: bool = False):
    sr, data = _prepare_audio(audio_data)
    logging.info("Input audio ready. length: %.2f seconds. Model: %s", len(data) / sr, model_type)

    load_language = language if language not in {None, "", "auto"} else "zh"
    model = load_model(load_language, model_type=model_type)

    try:
        speaker_enabled = _resolve_whisperx_diarize() if diarize is None else bool(diarize)
        if model_type == "whisperx":
            srt_text, full_text, sentence_info = _infer_whisperx(model, data, sr, language, speaker_enabled)
        else:
            srt_text, full_text, sentence_info = _infer_paraformer(model, data, language, speaker_enabled)

        if return_segments:
            return srt_text, full_text, _sentence_info_to_api_segments(sentence_info)
        return srt_text, full_text
    except Exception as e:
        logging.exception("%s inference failed: %s", model_type, e)
        raise


def clear_model_cache():
    _funasr_models.clear()
    _whisperx_models.clear()
    _whisperx_align_models.clear()
    _whisperx_diarize_models.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_model_cache_stats():
    return {
        "paraformer": list(_funasr_models.keys()),
        "whisperx": list(_whisperx_models.keys()),
        "align": list(_whisperx_align_models.keys()),
        "diarize": list(_whisperx_diarize_models.keys()),
    }
