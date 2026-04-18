"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

import gc
import threading
import numpy as np

from runpod.serverless.utils import rp_cuda

from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp

# Define available models (for validation)
AVAILABLE_MODELS = {
    "tiny.en",
}


class Predictor:
    """A Predictor class for the Whisper model with lazy loading"""

    def __init__(self):
        """Initializes the predictor with no models loaded."""
        self.models = {}
        self.model_lock = (
            threading.Lock()
        )  # Lock for thread-safe model loading/unloading

    def setup(self):
        """Pre-warm tiny.en so first request has no cold-load delay."""
        with self.model_lock:
            self._load_model_locked("tiny.en")

    def _load_model_locked(self, model_name):
        """Load model into self.models, evicting existing one. Must hold model_lock."""
        if model_name in self.models:
            return self.models[model_name]
        if self.models:
            existing = list(self.models.keys())[0]
            print(f"Unloading model: {existing}...")
            del self.models[existing]
            self.models.clear()
            gc.collect()
            print(f"Model {existing} unloaded.")
        print(f"Loading model: {model_name}...")
        try:
            loaded_model = WhisperModel(
                model_name,
                device="cuda" if rp_cuda.is_available() else "cpu",
                compute_type="float16" if rp_cuda.is_available() else "int8",
            )
            self.models[model_name] = loaded_model
            print(f"Model {model_name} loaded successfully.")
            return loaded_model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise ValueError(f"Failed to load model {model_name}: {e}") from e

    def predict(
        self,
        audio,
        model_name="tiny.en",
        transcription="plain_text",
        translate=False,
        translation="plain_text",  # Added in a previous PR
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=False,
        word_timestamps=False,
    ):
        """
        Run a single prediction on the model, loading/unloading models as needed.
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models are: {AVAILABLE_MODELS}"
            )

        with self.model_lock:
            model = self._load_model_locked(model_name)

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        segments, info = list(
            model.transcribe(
                str(audio),
                language=language,
                task="transcribe",
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                prefix=None,
                suppress_blank=True,
                suppress_tokens=[-1],  # Might need conversion from string
                without_timestamps=False,
                max_initial_timestamp=1.0,
                word_timestamps=word_timestamps,
                vad_filter=enable_vad,
            )
        )

        segments = list(segments)

        # Format transcription
        transcription_output = format_segments(transcription, segments)

        # Handle translation if requested
        translation_output = None
        if translate:
            translation_segments, _ = model.transcribe(
                str(audio),
                task="translate",
                temperature=temperature,  # Reuse temperature settings for translation
            )
            translation_output = format_segments(
                translation, list(translation_segments)
            )

        results = {
            "transcription": transcription_output,
            "detected_language": info.language,
        }

        if translation_output is not None:
            results["translation"] = translation_output

        if word_timestamps:
            word_timestamps_list = []
            for segment in segments:
                for word in segment.words:
                    word_timestamps_list.append(
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                        }
                    )
            results["word_timestamps"] = word_timestamps_list

        return results


def format_segments(format_type, segments):
    """
    Format the segments to the desired format
    """

    if format_type == "plain_text":
        return " ".join([segment.text.lstrip() for segment in segments])
    elif format_type == "formatted_text":
        return "\n".join([segment.text.lstrip() for segment in segments])
    elif format_type == "srt":
        return write_srt(segments)
    elif format_type == "vtt":  # Added VTT case
        return write_vtt(segments)
    else:  # Default or unknown format
        print(f"Warning: Unknown format '{format_type}', defaulting to plain text.")
        return " ".join([segment.text.lstrip() for segment in segments])


def write_vtt(transcript):
    """
    Write the transcript in VTT format.
    """
    result = ""

    for segment in transcript:
        # Using the consistent timestamp format from previous PR
        result += f"{format_timestamp(segment.start, always_include_hours=True)} --> {format_timestamp(segment.end, always_include_hours=True)}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result


def write_srt(transcript):
    """
    Write the transcript in SRT format.
    """
    result = ""

    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result
