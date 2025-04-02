"""
WhisperX-powered audio transcription engine.

Key Features:
-------------
- Language auto-detection or manual selection (ISO 639-1)
- Word & character-level timestamps with alignment
- WhisperX: https://github.com/m-bain/whisperX
"""

import logging
from datetime import datetime

# Internal imports
from .exceptions import NoSpeechError, TranscriberConfigError
from .transcription import Transcription
from ai_clips_maker.media.audio_file import AudioFile
from ai_clips_maker.media.editor import MediaEditor
from ai_clips_maker.utils.config_manager import ConfigManager
from ai_clips_maker.utils.pytorch import assert_valid_torch_device, get_compute_device
from ai_clips_maker.utils.type_checker import TypeChecker
from ai_clips_maker.utils.utils import find_missing_dict_keys

# External libraries
import torch
import whisperx


class WhisperTranscriber:
    """
    Audio transcriber using WhisperX models.
    """

    def __init__(self, model_size=None, device=None, precision=None) -> None:
        self._config = WhisperTranscriberConfig()
        self._type_checker = TypeChecker()

        self._device = device or get_compute_device()
        self._precision = precision or ("float16" if torch.cuda.is_available() else "int8")
        self._model_size = model_size or ("large-v2" if torch.cuda.is_available() else "tiny")

        assert_valid_torch_device(self._device)
        self._config.assert_valid_model_size(self._model_size)
        self._config.assert_valid_precision(self._precision)

        self._model = whisperx.load_model(
            whisper_arch=self._model_size,
            device=self._device,
            compute_type=self._precision,
        )

    def transcribe(self, audio_path: str, lang: str = None, batch_size: int = 16) -> Transcription:
        """
        Transcribes an audio or video file to text with aligned timestamps.
        """
        editor = MediaEditor()
        media = editor.instantiate_as_temporal_media_file(audio_path)
        media.assert_exists()
        media.assert_has_audio_stream()

        if lang:
            self._config.assert_valid_language(lang)

        # Step 1: WhisperX transcription
        raw_transcription = self._model.transcribe(
            media.path, language=lang, batch_size=batch_size
        )

        # Step 2: Align characters/words
        align_model, meta = whisperx.load_align_model(
            language_code=raw_transcription["language"],
            device=self._device
        )
        aligned = whisperx.align(
            raw_transcription["segments"],
            align_model,
            meta,
            media.path,
            self._device,
            return_char_alignments=True,
        )

        if not aligned["segments"]:
            raise NoSpeechError(f"No speech detected in: {media.path}")

        # Step 3: Parse character-level data
        char_info = []

        try:
            del aligned["segments"][0]["chars"][0]  # Remove leading space char
        except Exception as e:
            logging.error(f"Failed to clean first char: {str(e)}")
            raise

        for segment in aligned["segments"]:
            for char in segment["chars"]:
                char_info.append({
                    "char": char["char"],
                    "start_time": float(char["start"]) if "start" in char else None,
                    "end_time": float(char["end"]) if "end" in char else None,
                    "speaker": None
                })

        return Transcription({
            "source_software": "whisperx-v3",
            "time_created": datetime.now(),
            "language": raw_transcription["language"],
            "num_speakers": None,
            "char_info": char_info,
        })

    def detect_language(self, media_file: AudioFile) -> str:
        """
        Detects the spoken language in the media file.
        """
        self._type_checker.assert_type(media_file, "media_file", AudioFile)
        media_file.assert_exists()
        media_file.assert_has_audio_stream()
        audio = whisperx.load_audio(media_file.path)
        return self._model.detect_language(audio)


class WhisperTranscriberConfig(ConfigManager):
    """
    Configuration validator for WhisperTranscriber.
    """

    def check_valid_config(self, config: dict) -> str | None:
        required = {
            "language": self.check_valid_language,
            "model_size": self.check_valid_model_size,
            "precision": self.check_valid_precision,
        }

        missing = find_missing_dict_keys(config, required.keys())
        if missing:
            return f"Missing config keys: {missing}"

        for key, checker in required.items():
            if config[key] is None:
                continue
            msg = checker(config[key])
            if msg:
                return msg
        return None

    def get_valid_model_sizes(self) -> list[str]:
        return ["tiny", "base", "small", "medium", "large-v1", "large-v2"]

    def get_valid_languages(self) -> list[str]:
        return ["en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"]

    def get_valid_precisions(self) -> list[str]:
        return ["float32", "float16", "int8"]

    def check_valid_model_size(self, size: str) -> str | None:
        if size not in self.get_valid_model_sizes():
            return f"Invalid model size '{size}'. Valid options: {self.get_valid_model_sizes()}"
        return None

    def check_valid_language(self, code: str) -> str | None:
        if code not in self.get_valid_languages():
            return f"Invalid language code '{code}'. Valid options: {self.get_valid_languages()}"
        return None

    def check_valid_precision(self, precision: str) -> str | None:
        if precision not in self.get_valid_precisions():
            return f"Invalid precision '{precision}'. Valid: {self.get_valid_precisions()}"
        return None

    def assert_valid_model_size(self, size: str) -> None:
        msg = self.check_valid_model_size(size)
        if msg:
            raise TranscriberConfigError(msg)

    def assert_valid_language(self, code: str) -> None:
        msg = self.check_valid_language(code)
        if msg:
            raise TranscriberConfigError(msg)

    def assert_valid_precision(self, precision: str) -> None:
        msg = self.check_valid_precision(precision)
        if msg:
            raise TranscriberConfigError(msg)
