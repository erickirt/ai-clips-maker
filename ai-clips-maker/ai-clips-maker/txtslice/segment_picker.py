"""
Segment extraction from audio using embedding-based TextTiling segmentation.
Ideal for generating meaningful portions from transcripts.
"""

import logging
import torch

from .matcher import MediaSegment
from .exceptions import ClipFinderError
from .embed_vectorizer import TextEmbedder
from .tiler_algorithm import TextTiler, TextTilerConfigManager

from ai_clips_maker.transcribe.transcription import Transcription
from ai_clips_maker.utils.pytorch import get_compute_device, assert_compute_device_available
from ai_clips_maker.utils.utils import find_missing_dict_keys

BOUNDARY = 1


class ClipFinder:
    """
    Finds meaningful audio segments within a transcript by applying the TextTiling
    algorithm over sentence embeddings.
    """

    def __init__(
        self,
        device: str = None,
        min_clip_duration: int = 15,
        max_clip_duration: int = 900,
        cutoff_policy: str = "high",
        embedding_aggregation_pool_method: str = "max",
        smoothing_width: int = 3,
        window_compare_pool_method: str = "mean",
    ) -> None:
        """
        Initializes the ClipFinder with segmentation strategy configuration.
        """
        config_manager = ClipFinderConfigManager()
        config_manager.assert_valid_config(
            {
                "cutoff_policy": cutoff_policy,
                "embedding_aggregation_pool_method": embedding_aggregation_pool_method,
                "max_clip_duration": max_clip_duration,
                "min_clip_duration": min_clip_duration,
                "smoothing_width": smoothing_width,
                "window_compare_pool_method": window_compare_pool_method,
            }
        )
        if device is None:
            device = get_compute_device()
        assert_compute_device_available(device)
        self._device = device
        self._cutoff_policy = cutoff_policy
        self._embedding_aggregation_pool_method = embedding_aggregation_pool_method
        self._min_clip_duration = min_clip_duration
        self._max_clip_duration = max_clip_duration
        self._smoothing_width = smoothing_width
        self._window_compare_pool_method = window_compare_pool_method

    def find_clips(self, transcription: Transcription) -> list[MediaSegment]:
        """
        Executes segmentation on the transcription and returns segments.
        """
        sentences_info = transcription.get_sentence_info()
        sentences = [info["sentence"] for info in sentences_info]

        embedder = TextEmbedder()
        sentence_embeddings = embedder.embed_sentences(sentences)

        clips = []
        if transcription.end_time <= self._max_clip_duration:
            clips.append({
                "start_char": 0,
                "end_char": len(transcription.get_char_info()),
                "start_time": 0,
                "end_time": transcription.end_time,
                "norm": 1.0
            })

        for k_vals, min_sec in [([5, 7], self._min_clip_duration), ([11, 17], 180), ([37, 53, 73, 97], 600)]:
            for k in k_vals:
                clips = self._text_tile_multiple_rounds(
                    sentences_info,
                    sentence_embeddings,
                    k,
                    min_sec,
                    self._max_clip_duration,
                    clips
                )

        return [
            MediaSegment(
                clip["start_time"],
                clip["end_time"],
                clip["start_char"],
                clip["end_char"]
            ) for clip in clips
        ]

    def _text_tile_multiple_rounds(
        self,
        clips: list[dict],
        clip_embeddings: torch.tensor,
        k: int,
        min_clip_duration: int,
        max_clip_duration: int,
        final_clips: list[dict] = [],
    ) -> list[dict]:
        """
        Applies multi-round segmentation using TextTiling and filters results.
        """
        self._text_tile_round = 0
        while len(clip_embeddings) > 8:
            self._text_tile_round += 1
            super_clips, super_embeddings = self._text_tile(
                clips, clip_embeddings, k
            )
            new_clips = self._remove_duplicates(
                super_clips,
                final_clips,
                min_clip_duration,
                max_clip_duration,
            )
            final_clips += new_clips
            clips = super_clips
            clip_embeddings = super_embeddings

        return final_clips

    def _text_tile(
        self,
        clips: list[dict],
        clip_embeddings: torch.tensor,
        k: int,
    ) -> tuple[list, torch.Tensor]:
        """
        Runs the TextTiling algorithm and constructs new combined segments.
        """
        if len(clip_embeddings) != len(clips):
            msg = f"Embedding length ({len(clip_embeddings)}) and clips ({len(clips)}) mismatch."
            logging.error(msg)
            raise ClipFinderError(msg)

        tiler = TextTiler(self._device)
        k = min(k, max(3, len(clip_embeddings)))

        boundaries, new_embeddings = tiler.text_tile(
            clip_embeddings,
            k,
            self._window_compare_pool_method,
            self._embedding_aggregation_pool_method,
            self._smoothing_width,
            self._cutoff_policy,
        )

        super_clips = []
        start_idx = 0
        super_idx = 0

        for i in range(len(clips)):
            if boundaries[i] == BOUNDARY:
                end_idx = i
                combined = {
                    "start_char": clips[start_idx]["start_char"],
                    "end_char": clips[end_idx]["end_char"],
                    "start_time": clips[start_idx]["start_time"],
                    "end_time": clips[end_idx]["end_time"],
                    "norm": torch.linalg.norm(new_embeddings[super_idx], dim=0, ord=2).item()
                }
                super_clips.append(combined)
                start_idx = end_idx
                super_idx += 1

        return super_clips, new_embeddings

    def _remove_duplicates(
        self,
        potential: list[dict],
        existing: list[dict],
        min_dur: int,
        max_dur: int,
    ) -> list[dict]:
        """
        Filters out too short, too long, or overlapping (duplicate) segments.
        """
        results = []
        for clip in potential:
            duration = clip["end_time"] - clip["start_time"]
            if not (min_dur <= duration <= max_dur):
                continue
            if self._is_duplicate(clip, existing):
                continue
            results.append(clip)
        return results

    def _is_duplicate(self, seg: dict, existing: list[dict]) -> bool:
        """
        Determines if segment already exists (based on small time delta).
        """
        for ref in existing:
            if abs(seg["start_time"] - ref["start_time"]) + abs(seg["end_time"] - ref["end_time"]) < 15:
                return True
        return False


class ClipFinderConfigManager(TextTilerConfigManager):
    """
    Validates configuration and imputes default values for the segmentation process.
    """

    def __init__(self) -> None:
        super().__init__()

    def impute_default_config(self, config: dict) -> dict:
        defaults = {
            "compute_device": "cpu",
            "cutoff_policy": "high",
            "embedding_aggregation_pool_method": "max",
            "min_clip_time": 15,
            "max_clip_time": 900,
            "smoothing_width": 3,
            "window_compare_pool_method": "mean",
        }
        for k, v in defaults.items():
            config.setdefault(k, v)
        return config

    def check_valid_config(self, cfg: dict) -> str | None:
        required = [
            "cutoff_policy",
            "embedding_aggregation_pool_method",
            "max_clip_duration",
            "min_clip_duration",
            "smoothing_width",
            "window_compare_pool_method",
        ]
        missing = find_missing_dict_keys(cfg, required)
        if missing:
            return f"Missing config values: {missing}"

        err = self.check_valid_clip_times(
            cfg["min_clip_duration"], cfg["max_clip_duration"]
        )
        if err:
            return err

        validators = {
            "cutoff_policy": self.check_valid_cutoff_policy,
            "embedding_aggregation_pool_method": self.check_valid_embedding_aggregation_pool_method,
            "smoothing_width": self.check_valid_smoothing_width,
            "window_compare_pool_method": self.check_valid_window_compare_pool_method,
        }
        for key, func in validators.items():
            result = func(cfg[key])
            if result:
                return result

        return None

    def check_valid_clip_times(self, min_dur: float, max_dur: float) -> str | None:
        self._type_checker.check_type(min_dur, "min_clip_duration", (float, int))
        self._type_checker.check_type(max_dur, "max_clip_duration", (float, int))

        if min_dur < 0:
            return f"min_clip_duration must be >= 0, got {min_dur}"
        if max_dur <= min_dur:
            return f"max_clip_duration ({max_dur}) must be greater than min_clip_duration ({min_dur})"

        return None