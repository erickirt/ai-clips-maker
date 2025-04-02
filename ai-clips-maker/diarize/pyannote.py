"""
Speaker diarization using pyannote.audio's 3.1 pipeline.

This module provides an interface for performing speaker diarization on audio
files using the pre-trained "pyannote/speaker-diarization-3.1" model hosted on HuggingFace.
"""

import logging
import os
import uuid

import torch
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation

from ai_clips_maker.media.audio_file import AudioFile
from ai_clips_maker.utils.pytorch import get_compute_device, assert_compute_device_available


class PyannoteDiarizer:
    """
    Wrapper for the pyannote speaker diarization pipeline.
    """

    def __init__(self, auth_token: str, device: str = None) -> None:
        """
        Initialize the diarization pipeline.

        Parameters
        ----------
        auth_token : str
            HuggingFace authentication token for accessing the model.
        device : str, optional
            Device to run inference on (e.g., 'cpu', 'cuda').
            Defaults to auto-detected device.
        """
        if device is None:
            device = get_compute_device()
        assert_compute_device_available(device)

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token,
        ).to(torch.device(device))

        logging.debug(f"Initialized pyannote pipeline on device: {device}")

    def diarize(
        self,
        audio_file: AudioFile,
        min_segment_duration: float = 1.5,
        time_precision: int = 6,
    ) -> list[dict]:
        """
        Perform speaker diarization on an audio file.

        Parameters
        ----------
        audio_file : AudioFile
            The input audio file.
        min_segment_duration : float
            Minimum duration (in seconds) for a valid segment.
        time_precision : int
            Decimal precision for timestamps.

        Returns
        -------
        list[dict]
            List of speaker segments, each with keys: 'speakers', 'start_time', 'end_time'.
        """
        if audio_file.has_file_extension("wav"):
            wav_file = audio_file
        else:
            wav_path = os.path.join(
                audio_file.get_parent_dir_path(),
                f"{audio_file.get_filename_without_extension()}_{uuid.uuid4().hex}.wav",
            )
            wav_file = audio_file.extract_audio(
                extracted_audio_file_path=wav_path,
                audio_codec="pcm_s16le",
                overwrite=False,
            )

        annotation: Annotation = self.pipeline({"audio": wav_file.path})
        duration = audio_file.get_duration()

        segments = self._adjust_segments(
            annotation, duration, min_segment_duration, time_precision
        )

        if wav_file is not audio_file:
            wav_file.delete()

        return segments

    def _adjust_segments(
        self,
        annotation: Annotation,
        duration: float,
        min_segment_duration: float,
        time_precision: int,
    ) -> list[dict]:
        """
        Adjust speaker segments to be non-overlapping and contiguous.

        Returns
        -------
        list[dict]
            Cleaned list of speaker segments.
        """
        segments = []
        unique_speakers = set()

        cur_start = 0.0
        cur_speaker = None
        cur_end = None

        for segment, _, label in annotation.itertracks(yield_label=True):
            start, end = segment.start, segment.end
            if end - start < min_segment_duration:
                continue

            speaker = int(label.split("_")[1]) if label.split("_")[1] else None

            if cur_speaker is None:
                cur_speaker = speaker
                cur_end = end
                continue

            if cur_speaker == speaker:
                cur_end = max(cur_end, end)
                continue

            segments.append({
                "speakers": [cur_speaker] if cur_speaker is not None else [],
                "start_time": round(cur_start, time_precision),
                "end_time": round(start, time_precision),
            })
            unique_speakers.add(cur_speaker)

            cur_speaker = speaker
            cur_start = start
            cur_end = end

        segments.append({
            "speakers": [cur_speaker] if cur_speaker is not None else [],
            "start_time": round(cur_start, time_precision),
            "end_time": round(duration, time_precision),
        })
        unique_speakers.add(cur_speaker)

        return self._relabel_speakers(segments, unique_speakers)

    def _relabel_speakers(
        self,
        segments: list[dict],
        unique_speakers: set[int],
    ) -> list[dict]:
        """
        Make speaker labels contiguous (e.g., 0,1,2 instead of 0,2,4).

        Returns
        -------
        list[dict]
            Segments with remapped speaker labels.
        """
        if not unique_speakers:
            return segments

        mapping = {old: new for new, old in enumerate(sorted(unique_speakers))}

        for segment in segments:
            segment["speakers"] = [mapping[s] for s in segment["speakers"]]

        return segments

    def cleanup(self) -> None:
        """
        Free up GPU memory and remove pipeline from memory.
        """
        del self.pipeline
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
