"""
Handles operations on audio-only media files.
"""

import logging
import subprocess

from .temporal_media_file import TemporalMediaFile
from ai_clips_maker.filesys.file import File

SUCCESS = 0


class AudioFile(TemporalMediaFile):
    """
    Represents an audio-only media file.
    """

    def __init__(self, audio_file_path: str) -> None:
        super().__init__(audio_file_path)

    def get_type(self) -> str:
        return "AudioFile"

    def check_exists(self) -> str | None:
        msg = super().check_exists()
        if msg:
            return msg

        temp_file = TemporalMediaFile(self._path)
        if not temp_file.has_audio_stream():
            return f"'{self._path}' has no audio stream — not a valid {self.get_type()}."
        if not temp_file.is_audio_only():
            return (
                f"'{self._path}' is not audio-only — not a valid {self.get_type()}. "
                "Use AudioVideoFile instead."
            )

        return None

    def get_bitrate(self) -> int | None:
        return int(self.get_stream_info("a:0", "bit_rate"))

    def extract_audio(
        self,
        output_path: str,
        codec: str,
        overwrite: bool = True,
    ) -> AudioFile | None:
        self.assert_exists()

        if overwrite:
            self._filesys_manager.assert_parent_dir_exists(File(output_path))
        else:
            self._filesys_manager.assert_valid_path_for_new_fs_object(output_path)

        self._filesys_manager.assert_paths_not_equal(
            self.path,
            output_path,
            "source audio",
            "output audio",
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", self.path,
            "-c:a", codec,
            "-vn",  # strip video
            "-q:a", "0",  # highest quality
            "-map", "a",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != SUCCESS:
            logging.error(
                f"[extract_audio] Failed\nReturn code: {result.returncode}\n"
                f"Output: {result.stdout}\nError: {result.stderr}"
            )
            return None

        audio = AudioFile(output_path)
        audio.assert_exists()
        return audio
