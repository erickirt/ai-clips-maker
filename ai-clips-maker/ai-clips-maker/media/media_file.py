"""
Handles general media file operations (audio, video, image).
"""

import json
import logging
import subprocess

from ai_clips_maker.filesys.file import File
from ai_clips_maker.filesys.manager import FileSystemManager
from .exceptions import NoAudioStreamError, NoVideoStreamError

SUCCESS = 0
FALSE = 0


class MediaFile(File):
    """
    Base class for accessing and validating media files.
    """

    def __init__(self, media_path: str) -> None:
        super().__init__(media_path)
        self._fs_manager = FileSystemManager()

    def get_type(self) -> str:
        return "MediaFile"

    def check_exists(self) -> str | None:
        msg = super().check_exists()
        if msg is not None:
            return msg

        mime_type = File(self._path).get_mime_primary_type()
        if mime_type not in ["audio", "video", "image"]:
            return (
                f"'{self._path}' is not a valid MediaFile. Detected type: '{mime_type}'"
            )
        return None

    def get_format_info(self, field: str) -> str | None:
        self.assert_exists()
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", f"format={field}",
             "-of", "default=noprint_wrappers=1:nokey=1", self._path],
            capture_output=True, text=True
        )
        info = result.stdout.strip()
        if result.returncode != SUCCESS or not info:
            logging.error(f"ffprobe format query failed: {result.stderr.strip()}")
            return None
        return info

    def get_stream_info(self, stream: str, field: str) -> str | None:
        self.assert_exists()
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", stream,
             "-show_entries", f"stream={field}",
             "-of", "default=noprint_wrappers=1:nokey=1", self._path],
            capture_output=True, text=True
        )
        info = result.stdout.strip()
        if result.returncode != SUCCESS:
            logging.error(f"ffprobe stream query failed: {result.stderr.strip()}")
            return None
        return info

    def get_path(self) -> str:
        self.assert_exists()
        return self._path

    def get_streams(self) -> list[dict]:
        self.assert_exists()
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", self._path],
            capture_output=True, text=True
        )
        if result.returncode != SUCCESS:
            logging.error(f"ffprobe stream retrieval failed: {result.stderr.strip()}")
            return []

        return json.loads(result.stdout).get("streams", [])

    def get_audio_streams(self) -> list[dict]:
        return [s for s in self.get_streams() if s.get("codec_type") == "audio"]

    def get_video_streams(self) -> list[dict]:
        return [s for s in self.get_streams() if s.get("codec_type") == "video"]

    def check_has_audio_stream(self) -> str | None:
        if not self.get_audio_streams():
            return f"{self.get_type()} '{self._path}' has no audio stream."
        return None

    def assert_has_audio_stream(self) -> None:
        err = self.check_has_audio_stream()
        if err:
            raise NoAudioStreamError(err)

    def has_audio_stream(self) -> bool:
        return self.check_has_audio_stream() is None

    def has_video_stream(self) -> bool:
        for stream in self.get_video_streams():
            if stream.get("disposition", {}).get("attached_pic") != FALSE:
                return True
        return False

    def check_has_video_stream(self) -> str | None:
        if not self.has_video_stream():
            return f"{self.get_type()} '{self._path}' has no video stream."
        return None

    def assert_has_video_stream(self) -> None:
        err = self.check_has_video_stream()
        if err:
            raise NoVideoStreamError(err)

    def is_audio_only(self) -> bool:
        return self.has_audio_stream() and not self.has_video_stream()

    def is_video_only(self) -> bool:
        return self.has_video_stream() and not self.has_audio_stream()