"""
Handling temporal media files (audio/video with time-based properties).
"""

# standard library imports
import logging

# current package imports
from .media_file import MediaFile


class TemporalMediaFile(MediaFile):
    """
    A class for managing media files that contain time-based streams such as audio or video.
    """

    def __init__(self, media_file_path: str) -> None:
        """
        Initialize TemporalMediaFile.

        Parameters
        ----------
        media_file_path : str
            Absolute path to a temporal media file.
        """
        super().__init__(media_file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'TemporalMediaFile'.

        Returns
        -------
        str
        """
        return "TemporalMediaFile"

    def check_exists(self) -> str | None:
        """
        Checks if the file exists and is a valid temporal media file (contains audio or video stream).

        Returns
        -------
        str | None
            Returns None if valid, otherwise a descriptive error message.
        """
        msg = super().check_exists()
        if msg is not None:
            return msg

        media_file = MediaFile(self._path)
        if not media_file.has_audio_stream() and not media_file.has_video_stream():
            return (
                f"'{self._path}' is a valid {super().get_type()} but has neither audio "
                f"nor video stream, so it is not a valid {self.get_type()}."
            )

        return None

    def get_duration(self) -> float:
        """
        Retrieves duration of the media in seconds.

        Returns
        -------
        float
            Duration in seconds. Returns -1 if unavailable.
        """
        self.assert_exists()

        duration_str = self.get_format_info("duration")
        if duration_str is None:
            logging.error(f"Failed to retrieve duration for media file '{self._path}'.")
            return -1

        return float(duration_str)

    def get_bitrate(self, stream: str) -> int | None:
        """
        Retrieves the bitrate of a specific stream.

        Parameters
        ----------
        stream : str
            Stream specifier: 'a:0' for audio, 'v:0' for video.

        Returns
        -------
        int | None
            Bitrate of the stream. None if unavailable.
        """
        self.assert_exists()

        bitrate = self.get_stream_info(stream, "bit_rate")
        if bitrate is None:
            logging.error(f"Could not retrieve bitrate from stream '{stream}' in '{self._path}'.")
            return None

        return int(bitrate)
