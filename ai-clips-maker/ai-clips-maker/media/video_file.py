"""
Working with video files.

Notes
-----
- VideoFiles are defined to be files that contain only video and no other media.
"""

# standard library imports
from functools import lru_cache
import logging
from math import floor
from random import randint
import subprocess

# current package imports
from .exceptions import VideoFileError
from .image_file import ImageFile
from .temporal_media_file import TemporalMediaFile

# local imports
from ai_clips_maker.utils.conversions import seconds_to_hms_time_format


SUCCESS = 0


class VideoFile(TemporalMediaFile):
    """
    A class for working with video files.
    """

    def __init__(self, video_file_path: str) -> None:
        """
        Initialize VideoFile.

        Parameters
        ----------
        video_file_path: str
            absolute path to a video file.
        """
        super().__init__(video_file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'VideoFile' as a string.

        Returns
        -------
        str
            Object type 'VideoFile' as a string.
        """
        return "VideoFile"

    def check_exists(self) -> str or None:
        """
        Checks that the VideoFile exists in the file system. Returns None if so, a
        descriptive error message if not.

        Returns
        -------
        str or None
            None if the VideoFile exists in the file system, a descriptive error
            message if not.
        """
        msg = super().check_exists()
        if msg is not None:
            return msg

        temporal_media_file = TemporalMediaFile(self._path)
        if temporal_media_file.has_video_stream() is False:
            return (
                f"'{self._path}' is a valid {super().get_type()} but has no video stream, "
                f"so it is not a valid video file."
            )
        if temporal_media_file.is_video_only() is False:
            return (
                f"'{self._path}' is a valid {super().get_type()} but is not video-only. "
                f"Use 'AudioVideoFile' for files containing both audio and video."
            )

    @lru_cache(maxsize=1)
    def get_frame_rate(self) -> float:
        """
        Returns the frame rate of the video file.

        Returns
        -------
        float
            The frame rate of the video file.
        """
        frame_rate: str = self.get_stream_info("v:0", "r_frame_rate")
        numerator, denominator = map(int, frame_rate.split("/"))
        return numerator / denominator

    @lru_cache(maxsize=1)
    def get_height_pixels(self) -> int:
        """
        Returns the height in pixels of the video file.

        Returns
        -------
        int
            The height in pixels of the video file.
        """
        return int(self.get_stream_info("v:0", "height"))

    @lru_cache(maxsize=1)
    def get_width_pixels(self) -> int:
        """
        Returns the width in pixels of the video file.

        Returns
        -------
        int
            The width in pixels of the video file.
        """
        return int(self.get_stream_info("v:0", "width"))

    @lru_cache(maxsize=1)
    def get_bitrate(self) -> int or None:
        """
        Returns the bitrate in bits per second of the video file.

        Returns
        -------
        int
            The bitrate in bits per second of the video file.
        """
        return int(self.get_stream_info("v:0", "bit_rate"))

    def extract_frame(
        self,
        extract_sec: float,
        dest_image_file_path: str,
        overwrite: bool = True,
    ) -> ImageFile or None:
        """
        Extracts a frame at 'extract_sec' to 'dest_image_file_path'.

        Parameters
        ----------
        extract_sec: float
            The time (in seconds) at which to extract the frame.
        dest_image_file_path: str
            The path to save the extracted frame.
        overwrite: bool
            If True, overwrites the file; otherwise, does not overwrite.

        Returns
        -------
        ImageFile or None
            The extracted frame as an ImageFile if successful, None if unsuccessful.

        Raises
        ------
        VideoFileError: If extract_sec < 0 or extract_sec exceeds video duration.
        """
        self.assert_exists()
        if overwrite is True:
            self._filesys_manager.assert_parent_dir_exists(
                ImageFile(dest_image_file_path)
            )
        else:
            self._filesys_manager.assert_valid_path_for_new_fs_object(
                dest_image_file_path
            )
        self._filesys_manager.assert_paths_not_equal(
            self.path,
            dest_image_file_path,
            "video_file path",
            "dest_image_file_path",
        )

        if extract_sec < 0:
            msg = f"extract_sec ({extract_sec} seconds) cannot be negative."
            logging.error(msg)
            raise VideoFileError(msg)

        video_duration = self.get_duration()
        if video_duration == -1:
            msg = (
                f"Duration of video file '{self._path}' cannot be found. Continuing with "
                f"input of {extract_sec} seconds."
            )
            logging.warning(msg)
        elif extract_sec > video_duration:
            msg = (
                f"extract_sec ({extract_sec} seconds) cannot exceed video duration "
                f"({video_duration} seconds)."
            )
            logging.error(msg)
            raise VideoFileError(msg)

        extract_hms = seconds_to_hms_time_format(extract_sec)
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                extract_hms,
                "-i",
                self.path,
                "-frames:v",
                "1",
                "-q:v",
                "0",
                dest_image_file_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != SUCCESS:
            err_msg = (
                f"Extracting frame from video file '{self._path}' to '{dest_image_file_path}' "
                f"was unsuccessful. Details: {result.stderr}"
            )
            logging.error(err_msg)
            return None

        image_file = ImageFile(dest_image_file_path)
        image_file.assert_exists()
        return image_file

    def extract_thumbnail(
        self,
        thumbnail_file_path: str,
        overwrite: bool = True,
    ) -> ImageFile or None:
        """
        Extracts a thumbnail (image) from a random time between 30 seconds and 2 minutes
        into the video.

        Parameters
        ----------
        thumbnail_file_path: str
            The path to save the extracted thumbnail.
        overwrite: bool
            If True, overwrites the file at thumbnail_file_path; otherwise, does not overwrite.

        Returns
        -------
        ImageFile or None
            The extracted thumbnail if successful, None if unsuccessful.
        """
        self.assert_exists()

        video_duration = self.get_duration()
        if video_duration == -1:
            logging.warning(f"Unable to retrieve video duration from '{self._path}'. Using 120 seconds.")
            video_duration = 120

        max_time = min(120, floor(video_duration))
        min_time = max(min(30, floor(video_duration) - 30), 0)
        extract_sec = randint(min_time, max_time)

        image_file = self.extract_frame(
            extract_sec=extract_sec,
            dest_image_file_path=thumbnail_file_path,
            overwrite=overwrite,
        )

        if image_file is None:
            logging.error(f"Failed to extract thumbnail from '{self._path}' to '{thumbnail_file_path}'.")
            return None

        image_file.assert_exists()
        return image_file
