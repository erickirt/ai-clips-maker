import logging
import subprocess
import os
import uuid

# Imports from current package
from .exceptions import MediaEditorError
from .audio_file import AudioFile
from .audiovideo_file import AudioVideoFile
from .image_file import ImageFile
from .media_file import MediaFile
from .temporal_media_file import TemporalMediaFile
from .video_file import VideoFile

# Local imports for file and system management
from ai_clips_maker.filesys.file import File
from ai_clips_maker.filesys.manager import FileSystemManager
from ai_clips_maker.utils.conversions import seconds_to_hms_time_format
from ai_clips_maker.utils.type_checker import TypeChecker

SUCCESS = 0  # Return code from ffmpeg indicating success

class MediaEditor:
    """
    A class for handling media file editing operations using FFmpeg.
    It supports trimming, transcoding, watermarking, and other media file transformations.
    """

    def __init__(self) -> None:
        """
        Initializes the MediaEditor class with necessary file system and type checkers.
        """
        self._file_system_manager = FileSystemManager()
        self._type_checker = TypeChecker()

    def trim(
        self,
        media_file: TemporalMediaFile,
        start_time: float,
        end_time: float,
        trimmed_media_file_path: str,
        overwrite: bool = True,
        video_codec: str = "copy",
        audio_codec: str = "copy",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
        crop_width: int = None,
        crop_height: int = None,
        crop_x: int = None,
    ) -> TemporalMediaFile or None:
        """
        Trims and resizes a media file (video or audio) based on given start and end times.

        Parameters
        ----------
        media_file : TemporalMediaFile
            The media file to trim.
        start_time : float
            The start time in seconds for trimming.
        end_time : float
            The end time in seconds for trimming.
        trimmed_media_file_path : str
            Path to save the trimmed media file.
        overwrite : bool
            Whether to overwrite the existing file at `trimmed_media_file_path`.
        video_codec : str
            Codec used for video compression.
        audio_codec : str
            Codec used for audio compression.
        crf : str
            Constant rate factor for video quality.
        preset : str
            Preset for encoding speed and compression.
        num_threads : str
            The number of threads to use for encoding.
        crop_x, crop_y, crop_width, crop_height : Optional[int]
            If cropping is needed, these parameters define the crop area.

        Returns
        -------
        TemporalMediaFile or None
            A media file object representing the trimmed (and possibly resized) media.
        """
        # Validate the input media file
        self.assert_valid_media_file(media_file, TemporalMediaFile)
        if overwrite:
            self._file_system_manager.assert_parent_dir_exists(MediaFile(trimmed_media_file_path))
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(trimmed_media_file_path)
        self._file_system_manager.assert_paths_not_equal(media_file.path, trimmed_media_file_path, "media_file path", "trimmed_media_file_path")
        self._assert_valid_trim_times(media_file, start_time, end_time)

        # Prepare FFmpeg command
        duration_secs = end_time - start_time
        start_time_hms = seconds_to_hms_time_format(start_time)
        duration_hms = seconds_to_hms_time_format(duration_secs)

        ffmpeg_command = [
            "ffmpeg", "-y", "-ss", start_time_hms, "-t", duration_hms,
            "-i", media_file.path, "-c:v", video_codec, "-preset", preset,
            "-c:a", audio_codec, "-map", "0", "-crf", crf, "-threads", num_threads
        ]

        # Add cropping filter if specified
        if crop_width and crop_height and crop_x:
            logging.debug("Trim with resizing.")
            original_height = int(media_file.get_stream_info("v", "height"))
            crop_y = max(original_height // 2 - crop_height // 2, 0)
            crop_filter = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}"
            ffmpeg_command.extend(["-vf", crop_filter])

        # Output file path
        ffmpeg_command.append(trimmed_media_file_path)

        # Execute FFmpeg command
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        msg = f"Terminal return code: '{result.returncode}'\nOutput: '{result.stdout}'\nErr Output: '{result.stderr}'\n"
        if result.returncode != SUCCESS:
            logging.error(f"Trimming media file '{media_file.path}' to '{trimmed_media_file_path}' was unsuccessful. Details: {msg}")
            return None
        else:
            trimmed_media_file = self._create_media_file_of_same_type(trimmed_media_file_path, media_file)
            trimmed_media_file.assert_exists()
            return trimmed_media_file

    def copy_temporal_media_file(
        self,
        media_file: TemporalMediaFile,
        copied_media_file_path: str,
        overwrite: bool = True,
        video_codec: str = "copy",
        audio_codec: str = "copy",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
    ) -> TemporalMediaFile or None:
        """
        Copies a temporal media file (audio or video) by trimming it to its full duration.

        Parameters
        ----------
        media_file : TemporalMediaFile
            The media file to copy.
        copied_media_file_path : str
            The destination path for the copied media file.
        overwrite : bool
            Whether to overwrite the existing file at `copied_media_file_path`.
        video_codec : str
            Codec used for video compression.
        audio_codec : str
            Codec used for audio compression.
        crf : str
            Constant rate factor for video quality.
        preset : str
            Preset for encoding speed and compression.
        num_threads : str
            The number of threads to use for encoding.

        Returns
        -------
        TemporalMediaFile or None
            A media file object representing the copied media.
        """
        self.assert_valid_media_file(media_file, TemporalMediaFile)
        duration = media_file.get_duration()
        if duration == -1:
            logging.error(f"Can't retrieve duration from media file '{media_file.path}'")
            raise MediaEditorError(f"Can't retrieve duration from media file '{media_file.path}'")

        return self.trim(
            media_file,
            0,
            duration,
            copied_media_file_path,
            overwrite,
            video_codec,
            audio_codec,
            crf,
            preset,
            num_threads
        )

    def watermark_and_crop_video(
        self,
        video_file: VideoFile,
        watermark_file: ImageFile,
        watermarked_video_file_path: str,
        size_dim: str,
        watermark_to_video_ratio_size_dim: float,
        x: str,
        y: str,
        opacity: float,
        overwrite: bool = True,
        start_time: float = None,
        end_time: float = None,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
        crop_x: int = None,
        crop_y: int = None,
        crop_width: int = None,
        crop_height: int = None,
    ) -> VideoFile or None:
        """
        Watermarks a video with an image file and optionally crops the video.

        Parameters
        ----------
        video_file : VideoFile
            The video file to watermark.
        watermark_file : ImageFile
            The image file to watermark the video with.
        watermarked_video_file_path : str
            Path to save the watermarked video.
        size_dim : str
            Dimension ('h' or 'w') for scaling the watermark size relative to the video.
        watermark_to_video_ratio_size_dim : float
            Ratio of watermark size relative to the video size.
        x, y : str
            Coordinates for the watermark position.
        opacity : float
            The opacity of the watermark (0 to 1).
        overwrite : bool
            Whether to overwrite the existing file at `watermarked_video_file_path`.
        start_time, end_time : Optional[float]
            Start and end time for trimming the video.
        video_codec, audio_codec : str
            Codecs used for the video and audio streams.
        crf : str
            Constant rate factor for video compression.
        preset : str
            Encoding preset for compression.
        num_threads : str
            The number of threads to use for encoding.
        crop_x, crop_y, crop_width, crop_height : Optional[int]
            If cropping is needed, these parameters define the crop area.

        Returns
        -------
        VideoFile or None
            The watermarked video file if successful, None if unsuccessful.
        """
        self.assert_valid_media_file(video_file, VideoFile)
        self.assert_valid_media_file(watermark_file, ImageFile)

        # Validate paths and watermark properties
        if size_dim not in ["h", "w"]:
            raise MediaEditorError(f"Invalid size_dim '{size_dim}'. Must be 'h' or 'w'.")
        if watermark_to_video_ratio_size_dim <= 0:
            raise MediaEditorError(f"Watermark size ratio must be greater than zero, not '{watermark_to_video_ratio_size_dim}'.")
        if not (0 <= opacity <= 1):
            raise MediaEditorError(f"Opacity must be between 0 and 1, not '{opacity}'.")

        # Check trimming validity
        self._assert_valid_trim_times(video_file, start_time, end_time)

        # Prepare FFmpeg watermarking and cropping command
        return self.watermark_and_crop_video(
            video_file,
            watermark_file,
            watermarked_video_file_path,
            size_dim,
            watermark_to_video_ratio_size_dim,
            x,
            y,
            opacity,
            overwrite,
            start_time,
            end_time,
            video_codec,
            audio_codec,
            crf,
            preset,
            num_threads,
            crop_x,
            crop_y,
            crop_width,
            crop_height
        )

    def _create_media_file_of_same_type(
        self,
        file_path_to_create_media_file_from: str,
        media_file_to_copy_type_of: MediaFile,
    ) -> MediaFile:
        """
        Creates a new MediaFile object of the same type as the source media file.

        Parameters
        ----------
        file_path_to_create_media_file_from : str
            The path of the media file to create a new object from.
        media_file_to_copy_type_of : MediaFile
            The media file to copy the type of.

        Returns
        -------
        MediaFile
            A new MediaFile object of the same type as `media_file_to_copy_type_of`.
        """
        if isinstance(media_file_to_copy_type_of, VideoFile):
            return VideoFile(file_path_to_create_media_file_from)
        elif isinstance(media_file_to_copy_type_of, AudioFile):
            return AudioFile(file_path_to_create_media_file_from)
        elif isinstance(media_file_to_copy_type_of, ImageFile):
            return ImageFile(file_path_to_create_media_file_from)
        elif isinstance(media_file_to_copy_type_of, AudioVideoFile):
            return AudioVideoFile(file_path_to_create_media_file_from)
        else:
            raise MediaEditorError("Unsupported media file type.")
