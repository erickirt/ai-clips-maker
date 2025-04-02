"""
Custom exceptions for the media package to handle various errors during media operations.
"""
# Local imports
from ai_clips_maker.filesys.exceptions import FileError

class MediaFileError(FileError):
    """
    Base exception for errors related to media files.
    Inherits from FileError to handle general file-related errors.
    """
    pass


class ImageFileError(MediaFileError):
    """
    Exception raised for errors specific to image file operations.
    Inherits from MediaFileError.
    """
    pass


class TemporalMediaFileError(MediaFileError):
    """
    Exception raised for errors specific to temporal media files (audio/video).
    Inherits from MediaFileError.
    """
    pass


class AudioFileError(TemporalMediaFileError):
    """
    Exception raised for errors specific to audio file operations.
    Inherits from TemporalMediaFileError.
    """
    pass


class VideoFileError(TemporalMediaFileError):
    """
    Exception raised for errors specific to video file operations.
    Inherits from TemporalMediaFileError.
    """
    pass


class AudioVideoFileError(TemporalMediaFileError):
    """
    Exception raised for errors specific to audio-video file operations.
    Inherits from TemporalMediaFileError.
    """
    pass


class MediaEditorError(Exception):
    """
    General exception raised for errors in media editing operations.
    Inherits directly from the base Exception class.
    """
    pass


class NoAudioStreamError(AudioFileError):
    """
    Exception raised when an audio file does not contain an audio stream.
    Inherits from AudioFileError.
    """
    pass


class NoVideoStreamError(VideoFileError):
    """
    Exception raised when a video file does not contain a video stream.
    Inherits from VideoFileError.
    """
    pass
