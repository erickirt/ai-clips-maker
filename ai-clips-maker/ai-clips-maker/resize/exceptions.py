"""
Custom exception classes for errors raised within the resize package.
"""


class ResizerError(Exception):
    """
    Raised when a resizing operation fails, typically due to missing face data,
    invalid dimensions, or processing inconsistencies.
    """
    pass


class ImageProcessingError(Exception):
    """
    Raised when an image processing operation fails,
    such as grayscale conversion or array manipulation.
    """
    pass


class VideoProcessingError(Exception):
    """
    Raised when a video processing operation fails,
    such as frame extraction, decoding, or scene detection.
    """
    pass
