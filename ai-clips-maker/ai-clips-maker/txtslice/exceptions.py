"""
Custom exception types for the segmenting engine.
These ensure clarity in error handling throughout the pipeline.
"""


class ClipSegmentationError(Exception):
    """
    Base exception for all errors raised during the clip segmentation process.
    """
    pass


class TilingAlgorithmError(ClipSegmentationError):
    """
    Raised when the TextTiling algorithm fails due to invalid config or logic issues.
    """
    pass