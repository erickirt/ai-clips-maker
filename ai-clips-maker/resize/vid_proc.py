"""
Video processing utilities: frame extraction and scene detection.
"""

# Standard library
from concurrent.futures import ThreadPoolExecutor
import logging

# Internal imports
from .exceptions import VideoProcessingError
from .img_proc import rgb_to_gray
from ai_clips_maker.media.video_file import VideoFile

# Third-party libraries
import av
import cv2
import numpy as np
from scenedetect import detect, AdaptiveDetector


def extract_frames(
    video_file: VideoFile,
    extract_secs: list[float],
    grayscale: bool = False,
    downsample_factor: float = 1.0,
) -> list[np.ndarray]:
    """
    Extracts specific frames from a video based on given time stamps.

    Parameters
    ----------
    video_file : VideoFile
        The video file to read from.
    extract_secs : list[float]
        List of timestamps (in seconds) to extract frames at.
    grayscale : bool, optional
        Whether to convert extracted frames to grayscale. Default is False.
    downsample_factor : float, optional
        Factor by which to downsample the extracted frames. Default is 1.0 (no downsampling).

    Returns
    -------
    list[np.ndarray]
        List of extracted frames as NumPy arrays.
    """
    duration = video_file.get_duration()
    for sec in extract_secs:
        if sec > duration:
            msg = f"Requested frame at {sec}s exceeds video duration {duration}s"
            logging.error(msg)
            raise VideoProcessingError(msg)

    container = av.open(video_file.path)
    stream = container.streams.video[0]
    target_pts = [int(sec / stream.time_base) for sec in extract_secs]

    frames = []
    for pts in target_pts:
        container.seek(pts, stream=stream)
        prev_frame = None
        for frame in container.decode(stream):
            if frame.pts > pts:
                frames.append(prev_frame or frame)
                break
            prev_frame = frame

    assert len(frames) == len(extract_secs)

    def process(frame):
        img = np.array(frame.to_image())
        if downsample_factor != 1.0:
            h = int(img.shape[0] / downsample_factor)
            w = int(img.shape[1] / downsample_factor)
            img = cv2.resize(img, (w, h))
        if grayscale:
            img = rgb_to_gray(img).reshape(img.shape[0], img.shape[1])
        return img

    with ThreadPoolExecutor() as executor:
        return list(executor.map(process, frames))


def detect_scenes(video_file: VideoFile, min_scene_duration: float = 0.25) -> list[float]:
    """
    Detects scene transitions in a video using an adaptive threshold.

    Parameters
    ----------
    video_file : VideoFile
        The video file to analyze.
    min_scene_duration : float, optional
        Minimum duration (in seconds) for a scene to be considered. Default is 0.25s.

    Returns
    -------
    list[float]
        List of timestamps (in seconds) where scene changes occur.
    """
    min_len_frames = int(min_scene_duration * video_file.get_frame_rate())
    detector = AdaptiveDetector(min_scene_len=min_len_frames)
    scene_list = detect(video_file.path, detector)

    return [
        round(scene[1].get_seconds(), 6)
        for scene in scene_list[:-1]  # exclude last scene end (EOF)
    ]
