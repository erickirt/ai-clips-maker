"""
High-level pipeline for resizing a video to a target aspect ratio (default: 9:16).
Involves speaker diarization, scene detection, and facial region analysis.
"""

import logging

from .crops import Crops
from .resizer import Resizer
from .vid_proc import detect_scenes

from ai_clips_maker.diarize.pyannote import PyannoteDiarizer
from ai_clips_maker.media.audiovideo_file import AudioVideoFile


def resize(
    video_file_path: str,
    pyannote_auth_token: str,
    aspect_ratio: tuple[int, int] = (9, 16),
    min_segment_duration: float = 1.5,
    samples_per_segment: int = 13,
    face_detect_width: int = 960,
    face_detect_margin: int = 20,
    face_detect_post_process: bool = False,
    n_face_detect_batches: int = 8,
    min_scene_duration: float = 0.25,
    scene_merge_threshold: float = 0.25,
    time_precision: int = 6,
    device: str = None,
) -> Crops:
    """
    Resizes a video to the specified aspect ratio by aligning speaker diarization,
    face detection, and scene change information.

    Parameters
    ----------
    video_file_path : str
        Absolute path to the input video file.
    pyannote_auth_token : str
        HuggingFace authentication token for speaker diarization.
    aspect_ratio : tuple[int, int], optional
        Desired aspect ratio for the output video (width, height). Default is (9, 16).
    min_segment_duration : float, optional
        Minimum length (in seconds) for valid speaker segments. Default is 1.5.
    samples_per_segment : int, optional
        Number of face detection samples to extract per segment. Default is 13.
    face_detect_width : int, optional
        Downscale width used for face detection. Default is 960.
    face_detect_margin : int, optional
        Margin around detected faces. Default is 20.
    face_detect_post_process : bool, optional
        Apply smoothing post-process to facial crops. Default is False.
    n_face_detect_batches : int, optional
        Number of batches for batched face detection. Default is 8.
    min_scene_duration : float, optional
        Minimum scene length (in seconds) to avoid over-fragmentation. Default is 0.25.
    scene_merge_threshold : float, optional
        Merge tolerance between scene boundaries and speaker segments. Default is 0.25.
    time_precision : int, optional
        Decimal precision for timestamps. Default is 6.
    device : str, optional
        PyTorch device to use. E.g., 'cuda' or 'cpu'. If None, auto-detected.

    Returns
    -------
    Crops
        Object containing crop metadata, segments, and video resize parameters.
    """

    # Step 1: Load video and validate media type
    media = AudioVideoFile(video_file_path)
    media.assert_has_audio_stream()
    media.assert_has_video_stream()

    # Step 2: Speaker diarization (who speaks when)
    logging.debug(f"Running diarization on video: {media.get_filename()}")
    diarizer = PyannoteDiarizer(auth_token=pyannote_auth_token, device=device)
    diarized_segments = diarizer.diarize(
        media,
        min_segment_duration=min_segment_duration,
        time_precision=time_precision,
    )

    # Step 3: Scene change detection (detect cut points)
    logging.debug(f"Detecting scene changes: {media.get_filename()}")
    scene_changes = detect_scenes(media, min_duration=min_scene_duration)

    # Step 4: Resize video using speaker + scene + face info
    logging.debug(f"Resizing video: {media.get_filename()}")
    resizer = Resizer(
        face_detect_margin=face_detect_margin,
        face_detect_post_process=face_detect_post_process,
        device=device,
    )
    crops = resizer.resize(
        video_file=media,
        speaker_segments=diarized_segments,
        scene_changes=scene_changes,
        aspect_ratio=aspect_ratio,
        samples_per_segment=samples_per_segment,
        face_detect_width=face_detect_width,
        n_face_detect_batches=n_face_detect_batches,
        scene_merge_threshold=scene_merge_threshold,
    )

    # Step 5: Clean up intermediate files (if any)
    resizer.cleanup()
    return crops
