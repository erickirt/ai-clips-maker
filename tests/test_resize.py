# Standard library imports
from unittest.mock import patch, MagicMock

# Local package imports
from ai_clips_maker.media.video_file import VideoFile
from ai_clips_maker.resize.resizer import Resizer
from ai_clips_maker.resize.rect import Rect

# Third-party imports
import pytest

# --- Tests for _calc_resize_width_and_height_pixels() ---
@pytest.mark.parametrize(
    "original_width, original_height, aspect_ratio, expected",
    [
        (1920, 1080, (9, 16), (607, 1080)),
        (1280, 720, (9, 16), (405, 720)),
        (1080, 1920, (16, 9), (1080, 607)),
        (720, 1280, (16, 9), (720, 405)),
        (1920, 1080, (1, 100), (10, 1080)),
        (1920, 1080, (100, 1), (1920, 19)),
        (1920, 1080, (16, 9), (1920, 1080)),
        (1280, 720, (16, 9), (1280, 720)),
        (320, 240, (4, 3), (320, 240)),
        (10, 10, (1, 1), (10, 10)),
        (8000, 4500, (16, 9), (8000, 4500)),
        (4500, 8000, (9, 16), (4500, 8000)),
    ],
)
def test_calc_resize_dimensions(original_width, original_height, aspect_ratio, expected):
    resizer = Resizer()
    result = resizer._calc_resize_width_and_height_pixels(
        original_width_pixels=original_width,
        original_height_pixels=original_height,
        resize_aspect_ratio=aspect_ratio,
    )
    assert result == expected

# --- Tests for _merge_scene_change_and_speaker_segments() ---
@pytest.mark.parametrize(
    "speaker_segments, scene_changes, expected",
    [
        ([{"speakers": [0], "start_time": 0, "end_time": 10}], [], [{"speakers": [0], "start_time": 0, "end_time": 10}]),
        ([{"speakers": [0], "start_time": 0, "end_time": 5}], [5], [{"speakers": [0], "start_time": 0, "end_time": 5}]),
        ([{"speakers": [0], "start_time": 0, "end_time": 10}], [5],
         [{"speakers": [0], "start_time": 0, "end_time": 5}, {"speakers": [0], "start_time": 5, "end_time": 10}]),
        ([{"speakers": [0], "start_time": 0, "end_time": 5}, {"speakers": [1], "start_time": 5, "end_time": 10}],
         [3, 8],
         [{"speakers": [0], "start_time": 0, "end_time": 3}, {"speakers": [0], "start_time": 3, "end_time": 5},
          {"speakers": [1], "start_time": 5, "end_time": 8}, {"speakers": [1], "start_time": 8, "end_time": 10}]),
        ([{"speakers": [0], "start_time": 0, "end_time": 5}, {"speakers": [1], "start_time": 5, "end_time": 10}],
         [5],
         [{"speakers": [0], "start_time": 0, "end_time": 5}, {"speakers": [1], "start_time": 5, "end_time": 10}]),
        ([{"speakers": [0], "start_time": 0, "end_time": 5}, {"speakers": [1], "start_time": 5, "end_time": 10}],
         [4.8],
         [{"speakers": [0], "start_time": 0, "end_time": 4.8}, {"speakers": [1], "start_time": 4.8, "end_time": 10}]),
        ([{"speakers": [0], "start_time": 0, "end_time": 5}, {"speakers": [1], "start_time": 5, "end_time": 10}],
         [5.1],
         [{"speakers": [0], "start_time": 0, "end_time": 5.1}, {"speakers": [1], "start_time": 5.1, "end_time": 10}]),
    ],
)
def test_merge_speaker_scene_segments(speaker_segments, scene_changes, expected):
    resizer = Resizer()
    result = resizer._merge_scene_change_and_speaker_segments(
        speaker_segments=speaker_segments,
        scene_changes=scene_changes,
        scene_merge_threshold=0.25,
    )
    assert result == expected

# --- Tests for _calc_n_batches() ---
@pytest.mark.parametrize(
    (
        "width, height, num_frames, gpu_available, face_detect_width,"
        "n_face_detect_batches, expected_batches"
    ),
    [
        (640, 480, 100, False, 960, 8, 1),
        (1920, 1080, 100, False, 960, 8, 1),
        (640, 480, 100, True, 960, 8, 8),
        (1920, 1080, 100, True, 960, 8, 8),
    ],
)
def test_calc_batches(width, height, num_frames, gpu_available, face_detect_width, n_face_detect_batches, expected_batches):
    mock_video = MagicMock(spec=VideoFile)
    mock_video.get_width_pixels.return_value = width
    mock_video.get_height_pixels.return_value = height

    resizer = Resizer()

    with patch("torch.cuda.is_available", return_value=gpu_available), patch(
        "clipsai.utils.pytorch.get_free_cpu_memory", return_value=8e9
    ):
        n_batches = resizer._calc_n_batches(
            video_file=mock_video,
            num_frames=num_frames,
            face_detect_width=face_detect_width,
            n_face_detect_batches=n_face_detect_batches,
        )
        assert n_batches == expected_batches

# --- Tests for _calc_crop() ---
@pytest.mark.parametrize(
    "roi, resize_width, resize_height, expected_crop",
    [
        (Rect(400, 300, 200, 200), 200, 200, Rect(400, 300, 200, 200)),
        (Rect(0, 0, 100, 100), 200, 200, Rect(0, 0, 200, 200)),
        (Rect(800, 600, 100, 100), 200, 200, Rect(750, 550, 200, 200)),
        (Rect(800, 600, 100, 100), 200, 400, Rect(750, 450, 200, 400)),
    ],
)
def test_crop_coordinates(roi, resize_width, resize_height, expected_crop):
    resizer = Resizer()
    crop = resizer._calc_crop(roi, resize_width, resize_height)
    assert crop == expected_crop

# --- Tests for _merge_identical_segments() ---
@pytest.mark.parametrize(
    "segments, expected",
    [
        ([{"x": 100, "y": 0, "start_time": 0, "end_time": 10},
          {"x": 200, "y": 0, "start_time": 10, "end_time": 20}],
         [{"x": 100, "y": 0, "start_time": 0, "end_time": 10},
          {"x": 200, "y": 0, "start_time": 10, "end_time": 20}]),
        ([{"x": 100, "y": 0, "start_time": 0, "end_time": 10},
          {"x": 100, "y": 0, "start_time": 10, "end_time": 20}],
         [{"x": 100, "y": 0, "start_time": 0, "end_time": 20}]),
        ([{"x": 100, "y": 0, "start_time": 0, "end_time": 10},
          {"x": 100, "y": 0, "start_time": 10, "end_time": 20},
          {"x": 100, "y": 0, "start_time": 20, "end_time": 30}],
         [{"x": 100, "y": 0, "start_time": 0, "end_time": 30}]),
        ([{"x": 100, "y": 0, "start_time": 0, "end_time": 10},
          {"x": 100, "y": 50, "start_time": 10, "end_time": 20}],
         [{"x": 100, "y": 0, "start_time": 0, "end_time": 10},
          {"x": 100, "y": 50, "start_time": 10, "end_time": 20}]),
        ([{"x": 100, "y": 0, "start_time": 0, "end_time": 10}], [{"x": 100, "y": 0, "start_time": 0, "end_time": 10}]),
        ([], []),
        ([{"x": 100, "y": 0, "start_time": 0, "end_time": 10},
          {"x": 101, "y": 0, "start_time": 10, "end_time": 20}],
         [{"x": 100, "y": 0, "start_time": 0, "end_time": 20}]),
    ],
)
def test_merge_identicals(segments, expected):
    mock_video = MagicMock(spec=VideoFile)
    mock_video.get_width_pixels.return_value = 1000
    mock_video.get_height_pixels.return_value = 1000

    resizer = Resizer()
    result = resizer._merge_identical_segments(segments, mock_video)
    assert result == expected
