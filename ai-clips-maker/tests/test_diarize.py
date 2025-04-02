# Standard library imports
from unittest.mock import patch, Mock

# Local imports
from ai_clips_maker.diarize.pyannote import PyannoteDiarizer

# Third-party imports
import pandas as pd
from pyannote.core import Segment, Annotation
import pytest


@pytest.fixture
def mock_diarizer():
    """
    Creates a mocked instance of the PyannoteDiarizer, replacing the pipeline
    with a mock object to avoid actual model loading or inference.
    """
    with patch("pyannote.audio.Pipeline.from_pretrained", return_value=Mock()):
        diarizer = PyannoteDiarizer(auth_token="mock_token")
        diarizer.pipeline = Mock()
        return diarizer


@pytest.fixture
def mock_audio_file():
    """
    Creates a mocked audio file with mocked path and duration.
    """
    mock_audio_file = Mock()
    mock_audio_file.path.return_value = "mock_audio.mp3"
    mock_audio_file.get_duration.return_value = 30.0
    return mock_audio_file


@pytest.mark.parametrize(
    "annotation_data, expected_output",
    [
        # Test 1: Gaps between speaker segments
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
                {"segment": Segment(12, 20), "label": "speaker_1", "track": "_"},
                {"segment": Segment(21, 30), "label": "speaker_0", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 12},
                {"speakers": [1], "start_time": 12, "end_time": 21},
                {"speakers": [0], "start_time": 21, "end_time": 30},
            ],
        ),
        # Test 2: Overlapping segments between speakers
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
                {"segment": Segment(8, 12), "label": "speaker_2", "track": "_"},
                {"segment": Segment(10, 20), "label": "speaker_1", "track": "_"},
                {"segment": Segment(20, 30), "label": "speaker_0", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 8},
                {"speakers": [2], "start_time": 8, "end_time": 10},
                {"speakers": [1], "start_time": 10, "end_time": 20},
                {"speakers": [0], "start_time": 20, "end_time": 30},
            ],
        ),
        # Test 3: Ignore segments shorter than minimum duration
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
                {"segment": Segment(11, 20), "label": "speaker_1", "track": "_"},
                {"segment": Segment(15, 16), "label": "speaker_1", "track": "_"},  # Too short
                {"segment": Segment(21, 30), "label": "speaker_0", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 11},
                {"speakers": [1], "start_time": 11, "end_time": 21},
                {"speakers": [0], "start_time": 21, "end_time": 30},
            ],
        ),
        # Test 4: Merge continuous segments from the same speaker
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
                {"segment": Segment(10, 12), "label": "speaker_1", "track": "_"},
                {"segment": Segment(12, 15), "label": "speaker_1", "track": "_"},
                {"segment": Segment(15, 20), "label": "speaker_1", "track": "_"},
                {"segment": Segment(20, 30), "label": "speaker_0", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 10},
                {"speakers": [1], "start_time": 10, "end_time": 20},
                {"speakers": [0], "start_time": 20, "end_time": 30},
            ],
        ),
        # Test 5: No segments at all
        (
            [],
            [{"speakers": [], "start_time": 0, "end_time": 30}],
        ),
        # Test 6: Relabel discontiguous speaker labels to be continuous
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_2", "track": "_"},
                {"segment": Segment(10, 20), "label": "speaker_5", "track": "_"},
                {"segment": Segment(20, 30), "label": "speaker_2", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 10},
                {"speakers": [1], "start_time": 10, "end_time": 20},
                {"speakers": [0], "start_time": 20, "end_time": 30},
            ],
        ),
        # Test 7: No relabeling needed if labels are already contiguous
        (
            [
                {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
                {"segment": Segment(10, 20), "label": "speaker_1", "track": "_"},
                {"segment": Segment(20, 30), "label": "speaker_0", "track": "_"},
            ],
            [
                {"speakers": [0], "start_time": 0, "end_time": 10},
                {"speakers": [1], "start_time": 10, "end_time": 20},
                {"speakers": [0], "start_time": 20, "end_time": 30},
            ],
        ),
        # Test 8: Handle unlabeled speaker segments
        (
            [{"segment": Segment(0, 30), "label": "_", "track": "_"}],
            [{"speakers": [], "start_time": 0, "end_time": 30}],
        ),
    ],
)
def test_diarize(mock_diarizer, mock_audio_file, annotation_data, expected_output):
    """
    Parametrized test that checks if the diarizer correctly processes a variety
    of annotated speaker segment scenarios and produces the expected output.
    """
    if not annotation_data:
        # Handle the case with no speaker segments
        annotation = Annotation()
    else:
        # Convert test data into a Pyannote Annotation object
        df = pd.DataFrame(annotation_data)
        annotation = Annotation().from_df(df)

    mock_diarizer.pipeline.return_value = annotation

    # Run diarization
    output_segments = mock_diarizer.diarize(mock_audio_file)

    # Check output
    assert output_segments == expected_output
