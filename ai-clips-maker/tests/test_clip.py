# Third-party imports
import pytest
from unittest.mock import MagicMock

# Local package imports
from ai_clips_maker.clip.clipfinder import ClipFinderConfigManager
from ai_clips_maker.clip.texttiler import TextTilerConfigManager
from ai_clips_maker.transcribe.transcription import Transcription


# Fixtures
@pytest.fixture
def clip_finder_config_manager():
    return ClipFinderConfigManager()

@pytest.fixture
def texttiler_config_manager():
    return TextTilerConfigManager()

@pytest.fixture
def valid_transcription():
    mock_transcription = MagicMock(spec=Transcription)
    mock_transcription.end_time = 800.0
    mock_transcription.get_sentence_info.return_value = [{"sentence": "Example sentence"}]
    return mock_transcription


# ----------------------------
# ClipFinderConfigManager Tests
# ----------------------------

def test_valid_clip_finder_config(clip_finder_config_manager: ClipFinderConfigManager):
    """
    Ensure valid configuration passes without raising errors.
    """
    config = {
        "cutoff_policy": "high",
        "embedding_aggregation_pool_method": "max",
        "min_clip_duration": 15,
        "max_clip_duration": 900,
        "smoothing_width": 3,
        "window_compare_pool_method": "mean",
    }
    assert clip_finder_config_manager.check_valid_config(config) is None

def test_invalid_clip_finder_config(clip_finder_config_manager: ClipFinderConfigManager):
    """
    Ensure invalid configuration returns error message.
    """
    config = {
        "cutoff_policy": "invalid_policy",
        "embedding_aggregation_pool_method": "invalid_method",
        "min_clip_duration": -5,
        "max_clip_duration": 5,
        "smoothing_width": 1,
        "window_compare_pool_method": "invalid_method",
    }
    result = clip_finder_config_manager.check_valid_config(config)
    assert isinstance(result, str)


# ----------------------------
# TextTilerConfigManager Tests
# ----------------------------

def test_valid_texttiler_config(texttiler_config_manager: TextTilerConfigManager):
    """
    Ensure TextTiler config with valid parameters passes validation.
    """
    config = {
        "k": 5,
        "cutoff_policy": "high",
        "embedding_aggregation_pool_method": "max",
        "smoothing_width": 3,
        "window_compare_pool_method": "mean",
    }
    assert texttiler_config_manager.check_valid_config(config) is None

def test_invalid_texttiler_config(texttiler_config_manager: TextTilerConfigManager):
    """
    Ensure invalid TextTiler config returns error string.
    """
    config = {
        "k": 1,
        "cutoff_policy": "invalid_policy",
        "embedding_aggregation_pool_method": "invalid_method",
        "smoothing_width": 1,
        "window_compare_pool_method": "invalid_method",
    }
    result = texttiler_config_manager.check_valid_config(config)
    assert isinstance(result, str)
