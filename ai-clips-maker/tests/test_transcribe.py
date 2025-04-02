import pytest
from unittest.mock import patch
from datetime import datetime

from ai_clips_maker.filesys.json_file import JSONFile
from ai_clips_maker.media.audio_file import AudioFile
from ai_clips_maker.media.audiovideo_file import AudioVideoFile
from ai_clips_maker.media.editor import MediaEditor
from ai_clips_maker.media.exceptions import MediaEditorError
from ai_clips_maker.transcribe.exceptions import TranscriptionError
from ai_clips_maker.transcribe.transcriber import TranscriberConfigManager
from ai_clips_maker.transcribe.transcription import Transcription


# Fixture for TranscriberConfigManager instance
@pytest.fixture
def transcriber_config_manager():
    return TranscriberConfigManager()


# Fixture for MediaEditor instance
@pytest.fixture
def media_editor():
    return MediaEditor()


# ----------------------------
# TranscriberConfigManager Tests
# ----------------------------

def test_assert_valid_config(transcriber_config_manager: TranscriberConfigManager):
    config = {"language": "en", "precision": "float16", "model_size": "medium"}
    transcriber_config_manager.assert_valid_config(config)


# ----------------------------
# MediaEditor Tests
# ----------------------------

@patch("media.temporal_media_file.TemporalMediaFile.assert_exists")
def test_audio_file_initialization(mock_assert_exists, media_editor: MediaEditor):
    mock_assert_exists.return_value = None
    with patch("media.temporal_media_file.TemporalMediaFile.has_audio_stream", return_value=True), \
         patch("media.temporal_media_file.TemporalMediaFile.has_video_stream", return_value=False):
        result = media_editor.instantiate_as_temporal_media_file("audio.mp3")
    assert isinstance(result, AudioFile)


@patch("media.temporal_media_file.TemporalMediaFile.assert_exists")
def test_audiovideo_file_initialization(mock_assert_exists, media_editor: MediaEditor):
    mock_assert_exists.return_value = None
    with patch("media.temporal_media_file.TemporalMediaFile.has_audio_stream", return_value=True), \
         patch("media.temporal_media_file.TemporalMediaFile.has_video_stream", return_value=True):
        result = media_editor.instantiate_as_temporal_media_file("video.mp4")
    assert isinstance(result, AudioVideoFile)


@patch("media.temporal_media_file.TemporalMediaFile.assert_exists")
def test_invalid_file_initialization(mock_assert_exists, media_editor: MediaEditor):
    mock_assert_exists.return_value = None
    with patch("media.temporal_media_file.TemporalMediaFile.has_audio_stream", return_value=False), \
         patch("media.temporal_media_file.TemporalMediaFile.has_video_stream", return_value=False):
        with pytest.raises(MediaEditorError):
            media_editor.instantiate_as_temporal_media_file("invalid.file")


# ----------------------------
# Transcription Tests
# ----------------------------

valid_data = {
    "source_software": "TestSoftware",
    "time_created": datetime.now(),
    "language": "en",
    "num_speakers": 2,
    "char_info": [{"char": "H", "start_time": 0.0, "end_time": 0.2, "speaker": 1}],
}


def test_transcription_init_from_dict():
    transcription = Transcription(valid_data)
    assert transcription.language == "en"


def test_transcription_invalid_input():
    with pytest.raises(TypeError):
        Transcription("invalid_input")


def test_get_source_software():
    transcription = Transcription(valid_data)
    assert transcription.source_software == "TestSoftware"


def test_get_created_time():
    transcription = Transcription(valid_data)
    assert isinstance(transcription.created_time, datetime)


def test_get_char_info_range():
    transcription = Transcription(valid_data)
    result = transcription.get_char_info(start_time=0.0, end_time=0.2)
    assert len(result) == 1


def test_find_char_index():
    transcription = Transcription(valid_data)
    index = transcription.find_char_index(0.1, "start")
    assert index == 0


def test_invalid_char_info_times():
    transcription = Transcription(valid_data)
    with pytest.raises(TranscriptionError):
        transcription.get_char_info(start_time=-5, end_time=2)


def test_store_as_json():
    transcription = Transcription(valid_data)
    mock_file = JSONFile("output.json")

    with patch("filesys.json_file.JSONFile.assert_has_file_extension"), \
         patch("filesys.manager.FileSystemManager.assert_parent_dir_exists"), \
         patch("filesys.json_file.JSONFile.delete"), \
         patch("filesys.json_file.JSONFile.create", return_value=mock_file), \
         patch("filesys.json_file.JSONFile.assert_exists"):
        result = transcription.store_as_json_file("output.json")
        assert isinstance(result, JSONFile)
