"""
Transcriptions generated using WhisperX.

Notes
-----
- Character, word, and sentence level time stamps are available
- NLTK used for tokenizing sentences
- WhisperX GitHub: https://github.com/m-bain/whisperX
"""

from __future__ import annotations
from datetime import datetime
import logging

# Project-specific imports
from .exceptions import TranscriptionError
from .transcription_element import Sentence, Word, Character
from ai_clips_maker.filesys.json_file import JSONFile
from ai_clips_maker.filesys.manager import FileSystemManager
from ai_clips_maker.utils.type_checker import TypeChecker

# External dependencies
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


class Transcription:
    """
    Parses and processes WhisperX-generated transcription data.
    Allows querying and exporting character, word, and sentence level metadata.
    """

    def __init__(self, raw_transcription: dict | JSONFile) -> None:
        self._fs = FileSystemManager()
        self._type_checker = TypeChecker()
        self._type_checker.assert_type(raw_transcription, "transcription", (dict, JSONFile))

        self._source = None
        self._created = None
        self._lang = None
        self._speakers = None
        self._chars = None
        self._text = None
        self._words = None
        self._sentences = None

        if isinstance(raw_transcription, JSONFile):
            self._load_from_json(raw_transcription)
        else:
            self._load_from_dict(raw_transcription)

    @property
    def source(self) -> str:
        return self._source

    @property
    def created(self) -> datetime:
        return self._created

    @property
    def language(self) -> str:
        return self._lang

    @property
    def start_time(self) -> float:
        return 0.0

    @property
    def end_time(self) -> float:
        for ch in reversed(self._chars):
            if ch["end_time"] is not None:
                return ch["end_time"]
            if ch["start_time"] is not None:
                return ch["start_time"]

    @property
    def text(self) -> str:
        return self._text

    @property
    def characters(self) -> list[Character]:
        return [
            Character(
                start_time=ci["start_time"],
                end_time=ci["end_time"],
                word_index=ci["work_index"],
                sentence_index=ci["sentence_index"],
                text=ci["char"],
            ) for ci in self._chars
        ]

    @property
    def words(self) -> list[Word]:
        return [
            Word(
                start_time=wi["start_time"],
                end_time=wi["end_time"],
                start_char=wi["start_char"],
                end_char=wi["end_char"],
                text=wi["word"],
            ) for wi in self._words
        ]

    @property
    def sentences(self) -> list[Sentence]:
        return [
            Sentence(
                start_time=si["start_time"],
                end_time=si["end_time"],
                start_char=si["start_char"],
                end_char=si["end_char"],
            ) for si in self._sentences
        ]

    def get_char_info(self, start: float = None, end: float = None) -> list:
        self._validate_time_range(start, end)
        return self._slice_info(self._chars, start, end, self.find_char_index)

    def get_word_info(self, start: float = None, end: float = None) -> list:
        self._validate_time_range(start, end)
        return self._slice_info(self._words, start, end, self.find_word_index)

    def get_sentence_info(self, start: float = None, end: float = None) -> list:
        self._validate_time_range(start, end)
        return self._slice_info(self._sentences, start, end, self.find_sentence_index)

    def store_as_json_file(self, file_path: str) -> JSONFile:
        json_file = JSONFile(file_path)
        json_file.assert_has_file_extension("json")
        self._fs.assert_parent_dir_exists(json_file)
        json_file.delete()

        serialized_chars = [
            {"char": c["char"], "start_time": c["start_time"], "end_time": c["end_time"], "speaker": c["speaker"]}
            for c in self._chars
        ]

        json_file.create({
            "source_software": self._source,
            "time_created": str(self._created),
            "language": self._lang,
            "num_speakers": self._speakers,
            "char_info": serialized_chars,
        })

        return json_file

    def find_char_index(self, target: float, mode: str) -> int:
        return self._binary_search(self._chars, target, mode)

    def find_word_index(self, target: float, mode: str) -> int:
        return self._binary_search(self._words, target, mode)

    def find_sentence_index(self, target: float, mode: str) -> int:
        return self._binary_search(self._sentences, target, mode)

    def _load_from_json(self, file: JSONFile) -> None:
        self._type_checker.assert_type(file, "json_file", JSONFile)
        file.assert_exists()
        self._load_from_dict(file.read())

    def _load_from_dict(self, data: dict) -> None:
        self._validate_transcription_dict(data)
        if isinstance(data["time_created"], str):
            data["time_created"] = datetime.strptime(data["time_created"], "%Y-%m-%d %H:%M:%S.%f")

        self._source = data["source_software"]
        self._created = data["time_created"]
        self._lang = data["language"]
        self._speakers = data["num_speakers"]
        self._chars = data["char_info"]

        self._build_text()
        self._build_word_info()
        self._build_sentence_info()

    def _validate_transcription_dict(self, data: dict) -> None:
        self._type_checker.assert_dict_elems_type(data, {
            "source_software": str,
            "time_created": (datetime, str),
            "language": str,
            "num_speakers": (int, type(None)),
            "char_info": list,
        })
        for c in data["char_info"]:
            self._type_checker.assert_type(c, "char_info", dict)
            self._type_checker.are_dict_elems_of_type(c, {
                "char": str,
                "start_time": (float, type(None)),
                "end_time": (float, type(None)),
                "speaker": (int, type(None)),
            })

    def _build_text(self) -> None:
        self._text = "".join([c["char"] for c in self._chars])

    def _slice_info(self, items: list, start: float, end: float, index_fn) -> list:
        if start is None and end is None:
            return items
        start_idx = index_fn(start, "start")
        end_idx = index_fn(end, "end")
        return items[start_idx:end_idx + 1]

    def _validate_time_range(self, start: float, end: float) -> None:
        if type(start) is not type(end):
            raise TranscriptionError("start_time and end_time must be both float or None")
        if start is None and end is None:
            return
        if start < 0 or start >= end or end > self.end_time:
            raise TranscriptionError("Invalid time range: {} to {}".format(start, end))

    def _binary_search(self, data: list[dict], target: float, mode: str) -> int:
        left, right = 0, len(data) - 1
        while left <= right:
            mid = (left + right) // 2
            s, e = data[mid]["start_time"], data[mid]["end_time"]
            if s <= target <= e:
                return mid
            elif target > e:
                left = mid + 1
            else:
                right = mid - 1

        return (left - 1 if mode == "start" else right + 1)
