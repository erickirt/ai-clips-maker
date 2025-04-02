"""
Defines the core elements of a transcription: Sentence, Word, and Character.
Each element stores its textual content and its timing/position within the full transcription.
"""


class TranscriptionElement:
    """
    Base class for all transcription elements (sentence, word, character-like).
    Stores shared attributes such as text content and timing information.
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        start_char: int,
        end_char: int,
        text: str,
    ):
        self._start_time = start_time
        self._end_time = end_time
        self._start_char = start_char
        self._end_char = end_char
        self._text = text

    @property
    def start_time(self) -> float:
        """Start time (in seconds) of the element."""
        return self._start_time

    @property
    def end_time(self) -> float:
        """End time (in seconds) of the element."""
        return self._end_time

    @property
    def start_char(self) -> int:
        """Character index in full text where this element starts."""
        return self._start_char

    @property
    def end_char(self) -> int:
        """Character index in full text where this element ends."""
        return self._end_char

    @property
    def text(self) -> str:
        """The actual text of the element."""
        return self._text

    def to_dict(self) -> dict:
        """Returns the element attributes as a dictionary."""
        return {
            "start_time": self._start_time,
            "end_time": self._end_time,
            "start_char": self._start_char,
            "end_char": self._end_char,
            "text": self._text,
        }

    def __str__(self) -> str:
        """Returns string version (the text only)."""
        return self._text

    def __eq__(self, other: "TranscriptionElement") -> bool:
        """Check equality based on all fields."""
        return (
            self._start_time == other.start_time
            and self._end_time == other.end_time
            and self._start_char == other.start_char
            and self._end_char == other.end_char
            and self._text == other.text
        )

    def __ne__(self, other: object) -> bool:
        """Inequality operator (negates equality)."""
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        """Returns False if text is empty, True otherwise."""
        return bool(self._text)


class Sentence(TranscriptionElement):
    """
    Represents a sentence in the transcription.
    Inherits timing and text information from TranscriptionElement.
    """
    pass


class Word(TranscriptionElement):
    """
    Represents a word in the transcription.
    Inherits timing and text information from TranscriptionElement.
    """
    pass


class Character:
    """
    Represents a single character with timing and position metadata.
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        word_index: int,
        sentence_index: int,
        text: str,
    ):
        self._start_time = start_time
        self._end_time = end_time
        self._word_index = word_index
        self._sentence_index = sentence_index
        self._text = text

    @property
    def start_time(self) -> float:
        """Start time (in seconds) of the character."""
        return self._start_time

    @property
    def end_time(self) -> float:
        """End time (in seconds) of the character."""
        return self._end_time

    @property
    def word_index(self) -> int:
        """Index of the word this character belongs to."""
        return self._word_index

    @property
    def sentence_index(self) -> int:
        """Index of the sentence this character belongs to."""
        return self._sentence_index

    @property
    def text(self) -> str:
        """The character string (usually a single letter or punctuation)."""
        return self._text

    def to_dict(self) -> dict:
        """Returns the character's data as a dictionary."""
        return {
            "start_time": self._start_time,
            "end_time": self._end_time,
            "word_index": self._word_index,
            "sentence_index": self._sentence_index,
            "text": self._text,
        }

    def __str__(self) -> str:
        """String version (just the character)."""
        return self._text

    def __eq__(self, other: "Character") -> bool:
        """Equality check."""
        return (
            self._start_time == other.start_time
            and self._end_time == other.end_time
            and self._word_index == other.word_index
            and self._sentence_index == other.sentence_index
            and self._text == other.text
        )

    def __ne__(self, other: object) -> bool:
        """Inequality operator."""
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        """Returns True if character has content."""
        return bool(self._text)
