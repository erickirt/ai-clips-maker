"""
Represents a defined segment of a media stream with time and text indexes.
"""

class MediaSegment:
    """
    A time-based and text-based segment of a video/audio file.
    """

    def __init__(
        self,
        begin_sec: float,
        finish_sec: float,
        text_start_idx: int,
        text_end_idx: int,
    ):
        """
        Initialize a MediaSegment object.

        Parameters
        ----------
        begin_sec: float
            Segment start in seconds.
        finish_sec: float
            Segment end in seconds.
        text_start_idx: int
            Start index in transcript.
        text_end_idx: int
            End index in transcript.
        """
        self._begin_sec = begin_sec
        self._finish_sec = finish_sec
        self._text_start_idx = text_start_idx
        self._text_end_idx = text_end_idx

    @property
    def begin_sec(self) -> float:
        return self._begin_sec

    @property
    def finish_sec(self) -> float:
        return self._finish_sec

    @property
    def text_start_idx(self) -> int:
        return self._text_start_idx

    @property
    def text_end_idx(self) -> int:
        return self._text_end_idx

    def clone(self) -> "MediaSegment":
        return MediaSegment(
            self._begin_sec,
            self._finish_sec,
            self._text_start_idx,
            self._text_end_idx
        )

    def to_dict(self) -> dict:
        return {
            "begin_sec": self._begin_sec,
            "finish_sec": self._finish_sec,
            "text_start_idx": self._text_start_idx,
            "text_end_idx": self._text_end_idx,
        }

    def __str__(self) -> str:
        return (
            f"MediaSegment(begin_sec={self._begin_sec}, finish_sec={self._finish_sec}, "
            f"text_start_idx={self._text_start_idx}, text_end_idx={self._text_end_idx})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MediaSegment):
            return False
        return (
            self._begin_sec == other.begin_sec
            and self._finish_sec == other.finish_sec
            and self._text_start_idx == other.text_start_idx
            and self._text_end_idx == other.text_end_idx
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        return (
            bool(self._begin_sec)
            and bool(self._finish_sec)
            and bool(self._text_start_idx)
            and bool(self._text_end_idx)
        )
