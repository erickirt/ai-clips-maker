"""
Defines a temporal and spatial segment within a video, optionally tied to speaker IDs.
"""


class Segment:
    """
    Represents a segment of a video, defined by its start and end times, spatial
    crop coordinates, and associated speaker IDs (if any).

    Attributes
    ----------
    speakers : list[int]
        List of speaker IDs associated with the segment.
    start_time : float
        Start time of the segment in seconds.
    end_time : float
        End time of the segment in seconds.
    x : int
        X-coordinate of the top-left corner of the crop area.
    y : int
        Y-coordinate of the top-left corner of the crop area.
    """

    def __init__(
        self,
        speakers: list[int],
        start_time: float,
        end_time: float,
        x: int,
        y: int,
    ) -> None:
        self._speakers = speakers
        self._start_time = start_time
        self._end_time = end_time
        self._x = x
        self._y = y

    @property
    def speakers(self) -> list[int]:
        """Returns the list of speaker IDs in this segment."""
        return self._speakers

    @property
    def start_time(self) -> float:
        """Returns the segment's start time in seconds."""
        return self._start_time

    @property
    def end_time(self) -> float:
        """Returns the segment's end time in seconds."""
        return self._end_time

    @property
    def x(self) -> int:
        """Returns the x-coordinate of the crop region."""
        return self._x

    @property
    def y(self) -> int:
        """Returns the y-coordinate of the crop region."""
        return self._y

    def copy(self) -> "Segment":
        """
        Returns a deep copy of the current Segment instance.

        Returns
        -------
        Segment
            New instance with duplicated attributes.
        """
        return Segment(
            speakers=self._speakers.copy(),
            start_time=self._start_time,
            end_time=self._end_time,
            x=self._x,
            y=self._y,
        )

    def to_dict(self) -> dict:
        """
        Serializes the segment to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the segment.
        """
        return {
            "speakers": self._speakers,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "x": self._x,
            "y": self._y,
        }

    def __str__(self) -> str:
        """Returns a readable string representation."""
        return (
            f"Segment(speakers={self._speakers}, "
            f"start={self._start_time}, end={self._end_time}, "
            f"position=({self._x}, {self._y}))"
        )

    def __repr__(self) -> str:
        """Returns a detailed string for debugging purposes."""
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        """Compares this segment to another for equality."""
        if not isinstance(other, Segment):
            return False
        return (
            self._speakers == other.speakers and
            self._start_time == other.start_time and
            self._end_time == other.end_time and
            self._x == other.x and
            self._y == other.y
        )

    def __ne__(self, other: object) -> bool:
        """Returns whether this segment is not equal to another."""
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        """
        Returns True if the segment is non-empty:
        - Contains at least one speaker
        - Has non-zero spatial and temporal bounds
        """
        return (
            bool(self._speakers) and
            self._start_time is not None and
            self._end_time is not None and
            self._x is not None and
            self._y is not None
        )
