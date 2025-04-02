"""
Encapsulates cropping and resizing metadata for a video file.
"""

from .segment import Segment


class Crops:
    """
    Stores information related to video cropping and resizing operations,
    including original and target dimensions, and a list of cropped segments.

    Attributes
    ----------
    original_width : int
        Original width of the video.
    original_height : int
        Original height of the video.
    crop_width : int
        Target width after cropping.
    crop_height : int
        Target height after cropping.
    segments : list[Segment]
        List of segments representing cropped areas and their metadata.
    """

    def __init__(
        self,
        original_width: int,
        original_height: int,
        crop_width: int,
        crop_height: int,
        segments: list["Segment"],
    ) -> None:
        self._original_width = original_width
        self._original_height = original_height
        self._crop_width = crop_width
        self._crop_height = crop_height
        self._segments = segments

    @property
    def original_width(self) -> int:
        """Returns the original video width."""
        return self._original_width

    @property
    def original_height(self) -> int:
        """Returns the original video height."""
        return self._original_height

    @property
    def crop_width(self) -> int:
        """Returns the cropped video width."""
        return self._crop_width

    @property
    def crop_height(self) -> int:
        """Returns the cropped video height."""
        return self._crop_height

    @property
    def segments(self) -> list["Segment"]:
        """Returns the list of cropped video segments."""
        return self._segments

    def copy(self) -> "Crops":
        """
        Creates a deep copy of the Crops instance.

        Returns
        -------
        Crops
            A new instance with copied attributes and deep-copied segments.
        """
        return Crops(
            self._original_width,
            self._original_height,
            self._crop_width,
            self._crop_height,
            [segment.copy() for segment in self._segments],
        )

    def to_dict(self) -> dict:
        """
        Serializes the Crops instance into a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the Crops object.
        """
        return {
            "original_width": self._original_width,
            "original_height": self._original_height,
            "crop_width": self._crop_width,
            "crop_height": self._crop_height,
            "segments": [segment.to_dict() for segment in self._segments],
        }

    def __str__(self) -> str:
        """
        Returns a human-readable string summarizing the Crops instance.

        Returns
        -------
        str
            String with dimensions and segment details.
        """
        segment_list = ", ".join(str(s) for s in self._segments)
        return (
            f"Crops(Original: {self._original_width}x{self._original_height}, "
            f"Cropped: {self._crop_width}x{self._crop_height}, "
            f"Segments: [{segment_list}])"
        )

    def __eq__(self, other: object) -> bool:
        """
        Checks equality with another Crops instance.

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool
            True if equal, else False.
        """
        if not isinstance(other, Crops):
            return False
        return (
            self._original_width == other._original_width
            and self._original_height == other._original_height
            and self._crop_width == other._crop_width
            and self._crop_height == other._crop_height
            and self._segments == other._segments
        )

    def __ne__(self, other: object) -> bool:
        """Returns True if not equal to other Crops instance."""
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        """
        Returns whether the Crops instance has any segments.

        Returns
        -------
        bool
            True if segments exist, False otherwise.
        """
        return bool(self._segments)
