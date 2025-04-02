"""
Geometric rectangle representation with utility operations.
"""


class Rect:
    """
    Represents a rectangular region defined by its top-left corner (x, y) and dimensions.
    Commonly used for cropping or defining regions of interest (ROIs).
    """

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        """
        Initialize a Rect object.

        Parameters
        ----------
        x : int
            X-coordinate of the top-left corner.
        y : int
            Y-coordinate of the top-left corner.
        width : int
            Width of the rectangle.
        height : int
            Height of the rectangle.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self) -> str:
        """
        Returns a string representation of the rectangle.

        Returns
        -------
        str
            Rectangle as "(x, y, width, height)".
        """
        return f"({self.x}, {self.y}, {self.width}, {self.height})"

    def __eq__(self, other: object) -> bool:
        """
        Checks whether two Rect instances are equal.

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool
            True if all dimensions and positions are equal.
        """
        if not isinstance(other, Rect):
            return False
        return (
            self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
        )

    def __add__(self, other: "Rect") -> "Rect":
        """
        Adds two rectangles component-wise.

        Parameters
        ----------
        other : Rect
            Rectangle to add.

        Returns
        -------
        Rect
            Resulting rectangle after addition.
        """
        return Rect(
            self.x + other.x,
            self.y + other.y,
            self.width + other.width,
            self.height + other.height,
        )

    def __mul__(self, factor: float) -> "Rect":
        """
        Scales the rectangle by a factor.

        Parameters
        ----------
        factor : float
            The scale factor.

        Returns
        -------
        Rect
            Scaled rectangle.
        """
        return Rect(
            x=int(self.x * factor),
            y=int(self.y * factor),
            width=int(self.width * factor),
            height=int(self.height * factor),
        )

    def __truediv__(self, factor: float) -> "Rect":
        """
        Divides the rectangle's properties by a factor.

        Parameters
        ----------
        factor : float
            The division factor.

        Returns
        -------
        Rect
            Resulting rectangle after division.
        """
        return Rect(
            x=int(self.x / factor),
            y=int(self.y / factor),
            width=int(self.width / factor),
            height=int(self.height / factor),
        )
