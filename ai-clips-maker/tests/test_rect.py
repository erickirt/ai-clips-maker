"""
Unit tests for the Rect class used for representing rectangular regions.
"""

# local imports
from ai_clips_maker.resize.rect import Rect


def test_rect_initialization():
    rect = Rect(1, 2, 3, 4)
    assert rect.x == 1
    assert rect.y == 2
    assert rect.width == 3
    assert rect.height == 4


def test_rect_str_representation():
    rect = Rect(1, 2, 3, 4)
    assert str(rect) == "(1, 2, 3, 4)"


def test_rect_add_operator():
    rect1 = Rect(1, 2, 3, 4)
    rect2 = Rect(5, 6, 7, 8)
    result = rect1 + rect2
    assert result == Rect(6, 8, 10, 12)


def test_rect_mul_operator():
    rect = Rect(1, 2, 3, 4)
    result = rect * 2
    assert result == Rect(2, 4, 6, 8)


def test_rect_div_operator():
    rect = Rect(2, 4, 6, 8)
    result = rect / 2
    assert result == Rect(1, 2, 3, 4)


def test_rect_equality_check():
    assert Rect(1, 2, 3, 4) == Rect(1, 2, 3, 4)
    assert Rect(1, 2, 3, 4) != Rect(0, 0, 0, 0)