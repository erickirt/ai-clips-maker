"""
Image processing utilities.
"""

import numpy as np


def rgb_to_gray(rgb_image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to grayscale using the standard luminosity method.

    Parameters
    ----------
    rgb_image : np.ndarray
        Input image in RGB format, as a 3D NumPy array.

    Returns
    -------
    np.ndarray
        Grayscale version of the input image as a 2D NumPy array.
    """
    weights = np.array([0.299, 0.587, 0.114])
    return (rgb_image @ weights).astype(np.uint8)


def calc_img_bytes(height: int, width: int, channels: int) -> int:
    """
    Calculates the memory usage in bytes for an image with given dimensions and channels.

    Parameters
    ----------
    height : int
        Image height in pixels.
    width : int
        Image width in pixels.
    channels : int
        Number of color channels (e.g., 3 for RGB).

    Returns
    -------
    int
        Estimated memory usage in bytes.
    """
    return height * width * channels
