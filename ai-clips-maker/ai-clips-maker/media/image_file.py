"""
Handles operations related to image files, extending MediaFile class functionality.
"""
from .media_file import MediaFile

class ImageFile(MediaFile):
    """
    A class for working specifically with image files.
    Inherits from MediaFile to manage image-related operations.
    """

    def __init__(self, image_file_path: str) -> None:
        """
        Initializes an ImageFile instance.

        Parameters
        ----------
        image_file_path : str
            The absolute path to the image file.

        Returns
        -------
        None
        """
        super().__init__(image_file_path)

    def get_type(self) -> str:
        """
        Returns the type of the object as a string.

        Returns
        -------
        str
            The object type, which is 'ImageFile'.
        """
        return "ImageFile"

    def check_exists(self) -> None:
        """
        Checks if the image file exists in the file system and ensures it’s a valid image file.

        If the file exists, returns None. Otherwise, it provides a descriptive error message.

        Returns
        -------
        str or None
            Returns None if the image exists, otherwise returns an error message.
        """
        # First check if it's a valid media file
        error_msg = super().check_exists()
        if error_msg:
            return error_msg

        # Ensure the file is specifically an image file
        media_file = MediaFile(self._path)
        if media_file.has_audio_stream():
            return f"'{self._path}' is a valid media file but contains audio, making it invalid as an ImageFile."

        return None

    def get_stream_info(self, stream_field: str) -> str or None:
        """
        Retrieves stream information specific to the image file.

        Parameters
        ----------
        stream_field : str
            The field of the stream you need information about:
            - 'duration' for duration in seconds
            - 'r_frame_rate' for the frame rate
            - 'width' for the horizontal resolution
            - 'height' for the vertical resolution
            - 'pix_fmt' for the pixel format
            - 'bit_rate' for the bit rate

        Returns
        -------
        str
            The requested stream information as a string.
        """
        self.assert_exists()  # Ensure the file exists before fetching stream info
        return super().get_stream_info("v:0", stream_field)
