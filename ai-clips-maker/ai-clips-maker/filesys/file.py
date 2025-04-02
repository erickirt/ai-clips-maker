"""
Handles file-level operations within the local file system.

Notes
-----
- MIME type info: https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types
"""

import os
import logging
import magic
from ai_clips_maker.filesys.object import FileSystemObject
from ai_clips_maker.filesys.exceptions import FileError


class File(FileSystemObject):
    """
    Represents a file in the local file system and provides basic utilities.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize a File object.

        Parameters
        ----------
        file_path : str
            Absolute path of the file.
        """
        super().__init__(file_path)

    def get_type(self) -> str:
        """Returns the type of the object."""
        return "File"

    def get_filename(self) -> str:
        """Returns the file name with extension."""
        return os.path.basename(self._path)

    def get_filename_without_extension(self) -> str:
        """Returns the file name without its extension."""
        return os.path.splitext(self.get_filename())[0]

    def get_file_extension(self) -> str | None:
        """Returns the file extension, or None if it doesn't exist."""
        _, ext = os.path.splitext(self._path)
        return ext[1:] if ext else None

    def get_file_size(self) -> int:
        """Returns file size in bytes."""
        self.assert_exists()
        return os.path.getsize(self._path)

    def get_mime_type(self) -> str:
        """Returns the full MIME type of the file (e.g., text/plain)."""
        self.assert_exists()
        mime = magic.Magic(mime=True)
        return mime.from_file(self._path)

    def get_mime_primary_type(self) -> str:
        """Returns the primary MIME type (e.g., 'text' from 'text/plain')."""
        return self.get_mime_type().split("/")[0]

    def get_mime_secondary_type(self) -> str:
        """Returns the secondary MIME type (e.g., 'plain' from 'text/plain')."""
        return self.get_mime_type().split("/")[1]

    def check_exists(self) -> str | None:
        """Checks if the file exists and is a file. Returns None if valid, error message otherwise."""
        msg = super().check_exists()
        if msg is not None:
            return msg
        if not os.path.isfile(self._path):
            return f"'{self._path}' exists but is not a valid file."
        return None

    def create(self, data: str) -> None:
        """Creates a new file with the given content. Raises if file already exists."""
        self.assert_does_not_exist()
        with open(self._path, "x") as f:
            f.write(data)
        self.assert_exists()

    def delete(self) -> None:
        """Deletes the file if it exists."""
        if not self.exists():
            logging.warning(f"File '{self._path}' does not exist.")
            return
        os.remove(self._path)
        logging.debug(f"File '{self._path}' deleted.")

    def move(self, new_path: str) -> None:
        """Moves the file to a new location."""
        self.assert_exists()
        File(new_path).assert_does_not_exist()
        os.rename(self._path, new_path)
        self._path = new_path

    def check_has_file_extension(self, extension: str) -> str | None:
        """Returns error message if file doesn't have the expected extension, None if okay."""
        current_ext = self.get_file_extension()
        if current_ext != extension:
            return f"'{self._path}' should have extension '{extension}' not '{current_ext}'."

    def has_file_extension(self, extension: str) -> bool:
        """Checks if the file has the specified extension."""
        return self.check_has_file_extension(extension) is None

    def assert_has_file_extension(self, extension: str) -> None:
        """Raises FileError if file doesn't have the specified extension."""
        msg = self.check_has_file_extension(extension)
        if msg:
            logging.error(msg)
            raise FileError(msg)
