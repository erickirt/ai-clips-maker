"""
Base class for representing objects (files, directories, etc.) in the local file system.
"""

import os
import logging
from ai_clips_maker.filesys.exceptions import FileSystemObjectError
from ai_clips_maker.utils.type_checker import TypeChecker


class FileSystemObject:
    """
    Base class for working with file system objects (like files and directories).

    Attributes
    ----------
    _path : str
        Absolute path of the object in the file system.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the FileSystemObject with an absolute path.

        Parameters
        ----------
        path : str
            Absolute path to the file system object.
        """
        self._type_checker = TypeChecker()
        self._type_checker.assert_type(path, "path", str)
        self._path = path

    @property
    def path(self) -> str:
        """Returns the absolute path of the object."""
        return self._path

    def get_path(self) -> str:
        """Alias for self.path"""
        return self._path

    def set_path(self, new_path: str) -> None:
        """
        Sets a new absolute path for the object.

        Parameters
        ----------
        new_path : str
            The new absolute path to assign.
        """
        self._type_checker.assert_type(new_path, "new_path", str)
        self._path = new_path

    def get_type(self) -> str:
        """
        Returns the object type.

        Returns
        -------
        str
            The type of the file system object (default: "FileSystemObject").
        """
        return "FileSystemObject"

    def get_parent_dir_path(self) -> str:
        """
        Returns the absolute path of the object's parent directory.

        Returns
        -------
        str
            Path to the parent directory.
        """
        return os.path.dirname(self._path)

    def check_exists(self) -> str | None:
        """
        Checks if the object exists in the file system.

        Returns
        -------
        str | None
            None if it exists, error message otherwise.
        """
        if not os.path.exists(self._path):
            return f"{self.get_type()} '{self._path}' does not exist."
        return None

    def assert_exists(self) -> None:
        """
        Raises an error if the object does not exist.
        """
        msg = self.check_exists()
        if msg:
            logging.error(msg)
            raise FileSystemObjectError(msg)

    def exists(self) -> bool:
        """
        Returns True if the object exists.

        Returns
        -------
        bool
        """
        return self.check_exists() is None

    def check_does_not_exist(self) -> str | None:
        """
        Checks if the object does not exist.

        Returns
        -------
        str | None
            None if it doesn't exist, error message otherwise.
        """
        if self.exists():
            return f"{self.get_type()} '{self._path}' already exists."
        return None

    def assert_does_not_exist(self) -> None:
        """
        Raises an error if the object already exists.
        """
        msg = self.check_does_not_exist()
        if msg:
            logging.error(msg)
            raise FileSystemObjectError(msg)
