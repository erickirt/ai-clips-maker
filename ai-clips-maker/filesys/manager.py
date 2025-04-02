"""
Utilities for managing and validating operations in the local file system.
"""

# standard library imports
import logging
import re

# current package imports
from .exceptions import FileSystemObjectError
from .dir import Dir
from .object import FileSystemObject

# local imports
from ai_clips_maker.utils.type_checker import TypeChecker


class FileSystemManager:
    """
    High-level interface for managing and validating file system objects such as files and directories.
    """

    def __init__(self) -> None:
        """
        Initialize a new FileSystemManager instance.
        """
        self._type_checker = TypeChecker()

    def assert_paths_not_equal(
        self,
        path1: str,
        path2: str,
        path1_name: str,
        path2_name: str,
    ) -> None:
        """
        Ensures that two file system paths are not equal.

        Parameters
        ----------
        path1 : str
            First path to check.
        path2 : str
            Second path to check.
        path1_name : str
            Descriptive name for the first path (for error message).
        path2_name : str
            Descriptive name for the second path (for error message).

        Raises
        ------
        FileSystemObjectError: If both paths are equal.
        """
        if path1 == path2:
            msg = (
                f"{path1_name} with path '{path1}' is equal to {path2_name} with path '{path2}', "
                "but they are expected to be different."
            )
            logging.error(msg)
            raise FileSystemObjectError(msg)

    def check_valid_path_for_new_fs_object(self, path: str) -> str | None:
        """
        Validates whether the given path is suitable for a new file system object.

        Parameters
        ----------
        path : str
            Absolute path to validate.

        Returns
        -------
        str | None
            None if valid. Otherwise, an error message describing the issue.
        """
        fs_object = FileSystemObject(path)
        msg = fs_object.check_does_not_exist()
        if msg is not None:
            return msg

        parent_dir = Dir(fs_object.get_parent_dir_path())
        return parent_dir.check_exists()

    def is_valid_path_for_new_fs_object(self, path: str) -> bool:
        """
        Checks whether the provided path is valid for creating a new file or directory.

        Parameters
        ----------
        path : str
            Path to evaluate.

        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        return self.check_valid_path_for_new_fs_object(path) is None

    def assert_valid_path_for_new_fs_object(self, path: str) -> None:
        """
        Raises an error if the path is not valid for creating a new file or directory.

        Parameters
        ----------
        path : str
            Path to validate.

        Raises
        ------
        FileSystemObjectError: If the path is not valid.
        """
        msg = self.check_valid_path_for_new_fs_object(path)
        if msg is not None:
            logging.error(msg)
            raise FileSystemObjectError(msg)

    def check_parent_dir_exists(self, fs_object: FileSystemObject) -> str | None:
        """
        Checks whether the parent directory of a file system object exists.

        Parameters
        ----------
        fs_object : FileSystemObject
            The file system object to check.

        Returns
        -------
        str | None
            None if the parent directory exists, otherwise an error message.
        """
        parent_dir = Dir(fs_object.get_parent_dir_path())
        return parent_dir.check_exists()

    def parent_dir_exists(self, fs_object: FileSystemObject) -> bool:
        """
        Checks whether the parent directory of the given file system object exists.

        Parameters
        ----------
        fs_object : FileSystemObject

        Returns
        -------
        bool
            True if the parent directory exists, False otherwise.
        """
        return self.check_parent_dir_exists(fs_object) is None

    def assert_parent_dir_exists(self, fs_object: FileSystemObject) -> None:
        """
        Raises an error if the parent directory of the given file system object does not exist.

        Parameters
        ----------
        fs_object : FileSystemObject

        Raises
        ------
        FileSystemObjectError: If the parent directory does not exist.
        """
        msg = self.check_parent_dir_exists(fs_object)
        if msg is not None:
            logging.error(msg)
            raise FileSystemObjectError(msg)

    def filter_filename(self, filename: str) -> str:
        """
        Sanitizes a filename by removing characters that are invalid in most file systems.

        Invalid characters: \\/.,:*?"<>|

        Parameters
        ----------
        filename : str
            The original filename.

        Returns
        -------
        str
            A sanitized, file-system-safe version of the filename.
        """
        invalid_chars_pattern = r'[\\/.,:*?"<>|]'
        return re.sub(invalid_chars_pattern, "", filename)
