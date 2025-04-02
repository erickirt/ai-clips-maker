"""
A utility class for working with JSON files in the local file system.
"""

# standard library imports
from __future__ import annotations
import json

# current package imports
from .file import File

# local imports
from ai_clips_maker.utils.type_checker import TypeChecker


class JSONFile(File):
    """
    A specialized File class for reading, writing, and managing JSON files.

    Inherits from:
    -------------
    File: Base class for generic file operations.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize a JSONFile instance with the provided file path.

        Parameters
        ----------
        file_path : str
            The absolute path to the JSON file.

        Raises
        ------
        ValueError: If the file path does not end with a .json extension.
        """
        super().__init__(file_path)

    def get_type(self) -> str:
        """
        Returns the string identifier for this object type.

        Returns
        -------
        str
            The string 'JSONFile'.
        """
        return "JSONFile"

    def check_exists(self) -> str | None:
        """
        Validates that the file exists and has a .json extension.

        Returns
        -------
        str | None
            None if the file exists and is valid.
            A descriptive error string otherwise.
        """
        msg = super().check_exists()
        if msg is not None:
            return msg

        if self.get_file_extension() != "json":
            return (
                f"'{self._path}' exists as a file, but is not a valid JSONFile "
                f"because it has extension '{self.get_file_extension()}' instead of 'json'."
            )

        return None

    def create(self, data: dict) -> None:
        """
        Creates a new JSON file with the specified dictionary content.

        Parameters
        ----------
        data : dict
            The content to be written to the JSON file.

        Raises
        ------
        TypeError: If the data is not a dictionary.
        """
        type_checker = TypeChecker()
        type_checker.assert_type(data, "data", dict)

        super().create(json.dumps(data, indent=4))
        self.assert_exists()

    def read(self) -> dict:
        """
        Reads the contents of the JSON file as a dictionary.

        Returns
        -------
        dict
            The content of the file parsed as a dictionary.

        Raises
        ------
        FileNotFoundError: If the file does not exist.
        JSONDecodeError: If the file content is not valid JSON.
        """
        self.assert_exists()

        with open(self._path, "r", encoding="utf-8") as file:
            return json.load(file)

    def write(self, new_data: dict) -> None:
        """
        Overwrites the JSON file with new dictionary content.

        Parameters
        ----------
        new_data : dict
            The new data to write into the file.

        Raises
        ------
        TypeError: If the input is not a dictionary.
        """
        self.assert_exists()

        type_checker = TypeChecker()
        type_checker.assert_type(new_data, "data", dict)

        with open(self._path, "w", encoding="utf-8") as file:
            json.dump(new_data, file, indent=4)
