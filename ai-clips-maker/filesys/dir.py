"""
Handles directory-level operations within the local file system.
"""

import os
import shutil
import logging

from ai_clips_maker.filesys.object import FileSystemObject
from ai_clips_maker.filesys.file import File


class Dir(FileSystemObject):
    """
    Represents a directory in the local file system and provides utilities for
    managing its contents.
    """

    def __init__(self, dir_path: str) -> None:
        """
        Initialize a directory object.

        Parameters
        ----------
        dir_path : str
            Absolute path to the directory.
        """
        super().__init__(dir_path)

    def get_type(self) -> str:
        """Returns the object type as a string."""
        return "Dir"

    def check_exists(self) -> str | None:
        """Checks if the directory exists. Returns error message if not, otherwise None."""
        msg = super().check_exists()
        if msg:
            return msg
        if not os.path.isdir(self._path):
            return f"'{self._path}' is a valid {super().get_type()} but not a valid {self.get_type()}."
        return None

    def create(self) -> None:
        """Creates the directory. Ensures parent directory exists."""
        self.assert_does_not_exist()
        self.get_parent_dir().assert_exists()
        os.mkdir(self._path)

    def delete(self) -> None:
        """Deletes the directory and all its contents."""
        self.assert_exists()
        shutil.rmtree(self._path)
        logging.debug(f"Directory '{self._path}' deleted.")

    def move(self, new_path: str) -> None:
        """Moves the directory to a new location."""
        self.assert_exists()
        Dir(new_path).assert_does_not_exist()
        shutil.move(self._path, new_path)
        self._path = new_path
        logging.debug(f"Directory moved to '{new_path}'.")

    def get_parent_dir(self) -> Dir:
        """Returns the parent directory object."""
        return Dir(self.get_parent_dir_path())

    def scan_dir(self) -> list[FileSystemObject]:
        """Returns a list of all FileSystemObjects in the directory."""
        self.assert_exists()
        objects = []
        for entry in os.scandir(self._path):
            entry_path = os.path.join(self._path, entry.name)
            if entry.is_file():
                fs_object = File(entry_path)
            elif entry.is_dir():
                fs_object = Dir(entry_path)
            else:
                continue
            fs_object.assert_exists()
            objects.append(fs_object)
        return objects

    def get_files(self) -> list[File]:
        """Returns all file objects within the directory."""
        return [obj for obj in self.scan_dir() if isinstance(obj, File)]

    def get_subdirs(self) -> list[Dir]:
        """Returns all subdirectory objects within the directory."""
        return [obj for obj in self.scan_dir() if isinstance(obj, Dir)]

    def get_files_with_extension(self, extension: str) -> list[File]:
        """Returns all file objects in the directory with the given extension."""
        return [f for f in self.get_files() if f.get_file_extension() == extension]

    def get_file_paths_with_extension(self, extension: str) -> list[str]:
        """Returns paths of all files in the directory with the given extension."""
        return [f.path for f in self.get_files_with_extension(extension)]

    def zip(self, zip_file_name: str) -> File:
        """
        Zips the directory contents into a .zip file placed in the parent directory.

        Parameters
        ----------
        zip_file_name : str
            Name of the resulting zip file (without extension).

        Returns
        -------
        File
            File object of the generated zip file.
        """
        self.assert_exists()
        zip_path = shutil.make_archive(zip_file_name, "zip", self._path)
        final_path = os.path.join(self.get_parent_dir_path(), f"{zip_file_name}.zip")
        shutil.move(zip_path, final_path)
        return File(final_path)

    def delete_contents(self) -> None:
        """Deletes all contents inside the directory without deleting the directory itself."""
        for file in self.get_files():
            file.delete()
        for subdir in self.get_subdirs():
            subdir.delete()

    def delete_contents_except_asset(self) -> None:
        """
        Deletes all contents inside the directory except files
        starting with 'media_file_to_transcode'.
        """
        for file in self.get_files():
            if file.get_filename().startswith("media_file_to_transcode"):
                logging.debug(f"Skipping deletion of '{file.get_filename()}'")
                continue
            file.delete()
        for subdir in self.get_subdirs():
            subdir.delete()
