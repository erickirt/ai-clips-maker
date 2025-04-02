"""
Custom exceptions raised by the filesys package for file system operations.
"""


class FileSystemObjectError(OSError):
    """
    Base exception for all file system object errors.
    Raised when any general issue occurs with a file or directory.
    """
    pass


class FileError(FileSystemObjectError):
    """
    Raised when a file-specific error occurs (e.g. read, write, delete).
    """
    pass


class JsonFileError(FileError):
    """
    Raised when a JSON file operation fails (e.g. invalid structure, extension mismatch).
    """
    pass


class DirError(FileSystemObjectError):
    """
    Raised when a directory-specific error occurs (e.g. missing dir, permission issues).
    """
    pass
