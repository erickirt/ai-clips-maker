"""
Custom exceptions raised by the utils package.
"""


class ConfigError(Exception):
    """
    Raised when configuration settings are invalid or missing.
    """
    pass


class EnvironmentVariableNotSetError(Exception):
    """
    Raised when a required environment variable is not set.
    """
    pass


class InvalidComputeDeviceError(Exception):
    """
    Raised when the specified compute device is not valid or supported.
    """
    pass


class InvalidInputDataError(Exception):
    """
    Raised when input data is missing, malformed, or incompatible.
    """
    pass


class TimerError(Exception):
    """
    Raised when there is a misuse or failure in timing-related functionality.
    """
    pass
