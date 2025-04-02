"""
Handles and extracts structured information from Python exceptions.
"""

# standard library imports
import sys
import traceback

# current package imports
from .exceptions import InvalidInputDataError

# local imports
from ai_clips_maker.transcribe.exceptions import NoSpeechError


class ExceptionHandler:
    """
    A class for converting Exceptions into CAI-specific status codes
    and extracting detailed stack trace information.
    """

    # Status code definitions
    SUCCESS = 0
    INVALID_INPUT_DATA = 1
    NO_SPEECH_ERROR = 2
    OTHER = 3

    def get_status_code(self, exception: Exception) -> int:
        """
        Maps an exception to a CAI-specific status code.

        Parameters
        ----------
        exception: Exception

        Returns
        -------
        int
            Status code associated with the exception.
        """
        if isinstance(exception, InvalidInputDataError):
            return self.INVALID_INPUT_DATA
        if isinstance(exception, NoSpeechError):
            return self.NO_SPEECH_ERROR
        return self.OTHER

    def get_stack_trace_info(self) -> list[str]:
        """
        Extracts and formats traceback information from the most recent exception.

        Returns
        -------
        list[str]
            A list of formatted strings containing traceback details.
        """
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is None:
            return ["No active exception."]

        formatted_trace = []
        summary = traceback.extract_tb(exc_tb)
        error_type = exc_type.__name__
        error_msg = str(exc_value)

        for frame in summary:
            message = (
                f"Error Type: {error_type} | "
                f"Filename: {frame.filename} | "
                f"Function: {frame.name} | "
                f"Line: {frame.lineno} | "
                f"Code: {frame.line!r} | "
                f"Message: {error_msg} | "
                + "#" * 10
            )
            formatted_trace.append(message)

        return formatted_trace
