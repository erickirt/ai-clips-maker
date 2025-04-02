"""
Custom exception classes used within the transcription system.

Hierarchy
---------
Exception
 └── TranscriberConfigError
      └── NoSpeechError
           └── TranscriptionError
                ├── AlignmentError
                ├── UnsupportedLanguageError
                ├── InvalidAudioFormatError
"""


class TranscriberConfigError(Exception):
    """
    Raised when the transcriber configuration is invalid.
    Example: wrong model size, invalid precision, or device error.
    """
    pass


class NoSpeechError(TranscriberConfigError):
    """
    Raised when the transcription engine detects no speech in the media file.
    Usually returned directly from WhisperX when no segments are found.
    """
    pass


class TranscriptionError(NoSpeechError):
    """
    Raised during runtime or logical failures in the transcription process.
    Acts as a general superclass for specific transcription errors.
    """
    pass


class AlignmentError(TranscriptionError):
    """
    Raised when word/character alignment fails or cannot be completed.
    Example: WhisperX alignment model fails to align segments properly.
    """
    pass


class UnsupportedLanguageError(TranscriptionError):
    """
    Raised when the requested language is not supported by the current model.
    Example: trying to transcribe in a non-supported ISO 639-1 language code.
    """
    pass


class InvalidAudioFormatError(TranscriptionError):
    """
    Raised when the provided audio file has an unsupported format or is unreadable.
    Example: corrupt audio file, wrong codec, missing audio stream.
    """
    pass
