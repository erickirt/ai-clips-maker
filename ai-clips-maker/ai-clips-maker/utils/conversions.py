"""
Unit conversion utilities for time and memory units.
"""

# standard library imports
import math

# Time conversion constants
SECS_PER_SEC = 1
SECS_PER_MIN = 60
SECS_PER_HOUR = 3600
SECS_PER_DAY = 86400

# Memory unit constants
GIGA = 10**9           # Decimal GB
GIBI = 1024**3         # Binary GiB
NANO = 1e-9            # Nanosecond factor

# Index mapping for HMS parsing
SECS_IDX = 0
MINUTES_IDX = 1
HOURS_IDX = 2
DAYS_IDX = 3


def seconds_to_hms_time_format(seconds: float, num_digits: int = 3) -> str:
    """
    Converts seconds to 'HH:MM:SS.sss' format (human-readable time format).

    Parameters
    ----------
    seconds: float
        Duration in seconds.
    num_digits: int
        Decimal precision for seconds. Default is 3.

    Returns
    -------
    str
        Time formatted as 'HH:MM:SS.sss'
    """
    if num_digits < 0:
        raise ValueError(f"num_digits ({num_digits}) cannot be negative.")

    is_negative = seconds < 0
    seconds = abs(seconds)

    hours = int(seconds // 3600)
    remainder = seconds % 3600
    minutes = int(remainder // 60)
    secs = round(remainder % 60, num_digits)

    width = 3 + num_digits - (num_digits == 0)
    formatted = f"{hours:02}:{minutes:02}:{secs:0{width}.{num_digits}f}"

    return f"-{formatted}" if is_negative and (hours + minutes + secs) != 0 else formatted


def hms_time_format_to_seconds(hms_time: str) -> float:
    """
    Converts 'HH:MM:SS' time string to total seconds.

    Parameters
    ----------
    hms_time: str
        Time string in 'HH:MM:SS' format.

    Returns
    -------
    float
        Total time in seconds.
    """
    parts = [float(x) for x in hms_time.strip().split(":")]
    parts.reverse()

    factors = [SECS_PER_SEC, SECS_PER_MIN, SECS_PER_HOUR, SECS_PER_DAY]
    return sum(parts[i] * factors[i] for i in range(len(parts)))


def hours_to_seconds(hours: float) -> float:
    """
    Converts hours to seconds.
    """
    return hours * SECS_PER_HOUR


def seconds_to_hours(seconds: float) -> float:
    """
    Converts seconds to hours.
    """
    return seconds / SECS_PER_HOUR


def bytes_to_gigabytes(bytes_val: int) -> float:
    """
    Converts bytes to gigabytes (decimal).

    Returns
    -------
    float
        Gigabytes.
    """
    return bytes_val / GIGA


def gigabytes_to_bytes(gigabytes: float) -> int:
    """
    Converts gigabytes to bytes (decimal). Rounds up if needed.

    Returns
    -------
    int
        Bytes.
    """
    return math.ceil(gigabytes * GIGA)


def secs_to_nanosecs(seconds: float) -> int:
    """
    Converts seconds to nanoseconds.
    """
    return int(seconds / NANO)


def nano_secs_to_secs(nano_secs: int) -> float:
    """
    Converts nanoseconds to seconds.
    """
    return nano_secs * NANO


def bytes_to_gibibytes(bytes_val: int) -> float:
    """
    Converts bytes to gibibytes (binary).
    """
    return bytes_val / GIBI


def gibibytes_to_bytes(gibibytes: float) -> int:
    """
    Converts gibibytes to bytes (binary). Rounds up if needed.
    """
    return math.ceil(gibibytes * GIBI)