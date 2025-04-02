"""
General-purpose utility functions used across the project.
"""


def find_missing_dict_keys(data: dict, required_keys: list[str]) -> list[str]:
    """
    Returns the list of keys that are missing in 'data' from 'required_keys'.

    Parameters
    ----------
    data : dict
        Dictionary to check.
    required_keys : list[str]
        Keys that are expected to be present in 'data'.

    Returns
    -------
    list[str]
        Keys that are in 'required_keys' but missing from 'data'.
    """
    return [key for key in required_keys if key not in data]
