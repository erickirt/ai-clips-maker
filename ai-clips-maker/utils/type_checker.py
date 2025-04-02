"""
Utility class for validating types of variables, lists, and dictionaries.
"""

import logging


class TypeChecker:
    """
    Utility class to perform strict type checks on various Python data structures.
    """

    def check_type(self, data, label: str, expected_types: tuple) -> str | None:
        """
        Check if 'data' is of one of the 'expected_types'. Returns error message if not.

        Parameters
        ----------
        data : any
            Variable to check.
        label : str
            Variable name (used in error messages).
        expected_types : tuple
            Accepted types.

        Returns
        -------
        str | None
            None if valid, otherwise error string.
        """
        if not isinstance(data, expected_types):
            return (
                f"'{label}' must be of type {expected_types}, "
                f"but got type {type(data)} with value '{data}'."
            )
        return None

    def assert_type(self, data, label: str, expected_types: tuple) -> None:
        """
        Raise error if 'data' is not of one of the 'expected_types'.

        Raises
        ------
        TypeError
        """
        msg = self.check_type(data, label, expected_types)
        if msg:
            logging.error(msg)
            raise TypeError(msg)

    def check_list_types(self, data: list, labels: list[str], expected_types: tuple) -> str | None:
        """
        Check each item in 'data' has a type in 'expected_types'.

        Raises
        ------
        ValueError: If lengths of data and labels mismatch
        """
        self.assert_type(data, "data", (list,))
        self.assert_type(labels, "labels", (list,))
        if len(data) != len(labels):
            msg = f"'data' and 'labels' must be the same length. Got {len(data)} and {len(labels)}"
            logging.error(msg)
            raise ValueError(msg)

        for item, label in zip(data, labels):
            msg = self.check_type(item, label, expected_types)
            if msg:
                return msg
        return None

    def assert_list_elems_type(self, data: list, labels: list[str], expected_types: tuple) -> None:
        """
        Assert all elements in list are of valid types.

        Raises
        ------
        TypeError
        """
        msg = self.check_list_types(data, labels, expected_types)
        if msg:
            logging.error(msg)
            raise TypeError(msg)

    def check_dict_types(self, data: dict, expected_types: dict) -> str | None:
        """
        Check each element in dict has type specified in expected_types.

        Raises
        ------
        KeyError: if dict structure mismatch
        """
        data_keys = set(data.keys())
        type_keys = set(expected_types.keys())

        missing = type_keys - data_keys
        if missing:
            msg = f"Missing keys in 'data': {missing}"
            logging.error(msg)
            raise KeyError(msg)

        for key in type_keys:
            msg = self.check_type(data[key], key, expected_types[key])
            if msg:
                return msg
        return None

    def assert_dict_elems_type(self, data: dict, expected_types: dict) -> None:
        """
        Assert dictionary values have valid types.

        Raises
        ------
        TypeError
        """
        msg = self.check_dict_types(data, expected_types)
        if msg:
            logging.error(msg)
            raise TypeError(msg)

    def are_list_elems_of_type(self, data: list, expected_types: tuple) -> bool:
        """
        Check if all elements in a list are of the given type(s).
        """
        return all(isinstance(item, expected_types) for item in data)

    def are_dict_elems_of_type(self, data: dict, expected_types: dict) -> bool:
        """
        Check if all elements in a dict match expected types.
        """
        try:
            self.assert_dict_elems_type(data, expected_types)
            return True
        except TypeError:
            return False