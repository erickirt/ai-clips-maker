"""
Abstract base class for validating and managing configuration settings for ML components.
"""

# standard library imports
import abc

# current package imports
from .exceptions import ConfigError
from .type_checker import TypeChecker


class ConfigManager(abc.ABC):
    """
    Abstract base class for validating and managing config dictionaries
    used across ML-based components or services.
    """

    def __init__(self) -> None:
        """
        Initialize ConfigManager with internal type checker.
        """
        self._type_checker = TypeChecker()

    def impute_default_config(self, config: dict) -> dict:
        """
        Imputes missing config keys with default values.

        Parameters
        ----------
        config: dict
            Partial configuration dictionary.

        Returns
        -------
        dict
            Config dictionary filled with default values for missing keys.
        """
        return config  # To be overridden if needed

    @abc.abstractmethod
    def check_valid_config(self, config: dict) -> str or None:
        """
        Validates that a config dictionary is complete and consistent.

        Parameters
        ----------
        config: dict
            Configuration dictionary to validate.

        Returns
        -------
        str or None
            None if config is valid, otherwise a descriptive error message.
        """
        pass

    def is_valid_config(self, config: dict) -> bool:
        """
        Returns True if config is valid, False otherwise.

        Parameters
        ----------
        config: dict

        Returns
        -------
        bool
        """
        return self.check_valid_config(config) is None

    def assert_valid_config(self, config: dict) -> None:
        """
        Raises a ConfigError if config is invalid.

        Parameters
        ----------
        config: dict

        Raises
        ------
        ConfigError
        """
        error_msg = self.check_valid_config(config)
        if error_msg is not None:
            raise ConfigError(error_msg)
