# errors.py


class ConfigError(Exception):
    """
    Exception raised for errors in the configuration.

    Attributes
    ----------
        message -- explanation of the error (default "Invalid configuration")
    """

    def __init__(self, message="Invalid configuration"):
        self.message = message
        super().__init__(self.message)


class IdentifierError(Exception):
    """
    Exception raised when an identifier is invalid or problematic.

    Attributes
    ----------
        message -- explanation of the error (default "Invalid identifier")
    """

    def __init__(self, message="Invalid identifier"):
        self.message = message
        super().__init__(self.message)


class ModelNotFoundError(Exception):
    """
    Exception raised when a requested model cannot be found.

    Attributes
    ----------
        message -- explanation of the error (default "Model not found")
    """

    def __init__(self, message="Model not found"):
        self.message = message
        super().__init__(self.message)
