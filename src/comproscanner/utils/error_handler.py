"""
error_handler.py - Contains custom error handlers for the project

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-02-2025
"""

# Standard library imports
import sys


class TextColours:
    """ANSI colour codes used when printing error messages to the terminal."""

    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


class BaseError(Exception):
    """Base class for other exceptions"""

    def __init__(self, message=None, color=TextColours.FAIL):
        self.message = (
            message if message is not None else "An unexpected error occurred."
        )
        super().__init__(self.message)
        print(f"{color}{self.message}{TextColours.ENDC}")

    def exit_program(self):
        """Separate method for exiting to allow mocking in tests"""
        sys.exit()


class KeyboardInterruptHandler(BaseError):
    """Raised when the user hits the interrupt key"""

    def __init__(self, message=None):
        """
        Args:
            message (str, optional): Additional context appended to the standard interrupt message.
        """
        message = (
            f"Keyboard Interruption detected. Exiting the program... {message}"
            if message
            else "Keyboard Interruption detected. Exiting the program..."
        )
        super().__init__(message, color=TextColours.FAIL)
        self.exit_program()


class FileNotFoundErrorHandler(BaseError):
    """Raised when a file is not found"""

    def __init__(self, message=None):
        """
        Args:
            message (str, optional): Path or description of the missing file.
        """
        message = f"File not found. {message}" if message else "File not found."
        super().__init__(message)
        self.exit_program()


class ValueErrorHandler(BaseError):
    """Raised when a built-in operation or function receives an argument that has the right type but an inappropriate value"""

    def __init__(self, message=None):
        """
        Args:
            message (str, optional): Description of the invalid value or constraint violated.
        """
        message = (
            f"ValueErrorHandler: {message}" if message else "Value error occurred."
        )
        super().__init__(message)
        self.exit_program()


class ImportErrorHandler(BaseError):
    """Raised when an import statement has trouble loading a module"""

    def __init__(self, message=None):
        """
        Args:
            message (str, optional): Name of the missing module or import failure detail.
        """
        message = f"ImportError: {message}" if message else "Import error occurred."
        super().__init__(message)
        self.exit_program()


class CustomErrorHandler(BaseError):
    """Raised when a custom error is encountered"""

    def __init__(self, message=None, status_code=None):
        """
        Args:
            message (str, optional): Human-readable description of the error.
            status_code (int, optional): HTTP or application-level status code associated with the error.
        """
        self.status_code = status_code
        if message and status_code:
            full_message = (
                f"Error occurred. Status code: {status_code}, Message: {message}"
            )
        elif message:
            full_message = f"Error occurred. Message: {message}"
        elif status_code:
            full_message = f"Error occurred. Status code: {status_code}"
        else:
            full_message = "Error occurred."
        super().__init__(full_message)
