"""
logger.py - Contains the custom logger for the project

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 21-02-2025
"""

# Standard library imports
import os
import sys
import logging

# Define custom log level
VERBOSE = 15  # Between DEBUG (10) and INFO (20)
logging.addLevelName(VERBOSE, "VERBOSE")


class LoggerTextColours:
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    UNDERLINE = "\033[4m"
    PURPLE = "\033[95m"


class CustomFormatter(logging.Formatter):
    DEBUG_INFO_FORMAT = "[%(name)s] %(message)s"
    DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt = "%d-%m-%Y %H:%M:%S"

    FORMATS = {
        logging.DEBUG: LoggerTextColours.OKCYAN
        + DEBUG_INFO_FORMAT
        + LoggerTextColours.ENDC,
        VERBOSE: LoggerTextColours.PURPLE + DEBUG_INFO_FORMAT + LoggerTextColours.ENDC,
        logging.INFO: LoggerTextColours.OKGREEN
        + DEBUG_INFO_FORMAT
        + LoggerTextColours.ENDC,
        logging.WARNING: LoggerTextColours.WARNING
        + DETAILED_FORMAT
        + LoggerTextColours.ENDC,
        logging.ERROR: LoggerTextColours.FAIL
        + DETAILED_FORMAT
        + LoggerTextColours.ENDC,
        logging.CRITICAL: LoggerTextColours.UNDERLINE
        + LoggerTextColours.FAIL
        + DETAILED_FORMAT
        + LoggerTextColours.ENDC,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        if hasattr(record, "created"):
            record.asctime = self.formatTime(record, self.datefmt)
        return formatter.format(record)


# Add method to Logger class for verbose level
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)


# Add verbose method to Logger class
logging.Logger.verbose = verbose


def setup_logger(log_filename, module_name=None, log_level=logging.DEBUG):
    """
    Set up and return a logger with both file and console handlers.

    Args:
        log_filename (str): Name of the log file (e.g., 'article_processor.log')
        module_name (str): Name of the module for which the logger is being set up (default: None)
        log_level (int): Logging level (default: logging.DEBUG [All logs are captured])

    Returns:
        logging.Logger: Configured logger instance
    """
    if module_name:
        logger_name = f"comproscanner.{module_name}"
    else:
        logger_name = "comproscanner"

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Check if handlers are already configured for this logger
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in logger.handlers
    )

    if not has_file_handler:
        file_handler = logging.FileHandler(log_filename)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)

    # Prevent propagation to avoid duplicate messages from parent loggers
    logger.propagate = False

    return logger
