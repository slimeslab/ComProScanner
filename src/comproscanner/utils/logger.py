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
    DEBUG_INFO_FORMAT = "%(message)s"
    DETAILED_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
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


def setup_logger(log_filename, log_level=logging.DEBUG):
    """
    Set up and return a logger with both file and console handlers.

    Args:
        log_filename (str): Name of the log file (e.g., 'article_processor.log')
        log_level (int): Logging level (default: logging.DEBUG [All logs are captured])

    Returns:
        logging.Logger: Configured logger instance
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(log_filename)
    logger.setLevel(log_level)

    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        console_handler = logging.StreamHandler(sys.stdout)

        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(CustomFormatter())

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
