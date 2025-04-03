import os
import logging
import colorlog
from config import APP_LOGS


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler
    directory = os.path.dirname(APP_LOGS)
    os.makedirs(directory, exist_ok=True)

    file_handler = logging.FileHandler(APP_LOGS)
    file_handler.setLevel(logging.INFO)

    # Define log format
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(reset)s%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    # For single color
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()

logger.info("Logger is set up and writing to both console and file.")
