import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
log_path = os.path.join(LOG_DIR, log_filename)

_logger_dict = {}  # store different named loggers


def get_logger(name: str = "SemanticSegmentation") -> logging.Logger:
    """
    Returns a named logger instance.
    Same logger name will not create duplicate handlers.
    """

    if name in _logger_dict:
        return _logger_dict[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")

        # File Handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    _logger_dict[name] = logger

    return logger
