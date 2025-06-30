"""
Logging configuration for the API
"""
import logging
import sys
from typing import List
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).isoformat()

        json_record = {
            "timestamp": timestamp,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if record.exc_info:
            json_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(json_record)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with JSON formatting"""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

    # File handler for persistent logs
    file_handler = logging.FileHandler("api.log")
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Suppress some noisy logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    root_logger.info("Logging configured successfully")
