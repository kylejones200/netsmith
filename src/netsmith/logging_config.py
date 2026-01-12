"""
Logging configuration for NetSmith.

Provides structured logging with JSON format support for production environments.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_style: str = "simple",
    json_format: bool = False
) -> logging.Logger:
    """
    Configure logging for NetSmith.
    
    Parameters
    ----------
    level : int, default logging.INFO
        Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format_style : str, default "simple"
        Format style: "simple" or "detailed"
    json_format : bool, default False
        Use JSON format (useful for production/log aggregation)
    
    Returns
    -------
    logger : logging.Logger
        Configured logger
    """
    logger = logging.getLogger("netsmith")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    
    # Set format
    if json_format:
        try:
            import json
            import time
            
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_data = {
                        "timestamp": time.time(),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    }
                    if hasattr(record, "module"):
                        log_data["module"] = record.module
                    if hasattr(record, "funcName"):
                        log_data["function"] = record.funcName
                    if record.exc_info:
                        log_data["exception"] = self.formatException(record.exc_info)
                    return json.dumps(log_data)
            
            formatter = JSONFormatter()
        except ImportError:
            # Fallback to simple format if JSON not available
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
    else:
        if format_style == "detailed":
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
            )
        else:  # simple
            formatter = logging.Formatter(
                "%(levelname)s: %(message)s"
            )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Parameters
    ----------
    name : str, optional
        Logger name (defaults to "netsmith")
    
    Returns
    -------
    logger : logging.Logger
        Logger instance
    """
    if name is None:
        name = "netsmith"
    return logging.getLogger(name)

