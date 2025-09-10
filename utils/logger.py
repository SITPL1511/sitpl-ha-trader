"""
Logging configuration and utilities
"""

import logging
import os
from datetime import datetime
from typing import Optional


class TradingLogger:
    """Enhanced logging setup for trading engine"""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = logs_dir
        self.ensure_logs_directory()

    def ensure_logs_directory(self):
        """Create logs directory if it doesn't exist"""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

    def setup_logging(self, symbol: Optional[str] = None,
                      timeframe: Optional[str] = None,
                      log_level: int = logging.INFO) -> str:
        """Setup comprehensive logging with symbol and timeframe in filename"""

        # Create filename with symbol and timeframe
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if symbol and timeframe:
            # Clean symbol name for filename (remove special characters)
            clean_symbol = symbol.replace('.', '_').replace(':', '_').replace('/', '_')
            log_filename = f'trading_engine_{clean_symbol}_{timeframe}_{timestamp}.log'
        else:
            log_filename = f'trading_engine_{timestamp}.log'

        log_filepath = os.path.join(self.logs_dir, log_filename)

        # Configure logging only if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_filepath),
                    logging.StreamHandler()
                ]
            )

        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized: {log_filepath}")

        return log_filepath

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance"""
        return logging.getLogger(name)

    def set_log_level(self, level: int):
        """Set logging level for all handlers"""
        logger = logging.getLogger()
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def log_error(self, message: str, exc_info: bool = False):
        """Log error message"""
        self.logger.error(message, exc_info=exc_info)

    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)


# Global logger instance
trading_logger = TradingLogger()