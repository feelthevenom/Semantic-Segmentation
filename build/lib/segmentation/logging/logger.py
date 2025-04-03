import logging
import os
from datetime import datetime

class LoggerManager:
    _instance = None
    _loggers = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Create logs directory
            self.LOG_DIR = "logs"
            os.makedirs(self.LOG_DIR, exist_ok=True)
            
            # Create a base log filename - used for the current session
            self.log_filename = f"segmentation_{datetime.now().strftime('%Y-%m-%d-%H')}.log"
            self.log_path = os.path.join(self.LOG_DIR, self.log_filename)
            
            self._initialized = True
    
    def get_logger(self, name="SemanticSegmentation"):
        """
        Returns a configured logger instance.
        Same logger name will return the same logger to prevent duplicates.
        """
        # If logger with this name exists, return it
        if name in self._loggers:
            return self._loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to prevent duplicates
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s:%(lineno)d - %(levelname)s - %(message)s"
        )
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        # Store logger in our dictionary
        self._loggers[name] = logger
        
        return logger


# Create singleton instance
logger_manager = LoggerManager()

# Convenience function that matches the original API
def get_logger(name="SemanticSegmentation"):
    return logger_manager.get_logger(name)