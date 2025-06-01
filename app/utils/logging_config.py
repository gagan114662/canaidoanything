import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("celery").setLevel(logging.INFO)
    
    # Reduce noise from some libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class TaskLogger:
    """
    Context manager for task-specific logging
    """
    
    def __init__(self, task_id: str, task_name: str):
        self.task_id = task_id
        self.task_name = task_name
        self.logger = get_logger(f"task.{task_name}")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting task {self.task_name} (ID: {self.task_id})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed task {self.task_name} (ID: {self.task_id}) in {duration}")
        else:
            self.logger.error(f"Failed task {self.task_name} (ID: {self.task_id}) after {duration}: {exc_val}")
    
    def info(self, message: str):
        self.logger.info(f"[{self.task_id}] {message}")
    
    def warning(self, message: str):
        self.logger.warning(f"[{self.task_id}] {message}")
    
    def error(self, message: str):
        self.logger.error(f"[{self.task_id}] {message}")
    
    def debug(self, message: str):
        self.logger.debug(f"[{self.task_id}] {message}")

# Example usage:
# logger = setup_logging("DEBUG", "logs/app.log")
# 
# with TaskLogger("task_123", "image_processing") as task_logger:
#     task_logger.info("Processing started")
#     # ... do work ...
#     task_logger.info("Processing completed")