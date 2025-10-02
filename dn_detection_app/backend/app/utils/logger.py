import logging
import sys
from datetime import datetime
import os

def setup_logger(name: str = "dn_detection", level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with consistent formatting
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    log_dir = os.getenv("LOG_DIR", "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"dn_detection_{datetime.now().strftime('%Y%m%d')}.log")
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def log_request(request_data: dict, response_data: dict, execution_time: float):
    """
    Log API request and response
    """
    logger = setup_logger()
    logger.info(f"API Request processed in {execution_time:.3f}s")
    logger.debug(f"Request: {request_data}")
    logger.debug(f"Response: {response_data}")

def log_prediction(patient_id: str, risk_score: float, risk_level: str, confidence: float):
    """
    Log prediction results
    """
    logger = setup_logger()
    logger.info(
        f"Prediction - Patient: {patient_id}, Risk: {risk_level} ({risk_score:.1f}%), "
        f"Confidence: {confidence:.3f}"
    )

def log_error(error: Exception, context: str = ""):
    """
    Log error with context
    """
    logger = setup_logger()
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)