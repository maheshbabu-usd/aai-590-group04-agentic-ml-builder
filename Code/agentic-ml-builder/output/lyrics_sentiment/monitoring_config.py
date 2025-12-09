"""Model monitoring and observability"""
import logging
from datetime import datetime
import json
from pathlib import Path

class ModelMonitor:
    """Monitor model performance and health"""
    
    def __init__(self, log_dir: str = "monitoring_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup monitoring logger"""
        logger = logging.getLogger("ModelMonitor")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(
            self.log_dir / f"monitor_{datetime.now():%Y%m%d}.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_prediction(self, input_data, prediction, confidence=None):
        """Log prediction for monitoring"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction": str(prediction),
            "confidence": confidence,
            "input_shape": str(input_data.shape) if hasattr(input_data, 'shape') else None
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_performance(self, metrics: dict):
        """Log performance metrics"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        self.logger.info(f"Performance: {json.dumps(log_entry)}")
    
    def log_error(self, error: Exception, context: dict = None):
        """Log error with context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context or {}
        }
        self.logger.error(json.dumps(log_entry))
