from typing import Dict, Any, List
import time
import psutil
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque

from app.utils.logger import setup_logger

logger = setup_logger()

class SystemMonitor:
    """
    Monitor system metrics and API performance
    """
    
    def __init__(self):
        self.request_times = deque(maxlen=1000)  # Keep last 1000 request times
        self.request_counts = defaultdict(int)   # Count requests by endpoint
        self.error_counts = defaultdict(int)     # Count errors by type
        self.start_time = datetime.now()
        
    def record_request(self, endpoint: str, duration: float, status_code: int):
        """
        Record API request metrics
        """
        self.request_times.append(duration)
        self.request_counts[endpoint] += 1
        
        if status_code >= 400:
            self.error_counts[f"{status_code}"] += 1
            
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics
        """
        try:
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process-specific metrics
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            return {
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_used_gb": disk.used / (1024**3),
                    "disk_total_gb": disk.total / (1024**3)
                },
                "process": {
                    "memory_rss_mb": process_memory.rss / (1024*1024),
                    "memory_vms_mb": process_memory.vms / (1024*1024),
                    "cpu_percent": process.cpu_percent()
                }
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    def get_api_metrics(self) -> Dict[str, Any]:
        """
        Get API performance metrics
        """
        if not self.request_times:
            return {
                "total_requests": 0,
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
                "requests_per_endpoint": dict(self.request_counts),
                "error_counts": dict(self.error_counts),
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
            }
        
        return {
            "total_requests": len(self.request_times),
            "avg_response_time": sum(self.request_times) / len(self.request_times),
            "min_response_time": min(self.request_times),
            "max_response_time": max(self.request_times),
            "requests_per_endpoint": dict(self.request_counts),
            "error_counts": dict(self.error_counts),
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Get model-specific metrics (placeholder for future implementation)
        """
        return {
            "model_loaded": True,  # This would be checked from the predictor
            "predictions_made": sum(self.request_counts.values()),
            "average_confidence": 0.85,  # This would be calculated from actual predictions
            "model_version": "1.0.0"
        }
    
    def check_health(self) -> Dict[str, Any]:
        """
        Perform health check
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Check system resources
        try:
            system_metrics = self.get_system_metrics()
            cpu_healthy = system_metrics["system"]["cpu_percent"] < 90
            memory_healthy = system_metrics["system"]["memory_percent"] < 90
            disk_healthy = system_metrics["system"]["disk_percent"] < 90
            
            health_status["checks"]["system"] = {
                "cpu": "healthy" if cpu_healthy else "warning",
                "memory": "healthy" if memory_healthy else "warning",
                "disk": "healthy" if disk_healthy else "warning"
            }
            
            if not all([cpu_healthy, memory_healthy, disk_healthy]):
                health_status["status"] = "warning"
                
        except Exception as e:
            health_status["checks"]["system"] = {"error": str(e)}
            health_status["status"] = "unhealthy"
        
        # Check API performance
        api_metrics = self.get_api_metrics()
        avg_response_time = api_metrics.get("avg_response_time", 0)
        error_rate = sum(self.error_counts.values()) / max(sum(self.request_counts.values()), 1)
        
        api_healthy = avg_response_time < 5.0 and error_rate < 0.1  # Less than 5s avg, <10% error rate
        health_status["checks"]["api"] = {
            "response_time": "healthy" if avg_response_time < 5.0 else "warning",
            "error_rate": "healthy" if error_rate < 0.1 else "warning"
        }
        
        if not api_healthy and health_status["status"] == "healthy":
            health_status["status"] = "warning"
        
        return health_status

# Global monitor instance
monitor = SystemMonitor()

def get_monitor() -> SystemMonitor:
    """
    Get the global monitor instance
    """
    return monitor

def log_metrics():
    """
    Log current metrics (can be called periodically)
    """
    try:
        system_metrics = monitor.get_system_metrics()
        api_metrics = monitor.get_api_metrics()
        
        logger.info(f"System - CPU: {system_metrics['system']['cpu_percent']:.1f}%, "
                   f"Memory: {system_metrics['system']['memory_percent']:.1f}%")
        logger.info(f"API - Total requests: {api_metrics['total_requests']}, "
                   f"Avg response time: {api_metrics['avg_response_time']:.3f}s")
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")