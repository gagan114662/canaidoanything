from fastapi import APIRouter, Depends
from datetime import datetime
import psutil
import os
import logging # Import logging

router = APIRouter()
logger = logging.getLogger(__name__) # Initialize logger

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "garment-creative-ai"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "garment-creative-ai",
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                }
            }
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}", exc_info=True) # Add logging
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "garment-creative-ai",
            "error": str(e)
        }