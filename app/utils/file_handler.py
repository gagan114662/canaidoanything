import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException
from PIL import Image
import aiofiles
from app.core.config import settings

async def save_upload_file(upload_file: UploadFile, task_id: str) -> str:
    """
    Save uploaded file to upload directory
    
    Args:
        upload_file: FastAPI UploadFile object
        task_id: Unique task identifier
        
    Returns:
        Path to saved file
    """
    try:
        # Ensure upload directory exists
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_extension = upload_file.filename.split('.')[-1].lower()
        filename = f"{task_id}.{file_extension}"
        file_path = upload_dir / filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as buffer:
            content = await upload_file.read()
            await buffer.write(content)
        
        return str(file_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

def validate_file(upload_file: UploadFile) -> bool:
    """
    Validate uploaded file
    
    Args:
        upload_file: FastAPI UploadFile object
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        # Check file size
        if upload_file.size > settings.MAX_FILE_SIZE:
            return False
        
        # Check file extension
        if not upload_file.filename:
            return False
            
        file_extension = upload_file.filename.split('.')[-1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            return False
        
        return True
        
    except Exception:
        return False

def get_file_info(file_path: str) -> dict:
    """
    Get information about a file
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return {"error": "File not found"}
        
        # Get basic file info
        stat = path.stat()
        info = {
            "filename": path.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "extension": path.suffix.lower()
        }
        
        # If it's an image, get additional info
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            try:
                with Image.open(path) as img:
                    info.update({
                        "width": img.width,
                        "height": img.height,
                        "mode": img.mode,
                        "format": img.format
                    })
            except Exception:
                pass
        
        return info
        
    except Exception as e:
        return {"error": str(e)}

def cleanup_files(older_than_hours: int = 24):
    """
    Clean up old files from upload and output directories
    
    Args:
        older_than_hours: Delete files older than this many hours
    """
    import time
    
    try:
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        
        # Clean upload directory
        upload_dir = Path(settings.UPLOAD_DIR)
        if upload_dir.exists():
            for file_path in upload_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        print(f"Deleted old upload file: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
        
        # Clean output directory
        output_dir = Path(settings.OUTPUT_DIR)
        if output_dir.exists():
            for file_path in output_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        print(f"Deleted old output file: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
                        
    except Exception as e:
        print(f"Error during file cleanup: {e}")

def ensure_directories():
    """
    Ensure all required directories exist
    """
    directories = [
        settings.UPLOAD_DIR,
        settings.OUTPUT_DIR,
        "app/static"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def get_file_url(file_path: str, base_url: str = "") -> str:
    """
    Convert file path to URL
    
    Args:
        file_path: Local file path
        base_url: Base URL for the application
        
    Returns:
        Public URL for the file
    """
    try:
        path = Path(file_path)
        
        # Convert to relative path from static directory
        if "static" in str(path):
            relative_path = str(path).split("static/")[-1]
            return f"{base_url}/static/{relative_path}"
        
        return file_path
        
    except Exception:
        return file_path