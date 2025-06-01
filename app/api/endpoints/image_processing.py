from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from typing import Optional, List
import uuid
import os
from pathlib import Path
import shutil

from app.core.config import settings
from app.services.celery_app import process_garment_image
from app.utils.file_handler import save_upload_file, validate_file
from app.models.schemas import ImageProcessingRequest, ImageProcessingResponse, TaskStatus

router = APIRouter()

@router.post("/process-garment", response_model=ImageProcessingResponse)
async def process_garment_image_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    style_prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(""),
    enhance_quality: bool = Form(True),
    remove_background: bool = Form(False),
    upscale: bool = Form(True)
):
    """
    Process a garment image to create professional creative imagery
    """
    # Validate file
    if not validate_file(file):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        file_path = await save_upload_file(file, task_id)
        
        # Create processing request
        request_data = ImageProcessingRequest(
            task_id=task_id,
            input_file_path=file_path,
            style_prompt=style_prompt,
            negative_prompt=negative_prompt,
            enhance_quality=enhance_quality,
            remove_background=remove_background,
            upscale=upscale
        )
        
        # Queue the task for background processing
        task = process_garment_image.delay(request_data.dict())
        
        return ImageProcessingResponse(
            task_id=task_id,
            status="queued",
            message="Image processing task queued successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Get the status of a processing task
    """
    from app.services.celery_app import celery_app
    
    try:
        task = celery_app.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            return TaskStatus(
                task_id=task_id,
                status="pending",
                progress=0,
                message="Task is waiting to be processed"
            )
        elif task.state == 'PROGRESS':
            return TaskStatus(
                task_id=task_id,
                status="processing",
                progress=task.info.get('progress', 0),
                message=task.info.get('message', 'Processing image...')
            )
        elif task.state == 'SUCCESS':
            result = task.info
            return TaskStatus(
                task_id=task_id,
                status="completed",
                progress=100,
                message="Image processing completed successfully",
                result_url=result.get('output_url'),
                metadata=result.get('metadata')
            )
        else:
            return TaskStatus(
                task_id=task_id,
                status="failed",
                progress=0,
                message=str(task.info),
                error=str(task.info)
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.get("/download/{task_id}")
async def download_result(task_id: str):
    """
    Download the processed image result
    """
    output_path = Path(settings.OUTPUT_DIR) / f"{task_id}_processed.jpg"
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    return FileResponse(
        path=output_path,
        media_type="image/jpeg",
        filename=f"processed_{task_id}.jpg"
    )

@router.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a processing task
    """
    from app.services.celery_app import celery_app
    
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return {"message": f"Task {task_id} cancelled successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

@router.get("/tasks")
async def list_active_tasks():
    """
    List all active tasks
    """
    from app.services.celery_app import celery_app
    
    try:
        active_tasks = celery_app.control.inspect().active()
        return {"active_tasks": active_tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active tasks: {str(e)}")