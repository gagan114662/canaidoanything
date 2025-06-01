from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from typing import Optional, List
import uuid
import os
from pathlib import Path
import shutil
import logging # Import logging

from app.core.config import settings
from app.services.celery_app import process_garment_image # This should be the Celery task itself
from app.utils.file_handler import save_upload_file, validate_file
from app.models.schemas import ImageProcessingRequest, ImageProcessingResponse, TaskStatus

router = APIRouter()
logger = logging.getLogger(__name__) # Initialize logger

@router.post("/process-garment", response_model=ImageProcessingResponse)
async def process_garment_image_endpoint(
    # background_tasks: BackgroundTasks, # BackgroundTasks not used if switching to Celery directly for all tasks
    # Assuming process_garment_image is a Celery task, BackgroundTasks might not be needed here.
    # If it was for something else, it should be re-evaluated.
    # For now, let's assume it was for the task that's now Celery based.
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
    logger.info(f"Received request to process garment image: filename='{file.filename}'")
    # Validate file
    if not validate_file(file): # validate_file already raises HTTPException
        # This path might not be reached if validate_file raises, but as a safeguard:
        logger.warning(f"Invalid file format or size for file: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file format or size")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    logger.info(f"Generated task_id: {task_id} for file: {file.filename}")
    
    try:
        # Save uploaded file
        file_path = await save_upload_file(file, task_id)
        logger.info(f"File saved to: {file_path} for task_id: {task_id}")
        
        # Create processing request
        request_data = ImageProcessingRequest(
            task_id=task_id,
            input_file_path=str(file_path), # Ensure path is string
            style_prompt=style_prompt,
            negative_prompt=negative_prompt,
            enhance_quality=enhance_quality,
            remove_background=remove_background,
            upscale=upscale
        )
        
        # Queue the task for background processing
        # Assuming process_garment_image is a Celery task imported from celery_app
        process_garment_image.delay(request_data.dict())
        logger.info(f"Task {task_id} queued for processing. Request data: {request_data.dict(exclude={'input_file_path'})}") # Avoid logging full path if too verbose
        
        return ImageProcessingResponse(
            task_id=task_id,
            status="queued",
            message="Image processing task queued successfully"
        )
    except IOError as e:
        logger.error(f"File handling error for task_id {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File handling error: {str(e)}")
    except Exception as e: # Catch other potential errors (e.g., Celery connection)
        logger.error(f"Failed to queue image processing task {task_id}: {e}", exc_info=True)
        # Consider if this should be a more specific error to the client,
        # for now, a generic 500 is kept.
        raise HTTPException(status_code=500, detail=f"Failed to queue image processing task: {str(e)}")

@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Get the status of a processing task
    """
    logger.debug(f"Request for task status: {task_id}")
    from app.services.celery_app import celery_app # Import celery_app instance
    
    try:
        task = celery_app.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            logger.debug(f"Task {task_id} is PENDING")
            return TaskStatus(
                task_id=task_id,
                status="pending",
                progress=0,
                message="Task is waiting to be processed"
            )
        elif task.state == 'PROGRESS':
            logger.debug(f"Task {task_id} is in PROGRESS: {task.info}")
            return TaskStatus(
                task_id=task_id,
                status="processing",
                progress=task.info.get('progress', 0) if task.info else 0,
                message=task.info.get('message', 'Processing image...') if task.info else 'Processing image...'
            )
        elif task.state == 'SUCCESS':
            result = task.info if task.info else {}
            logger.info(f"Task {task_id} SUCCEEDED. Result info: {result}")
            return TaskStatus(
                task_id=task_id,
                status="completed",
                progress=100,
                message="Image processing completed successfully",
                result_url=result.get('output_url'), # Ensure these keys exist or handle gracefully
                metadata=result.get('metadata')
            )
        elif task.state == 'FAILURE': # Explicitly handle FAILURE
            logger.warning(f"Task {task_id} FAILED. Info: {task.info}")
            return TaskStatus(
                task_id=task_id,
                status="failed",
                progress=0,
                message=f"Task failed: {str(task.info)}", # Provide more context
                error=str(task.info)
            )
        else: # Other states like REVOKED, RETRY
            logger.info(f"Task {task_id} is in state: {task.state}. Info: {task.info}")
            return TaskStatus(
                task_id=task_id,
                status=task.state.lower(),
                progress=0, # Or current progress if available
                message=f"Task is in state: {task.state}",
                error=str(task.info) if task.info else None
            )
            
    except Exception as e: # Catch issues like Celery backend being unavailable
        logger.error(f"Failed to get task status for {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.get("/download/{task_id}")
async def download_result(task_id: str):
    """
    Download the processed image result
    """
    logger.info(f"Request to download result for task_id: {task_id}")
    output_path = Path(settings.OUTPUT_DIR) / f"{task_id}_processed.jpg" # Define clearly
    
    if not output_path.exists():
        logger.warning(f"Processed image not found for task_id {task_id} at path {output_path}")
        raise HTTPException(status_code=404, detail="Processed image not found.")
    
    logger.info(f"Serving file {output_path} for task_id {task_id}")
    return FileResponse(
        path=str(output_path), # Ensure path is string
        media_type="image/jpeg",
        filename=f"processed_{task_id}.jpg"
    )

@router.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a processing task
    """
    logger.info(f"Request to cancel task_id: {task_id}")
    from app.services.celery_app import celery_app # Import celery_app instance
    
    try:
        celery_app.control.revoke(task_id, terminate=True)
        logger.info(f"Task {task_id} cancellation requested successfully.")
        return {"message": f"Task {task_id} cancellation requested successfully"}
    except Exception as e: # Catch issues like Celery backend being unavailable
        logger.error(f"Failed to cancel task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

@router.get("/tasks")
async def list_active_tasks():
    """
    List all active tasks
    """
    logger.info("Request to list active tasks")
    from app.services.celery_app import celery_app # Import celery_app instance
    
    try:
        # Ensure inspector is available
        inspector = celery_app.control.inspect(timeout=1) # Add timeout
        active_tasks = inspector.active()

        if active_tasks is None: # Check if inspect() returned None (e.g. no workers available)
            logger.warning("Could not inspect active tasks. No workers might be available.")
            raise HTTPException(status_code=503, detail="Could not retrieve active tasks. Workers may be unavailable.")

        logger.info(f"Active tasks retrieved: {active_tasks}")
        return {"active_tasks": active_tasks}
    except Exception as e: # Catch issues like Celery backend being unavailable or timeout
        logger.error(f"Failed to get active tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get active tasks: {str(e)}")