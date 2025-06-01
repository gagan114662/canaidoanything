from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query # Removed BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List, Dict, Any
import uuid
import os
import json
from pathlib import Path
import time # Removed asyncio
import logging # Import logging

from app.core.config import settings
# ModelTransformationPipeline is no longer directly used here for task processing
# from app.services.tasks.model_transformation import ModelTransformationPipeline
from app.services.tasks.model_transformation import transform_model_task # Import the Celery task
from app.utils.file_handler import save_upload_file, validate_file, get_file_info
from app.models.schemas import (
    ModelTransformationRequest, 
    ModelTransformationResponse, 
    TransformationStatus,
    BrandGuidelinesRequest
)

router = APIRouter()
logger = logging.getLogger(__name__) # Initialize logger

# Global pipeline instance for startup loading - This can remain if other non-task operations need it
# Or be removed if `get_transformation_pipeline()` in tasks.py is the sole manager.
# For now, let's assume startup loading is beneficial for the app, not just Celery workers.
# If Celery workers are on different machines or scale differently, this startup event
# only affects the API server instances.
# The Celery task itself uses get_transformation_pipeline().
_startup_pipeline_instance = None

@router.on_event("startup")
async def startup_event():
    """Load models on startup for the API server instance if needed."""
    global _startup_pipeline_instance
    logger.info("Attempting to load transformation pipeline for API server startup...")
    try:
        # This is for the API server itself, Celery workers handle their own loading.
        from app.services.tasks.model_transformation import ModelTransformationPipeline as PipelineForStartup
        _startup_pipeline_instance = PipelineForStartup()
        _startup_pipeline_instance.load_all_models()
        logger.info("Model transformation pipeline loaded successfully for API server.")
    except Exception as e:
        logger.error(f"Failed to load transformation pipeline on API server startup: {e}", exc_info=True)
        # Depending on requirements, you might want to prevent startup or log more severely.

@router.post("/transform-model", response_model=ModelTransformationResponse)
async def transform_model_endpoint(
    # background_tasks: BackgroundTasks, # Removed
    file: UploadFile = File(...),
    style_prompt: str = Form(..., description="Style description for the transformation"),
    negative_prompt: Optional[str] = Form("", description="What to avoid in the transformation"),
    num_variations: int = Form(5, ge=1, le=5, description="Number of style variations (1-5)"),
    enhance_model: bool = Form(True, description="Whether to enhance model appearance"),
    optimize_garment: bool = Form(True, description="Whether to optimize garment presentation"),
    generate_scene: bool = Form(True, description="Whether to generate professional scenes"),
    quality_mode: str = Form("balanced", description="Processing quality mode: fast, balanced, high"),
    brand_name: Optional[str] = Form(None, description="Brand name for consistency (optional)"),
    seed: Optional[int] = Form(None, description="Random seed for reproducibility")
):
    """
    Transform model photo into professional photoshoot variations
    
    This endpoint processes a model photo and generates multiple professional
    style variations suitable for fashion campaigns, e-commerce, and marketing.
    """
    logger.info(f"Received request to transform model: filename='{file.filename}', style_prompt='{style_prompt}'")
    # Validate file
    if not validate_file(file): # validate_file already raises HTTPException
        logger.warning(f"Invalid file format or size for file: {file.filename}, transformation_id: {transformation_id}")
        raise HTTPException(status_code=400, detail="Invalid file format or size")
    
    # Generate unique transformation ID
    transformation_id = str(uuid.uuid4())
    logger.info(f"Generated transformation_id: {transformation_id} for file: {file.filename}")
    
    try:
        # Save uploaded file
        # Ensure this path is accessible by Celery workers if they are on different filesystems.
        # Using a shared volume or object storage is common for distributed Celery setups.
        file_path = await save_upload_file(file, transformation_id)
        
        # Input image is no longer loaded directly in the endpoint for the task.
        # The Celery task will load it from file_path.
        
        # Create processing request
        request_data = ModelTransformationRequest(
            transformation_id=transformation_id,
            input_file_path=str(file_path), # Ensure file_path is a string for dict conversion
            style_prompt=style_prompt,
            negative_prompt=negative_prompt,
            num_variations=num_variations,
            enhance_model=enhance_model,
            optimize_garment=optimize_garment,
            generate_scene=generate_scene,
            quality_mode=quality_mode,
            brand_name=brand_name,
            seed=seed
        )
        
        # Dispatch Celery task
        # Pass request_data as a dictionary for better serialization
        transform_model_task.delay(transformation_id, request_data.dict())
        logger.info(f"Transformation task {transformation_id} queued successfully.")
        
        return ModelTransformationResponse(
            transformation_id=transformation_id, # Client uses this to poll for status/results
            status="queued", # Initial status
            message="Model transformation has been queued successfully.",
            estimated_time=_estimate_processing_time(num_variations, quality_mode) # Keep estimation helper
        )
    except IOError as e:
        logger.error(f"File handling error for transformation_id {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File handling error: {str(e)}")
    except Exception as e: # Catch other potential errors (e.g., Celery connection)
        logger.error(f"Failed to queue transformation task {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to queue transformation task: {str(e)}")

@router.get("/transform-status/{transformation_id}", response_model=TransformationStatus)
async def get_transformation_status(transformation_id: str):
    """
    Get the status of a model transformation task.
    This relies on the status file written by the Celery task.
    """
    logger.debug(f"Request for transformation status: {transformation_id}")
    status_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_status.json"

    try:
        if not status_file.exists():
            logger.warning(f"Status file not found for transformation_id {transformation_id} at {status_file}")
            raise HTTPException(status_code=404, detail="Transformation not found or not yet started.")
        
        # Load status
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        logger.debug(f"Status for {transformation_id} retrieved successfully: {status_data.get('status')}")
        return TransformationStatus(**status_data)
        
    except HTTPException: # Re-raise HTTP exceptions directly
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse status file {status_file} for transformation_id {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to parse transformation status; file might be corrupted.")
    except Exception as e:
        logger.error(f"Error reading status for transformation_id {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get transformation status: {str(e)}")

@router.get("/transform-result/{transformation_id}")
async def get_transformation_result(transformation_id: str):
    """
    Get the complete transformation result with all variations
    """
    logger.debug(f"Request for transformation result: {transformation_id}")
    result_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_result.json"
    try:
        if not result_file.exists():
            logger.warning(f"Result file not found for transformation_id {transformation_id} at {result_file}")
            raise HTTPException(status_code=404, detail="Transformation result not found")
        
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        logger.debug(f"Result for {transformation_id} retrieved successfully.")
        return JSONResponse(content=result_data)
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse result file {result_file} for transformation_id {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to parse transformation result; file might be corrupted.")
    except Exception as e:
        logger.error(f"Error reading result for transformation_id {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get transformation result: {str(e)}")

@router.get("/download-variation/{transformation_id}/{variation_index}")
async def download_variation(transformation_id: str, variation_index: int):
    """
    Download a specific variation from the transformation result
    """
    logger.info(f"Request to download variation {variation_index} for transformation_id: {transformation_id}")
    variation_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_variation_{variation_index}.jpg"
    try:
        if not variation_file.exists():
            logger.warning(f"Variation {variation_index} not found for transformation_id {transformation_id} at {variation_file}")
            raise HTTPException(status_code=404, detail="Variation image not found")
        
        logger.info(f"Serving variation file {variation_file} for transformation_id {transformation_id}")
        return FileResponse(
            path=str(variation_file), # Ensure path is string
            media_type="image/jpeg",
            filename=f"transformed_model_{transformation_id}_variation_{variation_index}.jpg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download variation {variation_index} for {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download variation: {str(e)}")

@router.get("/download-all/{transformation_id}")
async def download_all_variations(transformation_id: str):
    """
    Download all variations as a ZIP file
    """
    logger.info(f"Request to download all variations for transformation_id: {transformation_id}")
    try:
        import zipfile
        import io
        
        output_dir = Path(settings.OUTPUT_DIR)
        variation_files = list(output_dir.glob(f"{transformation_id}_variation_*.jpg"))
        
        if not variation_files:
            logger.warning(f"No variations found for transformation_id {transformation_id} to create ZIP.")
            raise HTTPException(status_code=404, detail="No variations found for this transformation ID.")
            
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for variation_file in variation_files:
                zip_file.write(variation_file, variation_file.name)
            
            result_file = output_dir / f"{transformation_id}_result.json"
            if result_file.exists():
                zip_file.write(result_file, "transformation_metadata.json")
        
        zip_buffer.seek(0)
        logger.info(f"Successfully created ZIP package for transformation_id {transformation_id}")
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()), # Read bytes into a new BytesIO for StreamingResponse
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=transformation_{transformation_id}.zip"}
        )
        
    except HTTPException: # Re-raise HTTP exceptions directly
        raise
    except zipfile.BadZipFile as e:
        logger.error(f"Failed to create ZIP (BadZipFile) for transformation_id {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create download package due to ZIP error.")
    except IOError as e:
        logger.error(f"Failed to create ZIP (IOError) for transformation_id {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create download package due to file error.")
    except Exception as e:
        logger.error(f"Failed to create download package for transformation_id {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create download package: {str(e)}")

@router.delete("/transform/{transformation_id}")
async def cancel_transformation(transformation_id: str):
    """
    Cancel a running transformation task
    """
    # Note: This directly manipulates status files. If Celery manages task state,
    # this might need to interact with Celery's cancellation/revocation mechanisms
    # or be re-evaluated. For now, just adding logging.
    logger.info(f"Request to cancel transformation_id: {transformation_id}")
    status_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_status.json"

    try:
        if status_file.exists():
            with open(status_file, 'r+') as f: # Open for reading and writing
                try:
                    status_data = json.load(f)
                    status_data["status"] = "cancelled"
                    status_data["message"] = "Transformation cancelled by user"
                    f.seek(0) # Go to the beginning of the file
                    json.dump(status_data, f)
                    f.truncate() # Remove remaining part of old file if new data is shorter
                    logger.info(f"Updated status file for transformation_id {transformation_id} to cancelled.")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse status file {status_file} during cancellation for {transformation_id}: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail="Failed to update status file due to parsing error.")
            return {"message": f"Transformation {transformation_id} cancellation processed."}
        else:
            logger.warning(f"Status file not found for cancellation of transformation_id {transformation_id}.")
            # Depending on desired behavior, could return 404 or success if task never existed.
            # For now, let's assume it's okay if the file isn't there (task might be pre-start or already cleaned up)
            return {"message": f"Transformation {transformation_id} status file not found, assumed cancelled or completed."}
            
    except IOError as e:
        logger.error(f"File error during cancellation of transformation_id {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel transformation due to file error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to cancel transformation {transformation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel transformation: {str(e)}")

@router.get("/transformations")
async def list_transformations(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List transformation tasks with optional filtering
    """
    logger.info(f"Request to list transformations: status='{status}', limit={limit}, offset={offset}")
    try:
        output_dir = Path(settings.OUTPUT_DIR)
        # Ensure output_dir exists to prevent errors during glob
        if not output_dir.is_dir():
            logger.warning(f"Output directory {output_dir} not found for listing transformations.")
            return {"transformations": [], "total": 0, "limit": limit, "offset": offset}

        status_files = list(output_dir.glob("*_status.json"))
        
        transformations = []
        for status_file in status_files:
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                
                if status and status_data.get("status") != status:
                    continue
                transformations.append(status_data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse status file {status_file} during list: {e}", exc_info=True)
                continue # Skip corrupted status file
            except IOError as e:
                logger.error(f"IOError reading status file {status_file} during list: {e}", exc_info=True)
                continue # Skip unreadable status file
            except Exception as e: # Catch any other unexpected error for a specific file
                logger.error(f"Unexpected error reading status file {status_file} during list: {e}", exc_info=True)
                continue
        
        # Sort by creation time (newest first) - ensure 'created_at' exists or provide default
        transformations.sort(key=lambda x: x.get("created_at", 0.0), reverse=True)
        
        total = len(transformations)
        paginated_results = transformations[offset:offset + limit]
        
        logger.info(f"Found {total} transformations, returning {len(paginated_results)}.")
        return {
            "transformations": paginated_results,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to list transformations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list transformations: {str(e)}")

@router.post("/brand-guidelines/{brand_name}")
async def upload_brand_guidelines(
    brand_name: str,
    guidelines: BrandGuidelinesRequest
):
    """
    Upload brand guidelines for consistent transformations
    """
    logger.info(f"Request to upload brand guidelines for: {brand_name}")
    try:
        from app.services.ai.brand_consistency_service import BrandConsistencyService # Local import if not globally needed
        
        brand_service = BrandConsistencyService()
        # Assuming load_brand_guidelines might raise its own specific errors or return False
        success = brand_service.load_brand_guidelines(brand_name, guidelines.dict())
        
        if not success:
            logger.warning(f"BrandConsistencyService.load_brand_guidelines failed for brand: {brand_name}")
            raise HTTPException(status_code=400, detail="Failed to load brand guidelines via service.")
        
        guidelines_dir = Path("brand_guidelines") # Consider making this path configurable via settings
        guidelines_dir.mkdir(parents=True, exist_ok=True) # Ensure parent dirs are created
        
        guidelines_file = guidelines_dir / f"{brand_name}.json"
        with open(guidelines_file, 'w') as f:
            json.dump(guidelines.dict(), f, indent=2)
        
        logger.info(f"Brand guidelines for '{brand_name}' uploaded and saved to {guidelines_file}")
        return {
            "message": f"Brand guidelines uploaded successfully for '{brand_name}'",
            "brand_name": brand_name,
            "guidelines_loaded": True
        }
        
    except HTTPException: # Re-raise HTTP exceptions
        raise
    except IOError as e:
        logger.error(f"File error saving brand guidelines for {brand_name} to {guidelines_file}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save brand guidelines due to file error.")
    except Exception as e: # Catch other errors from BrandConsistencyService or json.dump
        logger.error(f"Failed to upload brand guidelines for {brand_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload brand guidelines: {str(e)}")

@router.get("/brand-guidelines/{brand_name}")
async def get_brand_guidelines(brand_name: str):
    """
    Get brand guidelines for a specific brand
    """
    logger.info(f"Request to get brand guidelines for: {brand_name}")
    guidelines_file = Path("brand_guidelines") / f"{brand_name}.json"
    try:
        if not guidelines_file.exists():
            logger.warning(f"Brand guidelines file not found for {brand_name} at {guidelines_file}")
            raise HTTPException(status_code=404, detail="Brand guidelines not found.")
        
        with open(guidelines_file, 'r') as f:
            guidelines = json.load(f)
        
        logger.info(f"Successfully retrieved brand guidelines for {brand_name}")
        return {
            "brand_name": brand_name,
            "guidelines": guidelines
        }
        
    except HTTPException: # Re-raise HTTP exceptions
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse brand guidelines file {guidelines_file} for {brand_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to parse brand guidelines file.")
    except IOError as e:
        logger.error(f"File error reading brand guidelines for {brand_name} from {guidelines_file}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read brand guidelines due to file error.")
    except Exception as e:
        logger.error(f"Failed to get brand guidelines for {brand_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get brand guidelines: {str(e)}")

@router.get("/brand-consistency-report/{transformation_id}")
async def get_brand_consistency_report(transformation_id: str, brand_name: str):
    """
    Generate brand consistency report for a transformation
    """
    logger.info(f"Request for brand consistency report: transformation_id={transformation_id}, brand_name={brand_name}")
    try:
        variation_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_variation_0.jpg"
        if not variation_file.exists():
            logger.warning(f"Variation 0 not found for transformation_id {transformation_id} to generate brand report.")
            raise HTTPException(status_code=404, detail="Transformation variation 0 not found, cannot generate report.")
        
        from app.utils.image_utils import load_image # Local import
        from app.services.ai.brand_consistency_service import BrandConsistencyService # Local import
        
        image = load_image(str(variation_file))
        if image is None: # load_image should ideally raise an error or return a placeholder
            logger.error(f"Failed to load image {variation_file} for brand report of transformation {transformation_id}")
            raise HTTPException(status_code=500, detail="Failed to load image for brand report.")

        brand_service = BrandConsistencyService()
        
        # Load brand guidelines if not already loaded by the service instance (idempotent check)
        # This logic might be better inside the service or handled by a global brand guideline cache
        guidelines_file = Path("brand_guidelines") / f"{brand_name}.json"
        if guidelines_file.exists():
            try:
                with open(guidelines_file, 'r') as f:
                    guidelines = json.load(f)
                brand_service.load_brand_guidelines(brand_name, guidelines) # Assuming this is safe to call multiple times
            except Exception as e_load: # Catch errors during guideline loading
                logger.warning(f"Could not load brand guidelines {brand_name} during report generation: {e_load}", exc_info=True)
                # Proceed without guidelines if they can't be loaded, or error out
        
        report = brand_service.generate_brand_report(image, brand_name)
        logger.info(f"Brand consistency report generated for transformation_id {transformation_id}, brand {brand_name}")
        return report
        
    except HTTPException: # Re-raise HTTP exceptions
        raise
    except Exception as e: # Catch errors from image loading, service methods etc.
        logger.error(f"Failed to generate brand report for {transformation_id}, brand {brand_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate brand report: {str(e)}")

@router.get("/pipeline-stats")
async def get_pipeline_stats():
    """
    Get transformation pipeline statistics
    """
    logger.info("Request for pipeline stats")
    try:
        if _startup_pipeline_instance and hasattr(_startup_pipeline_instance, 'get_pipeline_stats'):
            stats = _startup_pipeline_instance.get_pipeline_stats()
            logger.info("Pipeline stats retrieved successfully.")
            return stats
        else:
            logger.warning("Pipeline stats unavailable: _startup_pipeline_instance not available or has no get_pipeline_stats method.")
            return {"message": "Pipeline stats unavailable (pipeline not loaded or configured correctly on API server)."}
        
    except Exception as e:
        logger.error(f"Failed to get pipeline stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline stats: {str(e)}")

# Background task functions (process_transformation_task and _update_transformation_status)
# are now removed as their logic is encapsulated within the Celery task
# in app.services.tasks.model_transformation.py

def _estimate_processing_time(num_variations: int, quality_mode: str) -> int:
    """Estimate processing time based on parameters (remains unchanged)"""
    base_time = 15  # Base time in seconds
    
    # Adjust for variations
    time_per_variation = 4
    variation_time = num_variations * time_per_variation
    
    # Adjust for quality mode
    quality_multiplier = {
        "fast": 0.7,
        "balanced": 1.0,
        "high": 1.5
    }.get(quality_mode, 1.0)
    
    total_time = int((base_time + variation_time) * quality_multiplier)
    
    return total_time