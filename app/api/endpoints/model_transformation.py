from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List, Dict, Any
import uuid
import os
import json
from pathlib import Path
import asyncio
import time

from app.core.config import settings
from app.services.tasks.model_transformation import ModelTransformationPipeline
from app.utils.file_handler import save_upload_file, validate_file, get_file_info
from app.models.schemas import (
    ModelTransformationRequest, 
    ModelTransformationResponse, 
    TransformationStatus,
    BrandGuidelinesRequest
)

router = APIRouter()

# Global pipeline instance
transformation_pipeline = ModelTransformationPipeline()

@router.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        transformation_pipeline.load_all_models()
        print("Model transformation pipeline loaded successfully")
    except Exception as e:
        print(f"Failed to load transformation pipeline: {e}")

@router.post("/transform-model", response_model=ModelTransformationResponse)
async def transform_model_endpoint(
    background_tasks: BackgroundTasks,
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
    # Validate file
    if not validate_file(file):
        raise HTTPException(status_code=400, detail="Invalid file format or size")
    
    # Generate unique transformation ID
    transformation_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        file_path = await save_upload_file(file, transformation_id)
        
        # Load and validate image
        from app.utils.image_utils import load_image
        input_image = load_image(file_path)
        
        # Create processing request
        request_data = ModelTransformationRequest(
            transformation_id=transformation_id,
            input_file_path=file_path,
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
        
        # Queue transformation for background processing
        background_tasks.add_task(
            process_transformation_task,
            transformation_id,
            input_image,
            request_data
        )
        
        return ModelTransformationResponse(
            transformation_id=transformation_id,
            status="queued",
            message="Model transformation queued successfully",
            estimated_time=_estimate_processing_time(num_variations, quality_mode)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process transformation: {str(e)}")

@router.get("/transform-status/{transformation_id}", response_model=TransformationStatus)
async def get_transformation_status(transformation_id: str):
    """
    Get the status of a model transformation task
    """
    try:
        # Check if transformation exists in storage
        status_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_status.json"
        
        if not status_file.exists():
            raise HTTPException(status_code=404, detail="Transformation not found")
        
        # Load status
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        
        return TransformationStatus(**status_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transformation status: {str(e)}")

@router.get("/transform-result/{transformation_id}")
async def get_transformation_result(transformation_id: str):
    """
    Get the complete transformation result with all variations
    """
    try:
        # Load result file
        result_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_result.json"
        
        if not result_file.exists():
            raise HTTPException(status_code=404, detail="Transformation result not found")
        
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        return JSONResponse(content=result_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transformation result: {str(e)}")

@router.get("/download-variation/{transformation_id}/{variation_index}")
async def download_variation(transformation_id: str, variation_index: int):
    """
    Download a specific variation from the transformation result
    """
    try:
        # Construct file path
        variation_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_variation_{variation_index}.jpg"
        
        if not variation_file.exists():
            raise HTTPException(status_code=404, detail="Variation image not found")
        
        return FileResponse(
            path=variation_file,
            media_type="image/jpeg",
            filename=f"transformed_model_{transformation_id}_variation_{variation_index}.jpg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download variation: {str(e)}")

@router.get("/download-all/{transformation_id}")
async def download_all_variations(transformation_id: str):
    """
    Download all variations as a ZIP file
    """
    try:
        import zipfile
        import io
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all variation files
            output_dir = Path(settings.OUTPUT_DIR)
            variation_files = list(output_dir.glob(f"{transformation_id}_variation_*.jpg"))
            
            if not variation_files:
                raise HTTPException(status_code=404, detail="No variations found")
            
            for variation_file in variation_files:
                zip_file.write(variation_file, variation_file.name)
            
            # Add result metadata
            result_file = output_dir / f"{transformation_id}_result.json"
            if result_file.exists():
                zip_file.write(result_file, "transformation_metadata.json")
        
        zip_buffer.seek(0)
        
        # Return ZIP file
        from fastapi.responses import StreamingResponse
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=transformation_{transformation_id}.zip"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create download package: {str(e)}")

@router.delete("/transform/{transformation_id}")
async def cancel_transformation(transformation_id: str):
    """
    Cancel a running transformation task
    """
    try:
        # Update status to cancelled
        status_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_status.json"
        
        if status_file.exists():
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            
            status_data["status"] = "cancelled"
            status_data["message"] = "Transformation cancelled by user"
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f)
        
        return {"message": f"Transformation {transformation_id} cancelled successfully"}
        
    except Exception as e:
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
    try:
        output_dir = Path(settings.OUTPUT_DIR)
        status_files = list(output_dir.glob("*_status.json"))
        
        transformations = []
        
        for status_file in status_files:
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                
                # Filter by status if requested
                if status and status_data.get("status") != status:
                    continue
                
                transformations.append(status_data)
                
            except Exception as e:
                print(f"Failed to read status file {status_file}: {e}")
                continue
        
        # Sort by creation time (newest first)
        transformations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Apply pagination
        total = len(transformations)
        paginated = transformations[offset:offset + limit]
        
        return {
            "transformations": paginated,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list transformations: {str(e)}")

@router.post("/brand-guidelines/{brand_name}")
async def upload_brand_guidelines(
    brand_name: str,
    guidelines: BrandGuidelinesRequest
):
    """
    Upload brand guidelines for consistent transformations
    """
    try:
        # Load brand guidelines into the transformation pipeline
        from app.services.ai.brand_consistency_service import BrandConsistencyService
        
        brand_service = BrandConsistencyService()
        success = brand_service.load_brand_guidelines(brand_name, guidelines.dict())
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to load brand guidelines")
        
        # Save guidelines to file for persistence
        guidelines_dir = Path("brand_guidelines")
        guidelines_dir.mkdir(exist_ok=True)
        
        guidelines_file = guidelines_dir / f"{brand_name}.json"
        with open(guidelines_file, 'w') as f:
            json.dump(guidelines.dict(), f, indent=2)
        
        return {
            "message": f"Brand guidelines uploaded successfully for '{brand_name}'",
            "brand_name": brand_name,
            "guidelines_loaded": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload brand guidelines: {str(e)}")

@router.get("/brand-guidelines/{brand_name}")
async def get_brand_guidelines(brand_name: str):
    """
    Get brand guidelines for a specific brand
    """
    try:
        guidelines_file = Path("brand_guidelines") / f"{brand_name}.json"
        
        if not guidelines_file.exists():
            raise HTTPException(status_code=404, detail="Brand guidelines not found")
        
        with open(guidelines_file, 'r') as f:
            guidelines = json.load(f)
        
        return {
            "brand_name": brand_name,
            "guidelines": guidelines
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get brand guidelines: {str(e)}")

@router.get("/brand-consistency-report/{transformation_id}")
async def get_brand_consistency_report(transformation_id: str, brand_name: str):
    """
    Generate brand consistency report for a transformation
    """
    try:
        # Load the first variation image
        variation_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_variation_0.jpg"
        
        if not variation_file.exists():
            raise HTTPException(status_code=404, detail="Transformation result not found")
        
        # Load image and generate report
        from app.utils.image_utils import load_image
        from app.services.ai.brand_consistency_service import BrandConsistencyService
        
        image = load_image(str(variation_file))
        brand_service = BrandConsistencyService()
        
        # Load brand guidelines if not already loaded
        guidelines_file = Path("brand_guidelines") / f"{brand_name}.json"
        if guidelines_file.exists():
            with open(guidelines_file, 'r') as f:
                guidelines = json.load(f)
            brand_service.load_brand_guidelines(brand_name, guidelines)
        
        # Generate report
        report = brand_service.generate_brand_report(image, brand_name)
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate brand report: {str(e)}")

@router.get("/pipeline-stats")
async def get_pipeline_stats():
    """
    Get transformation pipeline statistics
    """
    try:
        stats = transformation_pipeline.get_pipeline_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline stats: {str(e)}")

# Background task functions
async def process_transformation_task(
    transformation_id: str,
    input_image,
    request_data: ModelTransformationRequest
):
    """Background task for processing model transformation"""
    try:
        # Update status to processing
        await _update_transformation_status(
            transformation_id,
            "processing",
            0,
            "Starting model transformation..."
        )
        
        # Process transformation
        result = transformation_pipeline.transform_model(
            model_image=input_image,
            style_prompt=request_data.style_prompt,
            negative_prompt=request_data.negative_prompt,
            num_variations=request_data.num_variations,
            enhance_model=request_data.enhance_model,
            optimize_garment=request_data.optimize_garment,
            generate_scene=request_data.generate_scene,
            quality_mode=request_data.quality_mode,
            seed=request_data.seed
        )
        
        # Save variation images
        output_dir = Path(settings.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_variations = []
        for i, variation in enumerate(result["variations"]):
            variation_path = output_dir / f"{transformation_id}_variation_{i}.jpg"
            
            # Save variation image
            from app.utils.image_utils import save_image
            save_success = save_image(variation["variation_image"], str(variation_path))
            
            if save_success:
                saved_variations.append({
                    "index": i,
                    "style_type": variation["style_type"],
                    "file_path": str(variation_path),
                    "quality_score": variation["quality_score"],
                    "download_url": f"/api/v1/download-variation/{transformation_id}/{i}"
                })
        
        # Save complete result
        result_data = {
            "transformation_id": transformation_id,
            "status": "completed",
            "variations": saved_variations,
            "metadata": result["metadata"],
            "quality_scores": result["quality_scores"],
            "performance_metrics": result["performance_metrics"],
            "request_data": request_data.dict()
        }
        
        result_file = output_dir / f"{transformation_id}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        # Update final status
        await _update_transformation_status(
            transformation_id,
            "completed",
            100,
            "Transformation completed successfully",
            result_data
        )
        
    except Exception as e:
        print(f"Transformation task failed: {e}")
        await _update_transformation_status(
            transformation_id,
            "failed",
            0,
            f"Transformation failed: {str(e)}"
        )

async def _update_transformation_status(
    transformation_id: str,
    status: str,
    progress: int,
    message: str,
    result_data: Optional[Dict[str, Any]] = None
):
    """Update transformation status"""
    try:
        status_data = {
            "transformation_id": transformation_id,
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": time.time(),
            "created_at": time.time()  # Will be overwritten if file exists
        }
        
        if result_data:
            status_data["result"] = result_data
        
        # Load existing status to preserve created_at
        status_file = Path(settings.OUTPUT_DIR) / f"{transformation_id}_status.json"
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    existing_data = json.load(f)
                status_data["created_at"] = existing_data.get("created_at", time.time())
            except:
                pass
        
        # Save status
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2, default=str)
            
    except Exception as e:
        print(f"Failed to update transformation status: {e}")

def _estimate_processing_time(num_variations: int, quality_mode: str) -> int:
    """Estimate processing time based on parameters"""
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