import torch
import numpy as np
from PIL import Image
import asyncio
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
from pathlib import Path

from app.services.ai.model_enhancement_service import ModelEnhancementService
from app.services.ai.garment_optimization_service import GarmentOptimizationService
from app.services.ai.scene_generation_service import SceneGenerationService
from app.utils.image_utils import resize_image, enhance_image, load_image, save_image
from app.core.config import settings
from app.services.celery_app import celery_app
from app.models.schemas import ModelTransformationRequest # Assuming this schema is used

logger = logging.getLogger(__name__)

# Helper function for updating status (adapted from endpoint)
# Made synchronous for now, as Celery tasks are often synchronous.
# If Celery task is async, this can remain async.
def _update_transformation_status_sync(
    transformation_id: str,
    status: str,
    progress: int,
    message: str,
    result_data: Optional[Dict[str, Any]] = None
):
    """Update transformation status synchronously"""
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
            except json.JSONDecodeError as e: # More specific error
                logger.error(f"Error decoding JSON from existing status file {status_file} for task {transformation_id}: {e}", exc_info=True)
            except IOError as e:
                logger.error(f"IOError reading existing status file {status_file} for task {transformation_id}: {e}", exc_info=True)
            except Exception as e: # Catch any other unexpected error
                logger.error(f"Unexpected error reading existing status file {status_file} for task {transformation_id}: {e}", exc_info=True)

        # Save status
        status_file.parent.mkdir(parents=True, exist_ok=True) # This is fine
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2, default=str) # default=str is good for non-serializable like datetime or Path

    except IOError as e:
        logger.error(f"IOError updating transformation status file for {transformation_id} at {status_file}: {e}", exc_info=True)
    except Exception as e: # Catch any other unexpected error
        logger.error(f"Unexpected error updating transformation status for {transformation_id}: {e}", exc_info=True)


# Global pipeline instance - consider if this is appropriate for Celery workers
# Each worker process might get its own instance.
# If models are large, loading them per task/worker can be resource-intensive.
# A shared, pre-loaded instance might be better if managed carefully.
# For now, let's assume it's loaded on worker startup or first task.
_pipeline_instance = None
_pipeline_lock = threading.Lock()

def get_transformation_pipeline():
    """Gets a thread-safe instance of the pipeline, loading models if necessary."""
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                logger.info("Initializing ModelTransformationPipeline for Celery worker...")
                temp_pipeline = ModelTransformationPipeline()
                try:
                    temp_pipeline.load_all_models()
                    _pipeline_instance = temp_pipeline
                    logger.info("ModelTransformationPipeline loaded successfully for Celery worker.")
                except Exception as e: # Catch any exception during pipeline init or model loading
                    logger.error(f"Failed to initialize or load transformation pipeline for Celery worker: {e}", exc_info=True)
                    # Decide if we should raise an error or allow tasks to fail if pipeline is not loaded
                    raise # Raising error to prevent tasks from running with an uninitialized pipeline
    return _pipeline_instance


@celery_app.task(name="model_transformation.transform_model_task")
def transform_model_task(
    transformation_id: str,
    request_data_dict: dict  # Pass dict instead of Pydantic model
):
    """Celery task for processing model transformation"""
    logger.info(f"Celery task started for transformation_id: {transformation_id}")

    # Recreate ModelTransformationRequest from dict
    request_data = ModelTransformationRequest(**request_data_dict)

    try:
        pipeline = get_transformation_pipeline()
        if not pipeline.services_loaded:
            logger.warning(f"Pipeline models not loaded for task {transformation_id}. Attempting to load now.")
            pipeline.load_all_models() # Attempt to load again if not loaded.
            if not pipeline.services_loaded:
                raise Exception("Pipeline models could not be loaded.")

        # Update status to processing
        _update_transformation_status_sync(
            transformation_id,
            "processing",
            0,
            "Starting model transformation via Celery..."
        )

        # Load input image using the file path from request_data
        logger.info(f"Loading input image from: {request_data.input_file_path}")
        input_image = load_image(request_data.input_file_path) # Assuming load_image is available
        if input_image is None: # load_image should ideally raise an error if it fails
            logger.error(f"Failed to load image from path: {request_data.input_file_path} for task {transformation_id}")
            raise ValueError(f"Could not load image from path: {request_data.input_file_path}")

        # Process transformation
        logger.info(f"Calling pipeline.transform_model for transformation_id: {transformation_id} with quality_mode: {request_data.quality_mode}")
        result = pipeline.transform_model(
            transformation_id=transformation_id, # Pass transformation_id to the pipeline method
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
        if result.get("variations"):
            for i, variation_data in enumerate(result["variations"]):
                variation_path = output_dir / f"{transformation_id}_variation_{i}.jpg"

                # Ensure 'variation_image' key exists and contains a PIL Image
                pil_image = variation_data.get("variation_image")
                if not isinstance(pil_image, Image.Image):
                    logger.error(f"Variation {i} for {transformation_id} is not a PIL image, skipping save.")
                    continue

                save_success = save_image(pil_image, str(variation_path)) # Assuming save_image is available

                if save_success:
                    saved_variations.append({
                        "index": i,
                        "style_type": variation_data.get("style_type", "unknown"),
                        "file_path": str(variation_path),
                        "quality_score": variation_data.get("quality_score", 0.0),
                        "download_url": f"/api/v1/download-variation/{transformation_id}/{i}" # Adjust API path if needed
                    })
                else:
                    logger.warning(f"Failed to save variation {i} for {transformation_id}")
        else:
            logger.warning(f"No variations found in result for {transformation_id}")

        # Save complete result
        result_data_to_save = {
            "transformation_id": transformation_id,
            "status": "completed",
            "variations": saved_variations,
            "metadata": result.get("metadata", {}),
            "quality_scores": result.get("quality_scores", {}),
            "performance_metrics": result.get("performance_metrics", {}),
            "request_data": request_data.dict()
        }

        result_file = output_dir / f"{transformation_id}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result_data_to_save, f, indent=2, default=str)

        # Update final status
        _update_transformation_status_sync(
            transformation_id,
            "completed",
            100,
            "Transformation completed successfully via Celery",
            result_data_to_save # Pass the saved result data
        )
        logger.info(f"Celery task completed successfully for transformation_id: {transformation_id}")
        return result_data_to_save # Return the final result

    except Exception as e:
        logger.error(f"Celery transformation task failed for {transformation_id}: {e}", exc_info=True)
        _update_transformation_status_sync(
            transformation_id,
            "failed",
            0,
            f"Transformation failed in Celery: {str(e)}"
        )
        # Optionally, re-raise to mark the task as failed in Celery
        raise # This will make Celery record the task as FAILED


class ModelTransformationPipeline:
    """Complete pipeline for transforming model photos into professional photoshoots"""
    
    def __init__(self):
        # Initialize services
        self.model_enhancement_service = ModelEnhancementService()
        self.garment_optimization_service = GarmentOptimizationService()
        self.scene_generation_service = SceneGenerationService()
        
        # Pipeline state
        self.services_loaded = False
        self.processing_stats = {
            "total_processed": 0,
            "average_processing_time": 0.0,
            "average_quality_score": 0.0
        }
        
        # Style configurations
        self.style_configs = {
            "editorial": {
                "style_prompt": "high fashion editorial photography, dramatic lighting, artistic composition, magazine quality",
                "negative_prompt": "amateur, casual, low quality, commercial, product photography",
                "background_type": "artistic",
                "lighting_type": "dramatic",
                "enhancement_level": 0.9,
                "creativity_level": 0.8
            },
            "commercial": {
                "style_prompt": "commercial product photography, clean studio lighting, professional presentation",
                "negative_prompt": "artistic, dramatic, editorial, personal, casual",
                "background_type": "studio",
                "lighting_type": "studio",
                "enhancement_level": 0.7,
                "creativity_level": 0.5
            },
            "lifestyle": {
                "style_prompt": "lifestyle photography, natural lighting, candid feel, everyday elegance",
                "negative_prompt": "studio, formal, posed, artificial, commercial",
                "background_type": "natural",
                "lighting_type": "natural",
                "enhancement_level": 0.6,
                "creativity_level": 0.7
            },
            "artistic": {
                "style_prompt": "artistic fashion photography, creative composition, unique perspective, avant-garde",
                "negative_prompt": "commercial, standard, basic, conventional, boring",
                "background_type": "abstract",
                "lighting_type": "creative",
                "enhancement_level": 0.8,
                "creativity_level": 0.9
            },
            "brand": {
                "style_prompt": "brand photography, signature style, professional presentation, brand consistent",
                "negative_prompt": "off-brand, inconsistent, generic, amateur",
                "background_type": "brand_environment",
                "lighting_type": "brand_standard",
                "enhancement_level": 0.8,
                "creativity_level": 0.6
            }
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "minimum_acceptable": 6.0,
            "good_quality": 7.5,
            "excellent_quality": 9.0
        }
        
        # Performance targets
        self.performance_targets = {
            "model_enhancement": 5.0,
            "garment_optimization": 8.0,
            "scene_generation": 12.0,
            "total_pipeline": 30.0
        }
    
    def load_all_models(self):
        """Load all AI models for the pipeline"""
        try:
            logger.info("Loading all models for transformation pipeline...")
            
            # Load models in parallel for faster startup
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self.model_enhancement_service.load_models): "model_enhancement",
                    executor.submit(self.garment_optimization_service.load_models): "garment_optimization",
                    executor.submit(self.scene_generation_service.load_models): "scene_generation"
                }
                
                for future in as_completed(futures):
                    service_name = futures[future]
                    try:
                        future.result() # Wait for the specific service to load
                        logger.info(f"'{service_name}' service loaded successfully.")
                    except Exception as e:
                        logger.error(f"Failed to load '{service_name}' service during parallel loading: {e}", exc_info=True)
                        # Optionally, re-raise or collect errors to decide if pipeline is usable

            # Check if all services actually loaded if individual errors don't halt execution
            # This depends on how robust each service's load_models() is.
            # For now, assume if no exception bubbles up from futures, it's loaded.
            self.services_loaded = True
            logger.info("Finished attempting to load all transformation pipeline models.")

        except Exception as e: # Catch error from ThreadPoolExecutor or other unexpected issues
            logger.error(f"An error occurred during the parallel loading of pipeline models: {e}", exc_info=True)
            # Decide on services_loaded state based on requirements.
            # If any model failing is critical, set self.services_loaded = False
            # Or rely on individual services' loaded status.
            # For now, current logic is to try to run in a fallback mode.
            self.services_loaded = False # Or true if some services might have loaded and fallback is desired
            logger.warning("Pipeline model loading process encountered an error. Status set to not fully loaded.")
            # raise # Optionally re-raise if this is critical
    
    def transform_model(
        self,
        transformation_id: str, # Added for logging context
        model_image: Image.Image,
        style_prompt: str,
        negative_prompt: str = "",
        num_variations: int = 5,
        enhance_model: bool = True,
        optimize_garment: bool = True,
        generate_scene: bool = True,
        quality_mode: str = "balanced",  # "fast", "balanced", "high"
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Transform model photo into professional photoshoot variations.
        """
        start_time = time.time()
        # Use the transformation_id passed from the Celery task for consistent logging
        logger.info(
            f"Starting model transformation for ID: {transformation_id}. "
            f"Params: Style='{style_prompt[:50]}...', Variations={num_variations}, EnhanceModel={enhance_model}, "
            f"OptimizeGarment={optimize_garment}, GenerateScene={generate_scene}, Quality={quality_mode}, Seed={seed}."
        )

        try:
            if not self.services_loaded:
                logger.warning(f"[{transformation_id}] Services not pre-loaded. Attempting to load now.")
                self.load_all_models() # Load models if they weren't loaded at startup/worker init
                if not self.services_loaded: # Check again after attempting load
                    logger.error(f"[{transformation_id}] Critical: Services still not loaded after attempt. Aborting transformation.")
                    # This should ideally not happen if load_all_models raises on critical failure
                    return self._create_fallback_result(model_image, num_variations, "Core services failed to load.", transformation_id)

            if seed is not None:
                logger.debug(f"[{transformation_id}] Setting random seed to: {seed}")
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            logger.debug(f"[{transformation_id}] Preprocessing input image. Original size: {model_image.size}, Mode: {model_image.mode}")
            processed_image = self._preprocess_input(model_image, quality_mode, transformation_id)
            
            stage_times = {}
            stage_results = {}
            error_handling = {"fallbacks_used": 0, "errors": []}
            
            current_image_state = processed_image # Keep track of the image as it's processed

            # Stage 1: Model Enhancement
            if enhance_model:
                logger.info(f"[{transformation_id}] Starting model enhancement stage.")
                stage_start_time = time.time()
                try:
                    enhancement_result = self.enhance_model_step(current_image_state, transformation_id)
                    current_image_state = enhancement_result["enhanced_image"]
                    stage_results["model_enhancement"] = enhancement_result
                except Exception as e:
                    logger.error(f"[{transformation_id}] Model enhancement stage failed: {e}", exc_info=True)
                    error_handling["fallbacks_used"] += 1
                    error_handling["errors"].append(f"Model enhancement: {str(e)}")
                stage_times["model_enhancement"] = time.time() - stage_start_time
                logger.info(f"[{transformation_id}] Model enhancement stage completed in {stage_times['model_enhancement']:.2f}s.")

            # Stage 2: Garment Optimization
            if optimize_garment:
                logger.info(f"[{transformation_id}] Starting garment optimization stage.")
                stage_start_time = time.time()
                try:
                    garment_result = self.optimize_garment_step(current_image_state, transformation_id)
                    current_image_state = garment_result["optimized_image"]
                    stage_results["garment_optimization"] = garment_result
                except Exception as e:
                    logger.error(f"[{transformation_id}] Garment optimization stage failed: {e}", exc_info=True)
                    error_handling["fallbacks_used"] += 1
                    error_handling["errors"].append(f"Garment optimization: {str(e)}")
                stage_times["garment_optimization"] = time.time() - stage_start_time
                logger.info(f"[{transformation_id}] Garment optimization stage completed in {stage_times['garment_optimization']:.2f}s.")
            
            # Stage 3: Generate Style Variations
            variations = []
            variation_times = []
            if num_variations > 0:
                logger.info(f"[{transformation_id}] Starting style variation generation for {num_variations} variations.")
                style_types = list(self.style_configs.keys())[:num_variations] # Determine styles to generate
                
                if quality_mode == "fast":
                    logger.info(f"[{transformation_id}] Generating variations sequentially (quality_mode: {quality_mode}).")
                    variations = self._generate_variations_sequential(
                        current_image_state, style_prompt, negative_prompt, style_types, generate_scene, transformation_id
                    )
                else:
                    logger.info(f"[{transformation_id}] Generating variations in parallel (quality_mode: {quality_mode}).")
                    variations = self._generate_variations_parallel(
                        current_image_state, style_prompt, negative_prompt, style_types, generate_scene, transformation_id
                    )
                variation_times = [v.get("processing_time", 0.0) for v in variations]
                logger.info(f"[{transformation_id}] Finished generating {len(variations)} variations.")
            
            logger.debug(f"[{transformation_id}] Calculating quality scores.")
            quality_scores = self._calculate_quality_scores(variations, stage_results, transformation_id)
            
            total_processing_time = time.time() - start_time
            logger.debug(f"[{transformation_id}] Updating processing stats.")
            self._update_processing_stats(total_processing_time, quality_scores.get("overall_average", 0.0), transformation_id)
            
            final_result = {
                "transformation_id": transformation_id, # Ensure this is the one from Celery task
                "variations": variations,
                "metadata": {
                    "original_image_size": model_image.size,
                    "processing_stages": list(stage_results.keys()),
                    "stage_times": {k: round(v, 2) for k, v in stage_times.items()}, # Rounded times
                    "variation_times": [round(vt, 2) for vt in variation_times], # Rounded times
                    "total_processing_time": round(total_processing_time, 2),
                    "quality_mode": quality_mode,
                    "num_variations_requested": num_variations,
                    "num_variations_generated": len(variations),
                    "seed_used": seed # Renamed from "seed" for clarity that it's the one used
                },
                "quality_scores": quality_scores,
                "performance_metrics": {
                    "meets_time_target": total_processing_time <= self.performance_targets["total_pipeline"],
                    "meets_quality_target": quality_scores.get("overall_average", 0.0) >= self.quality_thresholds["minimum_acceptable"],
                    "efficiency_score": self._calculate_efficiency_score(total_processing_time, quality_scores.get("overall_average", 0.0), transformation_id)
                },
                "error_handling": error_handling,
                # "stage_results": stage_results # Optional: can be large, include if needed for debug/client
            }
            
            logger.info(f"Transformation {transformation_id} completed successfully in {total_processing_time:.2f}s.")
            return final_result
            
        except Exception as e:
            # This is a critical failure for the entire pipeline
            logger.error(f"Model transformation pipeline failed critically for ID {transformation_id}: {e}", exc_info=True)
            return self._create_fallback_result(model_image, num_variations, str(e), transformation_id)
    
    def enhance_model_step(self, image: Image.Image, transformation_id: str) -> Dict[str, Any]: # Added transformation_id
        """Execute model enhancement step"""
        logger.debug(f"[{transformation_id}] Enhancing model...")
        try:
            start_time = time.time()
            
            enhancement_result = self.model_enhancement_service.enhance_model(image) # Assuming this service has its own logging
            
            original_quality = self.assess_image_quality(image, transformation_id, "original_for_enhancement") # Pass ID
            enhanced_quality = self.assess_image_quality(enhancement_result["enhanced_image"], transformation_id, "after_enhancement") # Pass ID
            quality_improvement = enhanced_quality - original_quality
            
            processing_time = time.time() - start_time
            logger.debug(f"[{transformation_id}] Model enhancement finished in {processing_time:.2f}s. Quality improvement: {quality_improvement:.2f}")
            return {
                "enhanced_image": enhancement_result["enhanced_image"],
                "enhancement_metadata": enhancement_result,
                "quality_improvement": quality_improvement,
                "processing_time": processing_time,
                "error_occurred": False
            }
            
        except Exception as e:
            logger.error(f"[{transformation_id}] Model enhancement step failed: {e}", exc_info=True)
            return {
                "enhanced_image": image,
                "enhancement_metadata": {"error": str(e), "details": "Model enhancement service failed."},
                "quality_improvement": 0.0,
                "processing_time": time.time() - start_time,
                "error_occurred": True
            }
    
    def optimize_garment_step(self, image: Image.Image, transformation_id: str) -> Dict[str, Any]: # Added transformation_id
        """Execute garment optimization step"""
        logger.debug(f"[{transformation_id}] Optimizing garment...")
        try:
            start_time = time.time()
            
            optimization_result = self.garment_optimization_service.optimize_garment(image) # Service should log internally
            
            processing_time = time.time() - start_time
            logger.debug(f"[{transformation_id}] Garment optimization finished in {processing_time:.2f}s. Score: {optimization_result.get('overall_score', 'N/A')}")
            return {
                "optimized_image": optimization_result["optimized_image"],
                "optimization_metadata": optimization_result,
                "garment_quality_score": optimization_result.get("overall_score", 5.0),
                "processing_time": processing_time,
                "error_occurred": False
            }
            
        except Exception as e:
            logger.error(f"[{transformation_id}] Garment optimization step failed: {e}", exc_info=True)
            return {
                "optimized_image": image,
                "optimization_metadata": {"error": str(e), "details": "Garment optimization service failed."},
                "garment_quality_score": 0.0,
                "processing_time": time.time() - start_time,
                "error_occurred": True
            }
    
    def generate_scene_step(
        self,
        image: Image.Image,
        style_prompt: str,
        background_type: str,
        transformation_id: str # Added transformation_id
    ) -> Dict[str, Any]:
        """Execute scene generation step"""
        logger.debug(f"[{transformation_id}] Generating scene. Style: '{style_prompt[:30]}...', Background: {background_type}")
        try:
            start_time = time.time()
            
            scene_result = self.scene_generation_service.compose_scene(
                model_image=image,
                background_type=background_type,
                composition_style=self._extract_composition_style(style_prompt, transformation_id) # Pass ID
            )
            
            lighting_result = self.scene_generation_service.apply_lighting(
                scene_result["composed_scene"],
                lighting_type=self._extract_lighting_type(style_prompt, transformation_id) # Pass ID
            )
            
            final_scene = lighting_result["lit_image"]
            comp_score = scene_result.get("composition_score", 0.0)
            light_qual = lighting_result.get("lighting_quality", 0.0)
            scene_quality = (comp_score + light_qual * 10) / 2.0
            
            processing_time = time.time() - start_time
            logger.debug(f"[{transformation_id}] Scene generation finished in {processing_time:.2f}s. Quality: {scene_quality:.2f}")
            return {
                "scene_image": final_scene,
                "scene_metadata": {"composition_result": scene_result, "lighting_result": lighting_result},
                "scene_quality_score": scene_quality,
                "processing_time": processing_time,
                "error_occurred": False
            }
            
        except Exception as e:
            logger.error(f"[{transformation_id}] Scene generation step failed: {e}", exc_info=True)
            return {
                "scene_image": image,
                "scene_metadata": {"error": str(e), "details": "Scene generation service failed."},
                "scene_quality_score": 0.0,
                "processing_time": time.time() - start_time,
                "error_occurred": True
            }
    
    def generate_style_variation(
        self,
        image: Image.Image,
        style_config: Dict[str, Any],
        variation_name: str,
        transformation_id: str # Added transformation_id
    ) -> Dict[str, Any]:
        """Generate a single style variation"""
        logger.debug(f"[{transformation_id}] Generating style variation: {variation_name}")
        try:
            start_time = time.time()
            
            enhanced_image = self._apply_style_enhancements(image, style_config, transformation_id) # Pass ID
            
            scene_quality = 8.0 # Default for no scene generation
            if style_config.get("background_type"):
                logger.debug(f"[{transformation_id}] Variation '{variation_name}' requires scene generation.")
                scene_result = self.generate_scene_step( # Pass transformation_id
                    enhanced_image, style_config["style_prompt"], style_config["background_type"], transformation_id
                )
                if scene_result.get("error_occurred"): # Check if sub-step failed
                    logger.warning(f"[{transformation_id}] Scene generation failed for variation '{variation_name}'. Using enhanced image without new scene.")
                    final_image = enhanced_image # Use image before scene attempt
                else:
                    final_image = scene_result["scene_image"]
                    scene_quality = scene_result["scene_quality_score"]
            else:
                final_image = enhanced_image
                logger.debug(f"[{transformation_id}] Variation '{variation_name}' does not require scene generation.")

            style_consistency = self._assess_style_consistency(final_image, style_config, transformation_id) # Pass ID
            quality_score = self.assess_image_quality(final_image, transformation_id, f"variation_{variation_name}") # Pass ID
            
            processing_time = time.time() - start_time
            logger.info(f"[{transformation_id}] Style variation '{variation_name}' generated in {processing_time:.2f}s. Quality: {quality_score:.2f}")
            return {
                "variation_image": final_image,
                "style_type": variation_name,
                "style_consistency": style_consistency,
                "quality_score": quality_score,
                "scene_quality": scene_quality,
                "processing_time": processing_time,
                "style_config_used": style_config,
                "error_occurred": False
            }
            
        except Exception as e:
            logger.error(f"[{transformation_id}] Style variation generation for '{variation_name}' failed: {e}", exc_info=True)
            return {
                "variation_image": image,
                "style_type": variation_name,
                "style_consistency": 0.0,
                "quality_score": 0.0,
                "scene_quality": 0.0,
                "processing_time": time.time() - start_time,
                "error": str(e),
                "error_occurred": True
            }
    
    def _preprocess_input(self, image: Image.Image, quality_mode: str, transformation_id: str) -> Image.Image: # Added transformation_id
        """Preprocess input image based on quality mode."""
        logger.debug(f"[{transformation_id}] Preprocessing input. Quality mode: {quality_mode}")
        try:
            processed_image = image.copy()
            
            resize_thresholds = {"fast": 512, "balanced": 1024, "high": 2048}
            max_size = resize_thresholds.get(quality_mode, 1024)
            
            if max(processed_image.size) > max_size:
                logger.debug(f"[{transformation_id}] Resizing image from {processed_image.size} to max_size {max_size}.")
                processed_image = resize_image(processed_image, max_size)
            
            if quality_mode in ["balanced", "high"]:
                logger.debug(f"[{transformation_id}] Applying basic enhancement for quality mode '{quality_mode}'.")
                processed_image = enhance_image(processed_image, brightness=1.05, contrast=1.05)
            
            logger.info(f"[{transformation_id}] Input preprocessing completed. New size: {processed_image.size}")
            return processed_image
            
        except Exception as e:
            logger.error(f"[{transformation_id}] Input preprocessing failed: {e}", exc_info=True)
            return image
    
    def _generate_variations_sequential(
        self,
        image: Image.Image,
        style_prompt: str,
        negative_prompt: str,
        style_types: List[str],
        generate_scene: bool
    ) -> List[Dict[str, Any]]:
        """Generate variations sequentially (faster, lower resource usage)"""
        variations = []
        
        for style_type in style_types:
            try:
                # Get style configuration
                style_config = self.style_configs[style_type].copy()
                
                # Merge with user prompts
                style_config["style_prompt"] = f"{style_config['style_prompt']}, {style_prompt}"
                style_config["negative_prompt"] = f"{style_config['negative_prompt']}, {negative_prompt}"
                
                # Generate variation
                variation = self.generate_style_variation(image, style_config, style_type)
                variations.append(variation)
                
            except Exception as e:
                logger.error(f"Failed to generate {style_type} variation: {e}")
                # Add fallback variation
                variations.append(self._create_fallback_variation(image, style_type))
        
        return variations
    
    def _generate_variations_parallel(
        self,
        image: Image.Image,
        style_prompt: str,
        negative_prompt: str,
        style_types: List[str],
        generate_scene: bool
    ) -> List[Dict[str, Any]]:
        """Generate variations in parallel (higher quality, more resources)"""
        variations = []
        
        def generate_single_variation(style_type):
            try:
                style_config = self.style_configs[style_type].copy()
                style_config["style_prompt"] = f"{style_config['style_prompt']}, {style_prompt}"
                style_config["negative_prompt"] = f"{style_config['negative_prompt']}, {negative_prompt}"
                
                return self.generate_style_variation(image, style_config, style_type)
            except Exception as e:
                logger.error(f"Parallel variation generation failed for {style_type}: {e}")
                return self._create_fallback_variation(image, style_type)
        
        # Use ThreadPoolExecutor for parallel generation
        with ThreadPoolExecutor(max_workers=min(len(style_types), 3)) as executor:
            future_to_style = {
                executor.submit(generate_single_variation, style_type): style_type 
                for style_type in style_types
            }
            
            for future in as_completed(future_to_style):
                style_type = future_to_style[future]
                try:
                    variation = future.result()
                    variations.append(variation)
                except Exception as e:
                    logger.error(f"Parallel processing failed for {style_type}: {e}")
                    variations.append(self._create_fallback_variation(image, style_type))
        
        # Sort by original order
        style_order = {style: i for i, style in enumerate(style_types)}
        variations.sort(key=lambda x: style_order.get(x["style_type"], 999))
        
        return variations
    
    def _apply_style_enhancements(
        self,
        image: Image.Image,
        style_config: Dict[str, Any]
    ) -> Image.Image:
        """Apply style-specific enhancements to image"""
        try:
            enhanced = image.copy()
            enhancement_level = style_config.get("enhancement_level", 0.7)
            
            # Apply enhancements based on style
            if "editorial" in style_config.get("style_prompt", "").lower():
                # Editorial: Higher contrast, dramatic
                enhanced = enhance_image(
                    enhanced,
                    contrast=1.0 + enhancement_level * 0.3,
                    sharpness=1.0 + enhancement_level * 0.2
                )
            elif "commercial" in style_config.get("style_prompt", "").lower():
                # Commercial: Clean, bright
                enhanced = enhance_image(
                    enhanced,
                    brightness=1.0 + enhancement_level * 0.1,
                    saturation=1.0 + enhancement_level * 0.1
                )
            elif "lifestyle" in style_config.get("style_prompt", "").lower():
                # Lifestyle: Soft, natural
                enhanced = enhance_image(
                    enhanced,
                    saturation=1.0 - enhancement_level * 0.05,
                    brightness=1.0 + enhancement_level * 0.05
                )
            elif "artistic" in style_config.get("style_prompt", "").lower():
                # Artistic: Creative, enhanced colors
                enhanced = enhance_image(
                    enhanced,
                    saturation=1.0 + enhancement_level * 0.2,
                    contrast=1.0 + enhancement_level * 0.15
                )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Style enhancement failed: {e}")
            return image
    
    def _extract_composition_style(self, style_prompt: str) -> str:
        """Extract composition style from prompt"""
        prompt_lower = style_prompt.lower()
        
        if any(word in prompt_lower for word in ["editorial", "dramatic", "artistic"]):
            return "editorial"
        elif any(word in prompt_lower for word in ["commercial", "product", "catalog"]):
            return "commercial"
        elif any(word in prompt_lower for word in ["lifestyle", "casual", "natural"]):
            return "lifestyle"
        else:
            return "commercial"  # Default
    
    def _extract_lighting_type(self, style_prompt: str) -> str:
        """Extract lighting type from prompt"""
        prompt_lower = style_prompt.lower()
        
        if any(word in prompt_lower for word in ["dramatic", "moody", "artistic"]):
            return "dramatic"
        elif any(word in prompt_lower for word in ["natural", "outdoor", "lifestyle"]):
            return "natural"
        else:
            return "studio"  # Default
    
    def _assess_style_consistency(
        self,
        image: Image.Image,
        style_config: Dict[str, Any]
    ) -> float:
        """Assess how well image matches the intended style"""
        try:
            # Simplified style consistency assessment
            # In production, this would use more sophisticated analysis
            
            base_score = 0.85  # Base consistency score
            
            # Adjust based on style type
            style_prompt = style_config.get("style_prompt", "").lower()
            
            if "editorial" in style_prompt:
                # Editorial should have higher contrast
                # This is a simplified check
                base_score += 0.05
            elif "commercial" in style_prompt:
                # Commercial should be clean and bright
                base_score += 0.03
            
            return min(base_score, 1.0)
            
        except Exception as e:
            logger.error(f"Style consistency assessment failed: {e}")
            return 0.8  # Default
    
    def assess_image_quality(self, image: Image.Image) -> float:
        """Assess overall image quality (0-10 scale)"""
        try:
            # Convert to array for analysis
            img_array = np.array(image)
            
            # Basic quality metrics
            quality_score = 7.0  # Base score
            
            # 1. Sharpness (variance of Laplacian)
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Calculate sharpness
            import cv2
            laplacian_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000, 1.0)
            
            # 2. Brightness distribution
            brightness = np.mean(img_array)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # 3. Color distribution (if color image)
            if len(img_array.shape) == 3:
                color_std = np.std(img_array, axis=(0, 1))
                color_score = min(np.mean(color_std) / 50, 1.0)
            else:
                color_score = 0.7
            
            # Combine scores
            quality_score = (
                sharpness_score * 0.4 +
                brightness_score * 0.3 +
                color_score * 0.3
            ) * 10
            
            return min(quality_score, 10.0)
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 7.0  # Default good score
    
    def _calculate_quality_scores(
        self,
        variations: List[Dict[str, Any]],
        stage_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality scores"""
        try:
            if not variations:
                return {
                    "overall_average": 5.0,
                    "variation_scores": [],
                    "stage_scores": {}
                }
            
            # Extract variation scores
            variation_scores = [v.get("quality_score", 5.0) for v in variations]
            style_consistency_scores = [v.get("style_consistency", 0.8) for v in variations]
            scene_quality_scores = [v.get("scene_quality", 5.0) for v in variations]
            
            # Calculate averages
            avg_quality = np.mean(variation_scores)
            avg_style_consistency = np.mean(style_consistency_scores)
            avg_scene_quality = np.mean(scene_quality_scores)
            
            # Stage-specific scores
            stage_scores = {}
            if "model_enhancement" in stage_results:
                stage_scores["model_enhancement"] = stage_results["model_enhancement"].get("quality_improvement", 0)
            if "garment_optimization" in stage_results:
                stage_scores["garment_optimization"] = stage_results["garment_optimization"].get("garment_quality_score", 5.0)
            
            # Overall score
            overall_score = (avg_quality * 0.5 + avg_style_consistency * 10 * 0.3 + avg_scene_quality * 0.2)
            
            return {
                "overall_average": overall_score,
                "variation_scores": variation_scores,
                "style_consistency_average": avg_style_consistency,
                "scene_quality_average": avg_scene_quality,
                "stage_scores": stage_scores,
                "quality_distribution": {
                    "min": min(variation_scores),
                    "max": max(variation_scores),
                    "std": np.std(variation_scores)
                }
            }
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return {
                "overall_average": 7.0,
                "variation_scores": [7.0] * len(variations),
                "stage_scores": {}
            }
    
    def _calculate_efficiency_score(self, processing_time: float, quality_score: float) -> float:
        """Calculate efficiency score based on time and quality"""
        try:
            # Normalize time score (30s target)
            time_score = max(0, 1.0 - (processing_time - 30.0) / 30.0)
            
            # Normalize quality score (8.5 target)
            quality_normalized = quality_score / 10.0
            
            # Combine (favor quality over speed)
            efficiency = quality_normalized * 0.7 + time_score * 0.3
            
            return min(efficiency, 1.0)
            
        except Exception as e:
            logger.error(f"Efficiency calculation failed: {e}")
            return 0.7
    
    def _update_processing_stats(self, processing_time: float, quality_score: float):
        """Update running statistics"""
        try:
            self.processing_stats["total_processed"] += 1
            total = self.processing_stats["total_processed"]
            
            # Update running averages
            old_avg_time = self.processing_stats["average_processing_time"]
            self.processing_stats["average_processing_time"] = (
                (old_avg_time * (total - 1) + processing_time) / total
            )
            
            old_avg_quality = self.processing_stats["average_quality_score"]
            self.processing_stats["average_quality_score"] = (
                (old_avg_quality * (total - 1) + quality_score) / total
            )
            
        except Exception as e:
            logger.error(f"Stats update failed: {e}")
    
    def _create_fallback_variation(self, image: Image.Image, style_type: str) -> Dict[str, Any]:
        """Create fallback variation when generation fails"""
        return {
            "variation_image": image,
            "style_type": style_type,
            "style_consistency": 0.7,
            "quality_score": 6.0,
            "scene_quality": 6.0,
            "processing_time": 0.1,
            "fallback_used": True
        }
    
    def _create_fallback_result(
        self,
        original_image: Image.Image,
        num_variations: int,
        error_message: str
    ) -> Dict[str, Any]:
        """Create fallback result when pipeline fails"""
        # Create basic variations using original image
        variations = []
        for i in range(num_variations):
            style_type = list(self.style_configs.keys())[i % len(self.style_configs)]
            variations.append(self._create_fallback_variation(original_image, style_type))
        
        return {
            "transformation_id": str(uuid.uuid4()),
            "variations": variations,
            "metadata": {
                "original_image_size": original_image.size,
                "processing_stages": [],
                "total_processing_time": 0.1,
                "fallback_mode": True
            },
            "quality_scores": {
                "overall_average": 6.0,
                "variation_scores": [6.0] * num_variations
            },
            "performance_metrics": {
                "meets_time_target": True,
                "meets_quality_target": False,
                "efficiency_score": 0.5
            },
            "error_handling": {
                "fallbacks_used": num_variations,
                "errors": [error_message],
                "fallback_mode": True
            }
        }
    
    def transform_batch(
        self,
        images: List[Image.Image],
        style_prompts: List[str],
        num_variations_per_image: int = 3
    ) -> List[Dict[str, Any]]:
        """Transform multiple images in batch"""
        results = []
        
        for i, (image, style_prompt) in enumerate(zip(images, style_prompts)):
            try:
                result = self.transform_model(
                    model_image=image,
                    style_prompt=style_prompt,
                    num_variations=num_variations_per_image,
                    quality_mode="balanced"
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch processing failed for image {i}: {e}")
                results.append(self._create_fallback_result(image, num_variations_per_image, str(e)))
        
        return results
    
    def clear_all_caches(self):
        """Clear all service caches"""
        try:
            self.model_enhancement_service.clear_cache()
            self.garment_optimization_service.clear_cache()
            self.scene_generation_service.clear_cache()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("All pipeline caches cleared")
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
    
    def unload_all_models(self):
        """Unload all models to free memory"""
        try:
            self.model_enhancement_service.unload_models()
            self.garment_optimization_service.unload_models()
            self.scene_generation_service.unload_models()
            
            self.services_loaded = False
            logger.info("All pipeline models unloaded")
            
        except Exception as e:
            logger.error(f"Model unloading failed: {e}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics"""
        return {
            "processing_stats": self.processing_stats.copy(),
            "services_loaded": self.services_loaded,
            "style_configs_available": list(self.style_configs.keys()),
            "quality_thresholds": self.quality_thresholds.copy(),
            "performance_targets": self.performance_targets.copy()
        }