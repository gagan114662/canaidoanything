from celery import current_task
from app.services.celery_app import celery_app
from app.services.ai.flux_service import FluxService
from app.services.ai.sam2_service import SAM2Service
from app.services.ai.esrgan_service import ESRGANService
from app.utils.image_utils import load_image, save_image, resize_image
from app.core.config import settings
from app.utils.logging_config import TaskLogger # Import TaskLogger
from pathlib import Path
import time
# import traceback # No longer needed if using logger with exc_info
import os
# import logging # Not strictly needed if TaskLogger handles it, but good for direct logger use if any

# logger = logging.getLogger(__name__) # Standard logger if needed outside TaskLogger

@celery_app.task(bind=True)
def process_garment_image(self, request_data: dict): # Added type hint for request_data
    """
    Main task for processing garment images through the AI pipeline.
    Uses TaskLogger for structured logging.
    """
    task_id = request_data.get('task_id', self.request.id) # Use provided ID or Celery's

    with TaskLogger(task_id=str(task_id), task_name=self.name) as task_logger:
        try:
            # task_logger.info(f"Full request data: {request_data}") # Be cautious with logging full input_file_path
            task_logger.info(
                f"Processing garment image. Style: '{request_data.get('style_prompt', '')[:50]}...', "
                f"Enhance: {request_data.get('enhance_quality', True)}, BG Remove: {request_data.get('remove_background', False)}, "
                f"Upscale: {request_data.get('upscale', True)}."
            )

            input_path = request_data['input_file_path']
            style_prompt = request_data['style_prompt']
            negative_prompt = request_data.get('negative_prompt', '')
            enhance_quality = request_data.get('enhance_quality', True)
            remove_background = request_data.get('remove_background', False)
            upscale = request_data.get('upscale', True)

            start_time = time.time()

            self.update_state(state='PROGRESS', meta={'progress': 5, 'message': 'Loading image...'})
            task_logger.info(f"Loading image from: {input_path}")
            image = load_image(input_path)
            if image is None:
                task_logger.error(f"Failed to load image from path: {input_path}")
                raise ValueError(f"Could not load image from {input_path}")
            original_size = image.size
            task_logger.info(f"Image loaded. Original size: {original_size}")

            if max(image.size) > settings.MAX_IMAGE_SIZE:
                task_logger.info(f"Image size {image.size} exceeds max {settings.MAX_IMAGE_SIZE}. Resizing.")
                image = resize_image(image, settings.MAX_IMAGE_SIZE)
                task_logger.info(f"Image resized to: {image.size}")

            processed_image = image
            models_used = []

            if remove_background:
                self.update_state(state='PROGRESS', meta={'progress': 20, 'message': 'Removing background...'})
                task_logger.info("Starting background removal.")
                sam2_service = SAM2Service() # Instantiation includes model load if not already loaded
                if not sam2_service.is_model_loaded() and sam2_service.SAM2_AVAILABLE: # Check if actual model load failed
                     task_logger.warning("SAM2 model failed to load previously, attempting again or using fallback.")
                processed_image = sam2_service.remove_background(processed_image)
                models_used.append('SAM2Service') # More descriptive
                task_logger.info("Background removal completed.")

            self.update_state(state='PROGRESS', meta={'progress': 40, 'message': 'Applying style transformation...'})
            task_logger.info("Starting style transformation with FLUX.")
            flux_service = FluxService()
            processed_image = flux_service.generate_styled_image(
                processed_image, style_prompt, negative_prompt
            )
            models_used.append('FluxService')
            task_logger.info("Style transformation completed.")

            if enhance_quality or upscale:
                self.update_state(state='PROGRESS', meta={'progress': 70, 'message': 'Enhancing quality/upscaling...'})
                task_logger.info(f"Starting quality enhancement/upscaling. Upscale: {upscale}, Enhance: {enhance_quality}")
                esrgan_service = ESRGANService()
                current_scale_factor = settings.UPSCALE_FACTOR if upscale else 1
                task_logger.info(f"Using ESRGAN with scale factor: {current_scale_factor}")
                processed_image = esrgan_service.upscale_image(
                    processed_image,
                    scale_factor=current_scale_factor
                )
                # Note: esrgan_service.enhance_quality could be a separate call if needed
                # For now, assuming upscale_image handles general quality.
                models_used.append('ESRGANService')
                task_logger.info("Quality enhancement/upscaling completed.")

            self.update_state(state='PROGRESS', meta={'progress': 90, 'message': 'Saving result...'})
            output_dir = Path(settings.OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            output_path = output_dir / f"{task_id}_processed.jpg"

            task_logger.info(f"Saving processed image to: {output_path}")
            save_image(processed_image, str(output_path))

            output_url = f"/api/v1/download/{task_id}" # Consider making base URL part of settings
            processing_time = time.time() - start_time
            task_logger.info(f"Image processing completed in {processing_time:.2f} seconds.")

            final_result = { # Renamed to avoid conflict with Celery 'result'
                'output_url': output_url,
                'metadata': {
                    'original_size': original_size,
                    'processed_size': processed_image.size,
                    'processing_time': round(processing_time, 2),
                    'models_used': models_used,
                    'enhancement_applied': enhance_quality,
                    'background_removed': remove_background,
                    'upscaled': upscale
                }
            }

            self.update_state(state='SUCCESS', meta=final_result)
            task_logger.info("Task completed successfully.")
            return final_result

        except ValueError as ve: # Specific error for things like image loading
            error_msg = f"ValueError during image processing: {str(ve)}"
            task_logger.error(error_msg, exc_info=True)
            self.update_state(state='FAILURE', meta={'error': error_msg, 'details': str(ve)})
            raise # Re-raise to mark task as failed in Celery
        except IOError as ioe:
            error_msg = f"IOError during image processing (e.g. file not found, disk full): {str(ioe)}"
            task_logger.error(error_msg, exc_info=True)
            self.update_state(state='FAILURE', meta={'error': error_msg, 'details': str(ioe)})
            raise
        except Exception as e: # General catch-all
            # Using task_logger now, traceback.print_exc() is redundant if exc_info=True
            error_msg = f"Unexpected error processing image: {str(e)}"
            task_logger.error(error_msg, exc_info=True) # exc_info=True will log the traceback

            self.update_state(
                state='FAILURE',
                # meta={'error': error_msg, 'traceback': traceback.format_exc()} # No need for manual traceback in meta
                meta={'error': error_msg, 'details': str(e)}
            )
            raise # Re-raise to mark task as failed in Celery; Celery will store traceback