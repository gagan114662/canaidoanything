from celery import current_task
from app.services.celery_app import celery_app
from app.services.ai.flux_service import FluxService
from app.services.ai.sam2_service import SAM2Service
from app.services.ai.esrgan_service import ESRGANService
from app.utils.image_utils import load_image, save_image, resize_image
from app.core.config import settings
from pathlib import Path
import time
import traceback
import os

@celery_app.task(bind=True)
def process_garment_image(self, request_data):
    """
    Main task for processing garment images through the AI pipeline
    """
    try:
        task_id = request_data['task_id']
        input_path = request_data['input_file_path']
        style_prompt = request_data['style_prompt']
        negative_prompt = request_data.get('negative_prompt', '')
        enhance_quality = request_data.get('enhance_quality', True)
        remove_background = request_data.get('remove_background', False)
        upscale = request_data.get('upscale', True)
        
        start_time = time.time()
        
        # Update task progress
        self.update_state(
            state='PROGRESS',
            meta={'progress': 5, 'message': 'Loading image...'}
        )
        
        # Load and validate input image
        image = load_image(input_path)
        original_size = image.size
        
        # Resize if too large
        if max(image.size) > settings.MAX_IMAGE_SIZE:
            image = resize_image(image, settings.MAX_IMAGE_SIZE)
        
        processed_image = image
        models_used = []
        
        # Step 1: Background removal (if requested)
        if remove_background:
            self.update_state(
                state='PROGRESS',
                meta={'progress': 20, 'message': 'Removing background...'}
            )
            sam2_service = SAM2Service()
            processed_image = sam2_service.remove_background(processed_image)
            models_used.append('SAM2')
        
        # Step 2: Style transformation with FLUX
        self.update_state(
            state='PROGRESS',
            meta={'progress': 40, 'message': 'Applying style transformation...'}
        )
        flux_service = FluxService()
        processed_image = flux_service.generate_styled_image(
            processed_image, 
            style_prompt, 
            negative_prompt
        )
        models_used.append('FLUX 1.1')
        
        # Step 3: Quality enhancement (if requested)
        if enhance_quality or upscale:
            self.update_state(
                state='PROGRESS',
                meta={'progress': 70, 'message': 'Enhancing quality...'}
            )
            esrgan_service = ESRGANService()
            processed_image = esrgan_service.upscale_image(
                processed_image, 
                scale_factor=settings.UPSCALE_FACTOR if upscale else 1
            )
            models_used.append('Real-ESRGAN')
        
        # Step 4: Save output
        self.update_state(
            state='PROGRESS',
            meta={'progress': 90, 'message': 'Saving result...'}
        )
        
        output_path = Path(settings.OUTPUT_DIR) / f"{task_id}_processed.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(processed_image, str(output_path))
        
        # Generate output URL
        output_url = f"/api/v1/download/{task_id}"
        
        processing_time = time.time() - start_time
        
        # Final result
        result = {
            'output_url': output_url,
            'metadata': {
                'original_size': original_size,
                'processed_size': processed_image.size,
                'processing_time': processing_time,
                'models_used': models_used,
                'enhancement_applied': enhance_quality,
                'background_removed': remove_background,
                'upscaled': upscale
            }
        }
        
        self.update_state(
            state='SUCCESS',
            meta=result
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        traceback.print_exc()
        
        self.update_state(
            state='FAILURE',
            meta={'error': error_msg, 'traceback': traceback.format_exc()}
        )
        
        raise Exception(error_msg)