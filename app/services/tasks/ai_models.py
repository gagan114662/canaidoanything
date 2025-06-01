from celery import current_task
from app.services.celery_app import celery_app
import time

@celery_app.task(bind=True)
def preload_models(self):
    """
    Task to preload AI models for faster processing
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'message': 'Loading FLUX model...'}
        )
        
        from app.services.ai.flux_service import FluxService
        flux_service = FluxService()
        flux_service.load_model()
        
        self.update_state(
            state='PROGRESS',
            meta={'progress': 40, 'message': 'Loading SAM2 model...'}
        )
        
        from app.services.ai.sam2_service import SAM2Service
        sam2_service = SAM2Service()
        sam2_service.load_model()
        
        self.update_state(
            state='PROGRESS',
            meta={'progress': 70, 'message': 'Loading Real-ESRGAN model...'}
        )
        
        from app.services.ai.esrgan_service import ESRGANService
        esrgan_service = ESRGANService()
        esrgan_service.load_model()
        
        self.update_state(
            state='SUCCESS',
            meta={'progress': 100, 'message': 'All models loaded successfully'}
        )
        
        return {'status': 'success', 'message': 'All models preloaded'}
        
    except Exception as e:
        error_msg = f"Error preloading models: {str(e)}"
        
        self.update_state(
            state='FAILURE',
            meta={'error': error_msg}
        )
        
        raise Exception(error_msg)

@celery_app.task
def health_check_models():
    """
    Task to check if all AI models are healthy
    """
    try:
        from app.services.ai.flux_service import FluxService
        from app.services.ai.sam2_service import SAM2Service
        from app.services.ai.esrgan_service import ESRGANService
        
        flux_service = FluxService()
        sam2_service = SAM2Service()
        esrgan_service = ESRGANService()
        
        # Basic health checks
        flux_healthy = flux_service.is_model_loaded()
        sam2_healthy = sam2_service.is_model_loaded()
        esrgan_healthy = esrgan_service.is_model_loaded()
        
        return {
            'flux': flux_healthy,
            'sam2': sam2_healthy,
            'esrgan': esrgan_healthy,
            'overall_healthy': all([flux_healthy, sam2_healthy, esrgan_healthy])
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'overall_healthy': False
        }