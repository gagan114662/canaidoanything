import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional
import logging
from app.core.config import settings

# Real-ESRGAN integration
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_AVAILABLE = True
except ImportError:
    ESRGAN_AVAILABLE = False
    logging.warning("Real-ESRGAN not available. Install Real-ESRGAN package for upscaling.")

logger = logging.getLogger(__name__)

class ESRGANService:
    """Service for Real-ESRGAN image upscaling and enhancement"""
    
    def __init__(self):
        self.upsampler = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the Real-ESRGAN model"""
        if not ESRGAN_AVAILABLE:
            logger.warning("Real-ESRGAN not available. Using fallback upscaling.")
            self.model_loaded = True
            return
            
        try:
            logger.info("Loading Real-ESRGAN model...")
            
            # Initialize RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )
            
            # Initialize RealESRGANer
            self.upsampler = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                gpu_id=0 if self.device == "cuda" else None
            )
            
            self.model_loaded = True
            logger.info("Real-ESRGAN model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN model: {str(e)}")
            # Fallback to basic upscaling
            self.model_loaded = True
            logger.info("Using fallback upscaling method")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded
    
    def upscale_image(
        self, 
        image: Image.Image,
        scale_factor: int = 4,
        enhance_face: bool = False
    ) -> Image.Image:
        """
        Upscale and enhance image quality
        
        Args:
            image: Input image
            scale_factor: Upscaling factor (2, 4, 8)
            enhance_face: Whether to apply face enhancement (for clothing with models)
            
        Returns:
            Enhanced and upscaled image
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            if ESRGAN_AVAILABLE and self.upsampler is not None:
                return self._esrgan_upscale(image, scale_factor, enhance_face)
            else:
                return self._fallback_upscale(image, scale_factor)
                
        except Exception as e:
            logger.error(f"Error in image upscaling: {str(e)}")
            # Return original image if upscaling fails
            return image
    
    def _esrgan_upscale(
        self, 
        image: Image.Image, 
        scale_factor: int = 4,
        enhance_face: bool = False
    ) -> Image.Image:
        """Upscale using Real-ESRGAN"""
        try:
            # Convert PIL to numpy array
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Enhance with Real-ESRGAN
            output, _ = self.upsampler.enhance(
                img_cv, 
                outscale=scale_factor,
                face_enhance=enhance_face
            )
            
            # Convert back to PIL
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(output_rgb)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Real-ESRGAN processing failed: {str(e)}")
            return self._fallback_upscale(image, scale_factor)
    
    def _fallback_upscale(
        self, 
        image: Image.Image, 
        scale_factor: int = 4
    ) -> Image.Image:
        """Fallback upscaling using PIL and OpenCV"""
        try:
            # Calculate new size
            new_width = image.width * scale_factor
            new_height = image.height * scale_factor
            
            # Use Lanczos resampling for high-quality upscaling
            upscaled = image.resize(
                (new_width, new_height), 
                Image.Resampling.LANCZOS
            )
            
            # Apply some sharpening
            img_cv = cv2.cvtColor(np.array(upscaled), cv2.COLOR_RGB2BGR)
            
            # Unsharp mask for enhancement
            gaussian = cv2.GaussianBlur(img_cv, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(img_cv, 1.5, gaussian, -0.5, 0)
            
            # Convert back to PIL
            result_rgb = cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(result_rgb)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Fallback upscaling failed: {str(e)}")
            # Simple resize as last resort
            new_width = image.width * scale_factor
            new_height = image.height * scale_factor
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def enhance_quality(
        self, 
        image: Image.Image,
        denoise: bool = True,
        sharpen: bool = True
    ) -> Image.Image:
        """
        Enhance image quality without upscaling
        
        Args:
            image: Input image
            denoise: Whether to apply denoising
            sharpen: Whether to apply sharpening
            
        Returns:
            Quality-enhanced image
        """
        try:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Denoising
            if denoise:
                img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75)
            
            # Sharpening
            if sharpen:
                # Create sharpening kernel
                kernel = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
                img_cv = cv2.filter2D(img_cv, -1, kernel)
            
            # Color enhancement
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Convert back to PIL
            result_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(result_rgb)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Quality enhancement failed: {str(e)}")
            return image
    
    def batch_upscale(
        self, 
        images: list[Image.Image], 
        scale_factor: int = 4
    ) -> list[Image.Image]:
        """
        Upscale multiple images in batch
        
        Args:
            images: List of input images
            scale_factor: Upscaling factor
            
        Returns:
            List of upscaled images
        """
        results = []
        for image in images:
            try:
                upscaled = self.upscale_image(image, scale_factor)
                results.append(upscaled)
            except Exception as e:
                logger.error(f"Failed to upscale image: {str(e)}")
                results.append(image)  # Return original if failed
        
        return results
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.upsampler:
            del self.upsampler
            self.upsampler = None
            self.model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Real-ESRGAN model unloaded")