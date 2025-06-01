import torch
from diffusers import FluxPipeline
from PIL import Image
import numpy as np
from typing import Optional, Union
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class FluxService:
    """Service for FLUX 1.1 image generation and style transfer"""
    
    def __init__(self):
        self.pipeline = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the FLUX 1.1 model"""
        try:
            logger.info("Loading FLUX 1.1 model...")
            
            self.pipeline = FluxPipeline.from_pretrained(
                settings.FLUX_MODEL_PATH,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_attention_slicing()
            
            self.model_loaded = True
            logger.info("FLUX 1.1 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load FLUX model: {str(e)}")
            raise e
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded
    
    def generate_styled_image(
        self, 
        input_image: Image.Image, 
        style_prompt: str,
        negative_prompt: str = "",
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20
    ) -> Image.Image:
        """
        Generate a styled version of the input garment image
        
        Args:
            input_image: Input garment image
            style_prompt: Style description (e.g., "professional fashion photography, studio lighting")
            negative_prompt: What to avoid in the generation
            strength: How much to transform the original image (0.0-1.0)
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            # Ensure image is RGB
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")
            
            # Resize image to optimal size for FLUX
            target_size = (1024, 1024)
            input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Create enhanced prompt for garment styling
            enhanced_prompt = f"""
            {style_prompt}, professional garment photography, high quality, detailed, 
            fashion photography, studio lighting, clean background, commercial photography,
            high resolution, sharp focus, professional styling
            """
            
            enhanced_negative_prompt = f"""
            {negative_prompt}, blurry, low quality, pixelated, amateur, bad lighting,
            cluttered background, distorted, ugly, deformed, low resolution
            """
            
            # Generate styled image
            result = self.pipeline(
                prompt=enhanced_prompt.strip(),
                image=input_image,
                negative_prompt=enhanced_negative_prompt.strip(),
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Error in FLUX style generation: {str(e)}")
            raise e
    
    def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20
    ) -> Image.Image:
        """
        Generate image from text prompt only
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Error in FLUX text-to-image generation: {str(e)}")
            raise e
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("FLUX model unloaded")