import torch
from diffusers import FluxPipeline
from PIL import Image
import numpy as np
from typing import Optional, Union
import logging
from app.core.config import settings
from app.core.exceptions import ModelLoadError # Import custom exception

logger = logging.getLogger(__name__)

class FluxService:
    """Service for FLUX 1.1 image generation and style transfer"""
    
    MODEL_NAME = "FLUX 1.1" # Class attribute for model name

    def __init__(self):
        self.pipeline = None
        self.model_loaded = False

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if settings.GPU_DEVICE_ID is not None:
                if settings.GPU_DEVICE_ID < num_gpus:
                    self.device = f"cuda:{settings.GPU_DEVICE_ID}"
                else:
                    logger.warning(
                        f"Invalid GPU_DEVICE_ID {settings.GPU_DEVICE_ID} (Available GPUs: {num_gpus}). "
                        f"Falling back to default CUDA device (cuda:0)."
                    )
                    self.device = "cuda:0" # Default to cuda:0 if ID is invalid
            else:
                self.device = "cuda" # Default to cuda (usually cuda:0) if no ID specified
        else:
            self.device = "cpu"
        logger.info(f"{self.MODEL_NAME}Service initialized. Device set to: {self.device}")
        
    def load_model(self):
        """Load the FLUX 1.1 model."""
        if self.model_loaded:
            logger.info(f"{self.MODEL_NAME} model is already loaded.")
            return

        model_path = settings.FLUX_MODEL_PATH
        logger.info(f"Loading {self.MODEL_NAME} model from {model_path} to device {self.device}...")

        try:
            self.pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None # Handles optional token
            )
            self.pipeline = self.pipeline.to(self.device)
            
            if self.device == "cuda": # Specific CUDA optimizations
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_attention_slicing()
            
            self.model_loaded = True
            logger.info(f"{self.MODEL_NAME} model loaded successfully from {model_path} on {self.device}.")
            
        except FileNotFoundError as e:
            logger.error(f"{self.MODEL_NAME} model file not found at {model_path}: {e}", exc_info=True)
            self.model_loaded = False
            raise ModelLoadError(model_name=self.MODEL_NAME, original_exception=e, path=model_path)
        except RuntimeError as e:
            logger.error(f"Runtime error loading {self.MODEL_NAME} model (e.g., CUDA OOM, HuggingFace Hub issue): {e}", exc_info=True)
            self.model_loaded = False
            raise ModelLoadError(model_name=self.MODEL_NAME, original_exception=e, path=model_path)
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"An unexpected error occurred while loading {self.MODEL_NAME} model: {e}", exc_info=True)
            self.model_loaded = False
            raise ModelLoadError(model_name=self.MODEL_NAME, original_exception=e, path=model_path)
    
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
        Generate a styled version of the input garment image.
        """
        logger.info(
            f"Generating styled image with FLUX. Style: '{style_prompt}', Strength: {strength}, "
            f"Guidance: {guidance_scale}, Steps: {num_inference_steps}."
        )
        if not self.model_loaded:
            logger.info("FLUX model not loaded, attempting to load now for generate_styled_image.")
            self.load_model() # This will raise an error if loading fails
        
        try:
            original_size = input_image.size
            # Ensure image is RGB
            if input_image.mode != "RGB":
                logger.debug(f"Input image mode is {input_image.mode}, converting to RGB.")
                input_image = input_image.convert("RGB")
            
            # Resize image to optimal size for FLUX
            target_size = (1024, 1024) # Consider making this configurable or model-dependent
            logger.debug(f"Resizing input image from {original_size} to {target_size}.")
            input_image_resized = input_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Create enhanced prompt for garment styling
            # Using f-string for clarity, strip() will be applied before passing to pipeline
            enhanced_prompt = (
                f"{style_prompt}, professional garment photography, high quality, detailed, "
                f"fashion photography, studio lighting, clean background, commercial photography, "
                f"high resolution, sharp focus, professional styling"
            )
            
            enhanced_negative_prompt = (
                f"{negative_prompt}, blurry, low quality, pixelated, amateur, bad lighting, "
                f"cluttered background, distorted, ugly, deformed, low resolution"
            )
            
            logger.debug(f"Enhanced prompt: {enhanced_prompt}")
            logger.debug(f"Enhanced negative prompt: {enhanced_negative_prompt}")

            # Generate styled image
            result = self.pipeline(
                prompt=enhanced_prompt.strip(),
                image=input_image_resized, # Use resized image
                negative_prompt=enhanced_negative_prompt.strip(),
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=self.device).manual_seed(settings.RANDOM_SEED) # Use configured seed
            )
            
            styled_image = result.images[0]
            logger.info("FLUX styled image generated successfully.")
            # Optionally, resize back to original aspect ratio or a desired output size
            # styled_image = styled_image.resize(original_size, Image.Resampling.LANCZOS)
            return styled_image
            
        except ValueError as e: # Catch specific errors from PIL or value issues
            logger.error(f"ValueError during FLUX style generation (e.g. invalid image data): {e}", exc_info=True)
            raise
        except RuntimeError as e: # Catch errors from PyTorch/diffusers
            logger.error(f"RuntimeError during FLUX style generation (e.g. CUDA OOM): {e}", exc_info=True)
            raise
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"An unexpected error occurred in FLUX style generation: {e}", exc_info=True)
            raise
    
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
        Generate image from text prompt only.
        """
        logger.info(
            f"Generating text-to-image with FLUX. Prompt: '{prompt}', Size: {width}x{height}, "
            f"Guidance: {guidance_scale}, Steps: {num_inference_steps}."
        )
        if not self.model_loaded:
            logger.info("FLUX model not loaded, attempting to load now for text_to_image.")
            self.load_model() # Raises error if fails
        
        try:
            logger.debug(f"Text-to-image prompt: {prompt}")
            logger.debug(f"Text-to-image negative prompt: {negative_prompt}")

            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=self.device).manual_seed(settings.RANDOM_SEED) # Use configured seed
            )
            
            generated_image = result.images[0]
            logger.info("FLUX text-to-image generated successfully.")
            return generated_image
            
        except RuntimeError as e: # Catch errors from PyTorch/diffusers
            logger.error(f"RuntimeError during FLUX text-to-image generation: {e}", exc_info=True)
            raise
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"An unexpected error occurred in FLUX text-to-image generation: {e}", exc_info=True)
            raise
    
    def unload_model(self):
        """Unload model to free memory."""
        logger.info(f"Attempting to unload {self.MODEL_NAME} model...")
        if self.pipeline is not None:
            try:
                # If model has a .to method and is on CUDA, try moving to CPU first.
                # This is more relevant for torch.nn.Module, diffusers pipelines might not need/benefit as much.
                if self.device == "cuda" and hasattr(self.pipeline, 'to'):
                    try:
                        self.pipeline.to('cpu')
                        logger.debug(f"{self.MODEL_NAME} moved to CPU before unloading.")
                    except Exception as e_cpu:
                        logger.warning(f"Could not move {self.MODEL_NAME} to CPU before unloading: {e_cpu}", exc_info=True)

                del self.pipeline
                self.pipeline = None
                logger.debug(f"{self.MODEL_NAME} pipeline object deleted.")

                if torch.cuda.is_available():
                    logger.debug("Clearing CUDA cache after unloading FLUX model.")
                    torch.cuda.empty_cache()

                self.model_loaded = False # Set after successful unload
                logger.info(f"{self.MODEL_NAME} model unloaded successfully.")
            except Exception as e:
                logger.error(f"Error during {self.MODEL_NAME} model unloading: {e}", exc_info=True)
                # Even if an error occurs, try to ensure the state reflects that the model is likely no longer usable/loaded.
                self.pipeline = None
                self.model_loaded = False
                # raise # Optionally re-raise if critical
        else:
            logger.info(f"{self.MODEL_NAME} model was not loaded, no unload action needed.")
        # Ensure model_loaded is false if pipeline is None, regardless of how it got to be None
        if self.pipeline is None:
            self.model_loaded = False