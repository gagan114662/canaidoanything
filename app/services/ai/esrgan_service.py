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
    
    MODEL_NAME = "Real-ESRGAN" # Class attribute for model name

    def __init__(self):
        self.upsampler = None
        self.model_loaded = False

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if settings.GPU_DEVICE_ID is not None:
                if settings.GPU_DEVICE_ID < num_gpus:
                    self.device = f"cuda:{settings.GPU_DEVICE_ID}"
                    # For RealESRGANer, gpu_id parameter might be 0 even if on cuda:1 if CUDA_VISIBLE_DEVICES is set.
                    # The self.device will be used for torch.float16, and gpu_id for RealESRGANer if it differs.
                    # RealESRGANer's gpu_id is relative to visible devices.
                else:
                    logger.warning(
                        f"Invalid GPU_DEVICE_ID {settings.GPU_DEVICE_ID} (Available GPUs: {num_gpus}). "
                        f"Falling back to default CUDA device (cuda:0)."
                    )
                    self.device = "cuda:0"
            else:
                self.device = "cuda" # Default to cuda (usually cuda:0)
        else:
            self.device = "cpu"
        logger.info(f"{self.MODEL_NAME}Service initialized. Device set to: {self.device}")
        
    def load_model(self):
        """Load the Real-ESRGAN model"""
        if self.model_loaded and ESRGAN_AVAILABLE and self.upsampler is not None:
            logger.info("Real-ESRGAN model is already loaded.")
            return

        if not ESRGAN_AVAILABLE:
            logger.warning("Real-ESRGAN package not available. Service will rely on fallback upscaling methods.")
            # self.model_loaded = True # Implies fallback is "loaded"
            self.model_loaded = True # Or False if strict about ESRGAN model itself
            return
            
        logger.info(f"Loading Real-ESRGAN model to device {self.device}...")
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        logger.info(f"ESRGAN model URL: {model_url}")
        try:
            model = RRDBNet( # Define model architecture
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
            
            # Determine gpu_id for RealESRGANer. It expects an int (0, 1, ..) if using CUDA.
            # If self.device is "cuda:1", gpu_id should be 1. If "cuda" (implies "cuda:0"), it's 0.
            current_gpu_id = None
            if self.device.startswith("cuda"):
                if ":" in self.device:
                    try:
                        current_gpu_id = int(self.device.split(":")[1])
                    except ValueError:
                        logger.error(f"Could not parse GPU ID from device string: {self.device}. Defaulting gpu_id for RealESRGANer.")
                        current_gpu_id = 0 # Default to 0 if parsing fails
                else:
                    current_gpu_id = 0 # Default 'cuda' implies 'cuda:0'

            self.upsampler = RealESRGANer(
                scale=settings.ESRGAN_DEFAULT_SCALE_PARAM,
                model_path=settings.ESRGAN_MODEL_PATH, # Use actual setting
                model=model,
                tile=settings.ESRGAN_TILE_SIZE,
                tile_pad=settings.ESRGAN_TILE_PAD,
                pre_pad=settings.ESRGAN_PRE_PAD,
                half= (self.device != "cpu"), # half precision if on any CUDA device
                gpu_id=current_gpu_id
            )
            
            self.model_loaded = True
            logger.info(f"{self.MODEL_NAME} model loaded successfully from {settings.ESRGAN_MODEL_PATH} on {self.device} (RealESRGANer gpu_id: {current_gpu_id}).")
            
        except FileNotFoundError as e:
            logger.error(f"{self.MODEL_NAME} model file not found at {settings.ESRGAN_MODEL_PATH}: {e}", exc_info=True)
            self.model_loaded = False
            raise ModelLoadError(model_name=self.MODEL_NAME, original_exception=e, path=settings.ESRGAN_MODEL_PATH)
        except RuntimeError as e:
            logger.error(f"Runtime error loading {self.MODEL_NAME} model (e.g. CUDA OOM, HF issue): {e}", exc_info=True)
            self.model_loaded = False
            raise ModelLoadError(model_name=self.MODEL_NAME, original_exception=e, path=settings.ESRGAN_MODEL_PATH)
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading {self.MODEL_NAME} model: {e}", exc_info=True)
            self.model_loaded = False
            raise ModelLoadError(model_name=self.MODEL_NAME, original_exception=e, path=settings.ESRGAN_MODEL_PATH)

        # Similar to SAM2Service, if loading fails, current logic allows fallback.
        # If strict ESRGAN loading is required, re-raise exceptions.

    def is_model_loaded(self) -> bool:
        """Check if the Real-ESRGAN model is loaded and usable."""
        return ESRGAN_AVAILABLE and self.upsampler is not None and self.model_loaded
    
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
        if not self.is_model_loaded(): # Use the enhanced is_model_loaded()
            logger.warning(f"{self.MODEL_NAME} model not available/loaded. Attempting load for upscale_image.")
            try:
                self.load_model()
                if not self.is_model_loaded():
                    logger.error(f"{self.MODEL_NAME} failed to load. Using fallback for upscale_image.")
                    return self._fallback_upscale(image, scale_factor)
            except ModelLoadError:
                 logger.error(f"{self.MODEL_NAME} failed to load on demand. Using fallback for upscale_image.")
                 return self._fallback_upscale(image, scale_factor)
        
        logger.info(
            f"Request to upscale image of size {image.size}, mode {image.mode}. "
            f"Scale: {scale_factor}, Enhance Face: {enhance_face}."
        )
        try:
            logger.debug(f"Attempting upscaling with {self.MODEL_NAME} model.")
            return self._esrgan_upscale(image, scale_factor, enhance_face)
                
        except Exception as e: # Catch errors from _esrgan_upscale
            logger.error(f"Error during {self.MODEL_NAME} upscaling process: {e}", exc_info=True)
            logger.warning("ESRGAN upscaling failed. Returning original image.")
            return image # Return original image if _esrgan_upscale fails unexpectedly beyond its own fallback
    
    def _esrgan_upscale(
        self, 
        image: Image.Image, 
        scale_factor: int = 4, # Note: RealESRGANer is often initialized for a fixed scale.
                               # Dynamic outscale is supported by some versions/implementations.
        enhance_face: bool = False
    ) -> Image.Image:
        """Upscale using Real-ESRGAN."""
        logger.info(f"Performing upscaling with Real-ESRGAN. Target scale: {scale_factor}, Enhance Face: {enhance_face}")
        original_size = image.size
        try:
            # Convert PIL to numpy array (BGR format for OpenCV)
            if image.mode == 'RGBA':
                logger.debug("ESRGAN: Converting RGBA image to RGB for upscaling.")
                image_rgb = image.convert('RGB')
            elif image.mode != 'RGB':
                logger.debug(f"ESRGAN: Converting {image.mode} image to RGB for upscaling.")
                image_rgb = image.convert('RGB')
            else:
                image_rgb = image

            img_cv = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
            
            # Enhance with Real-ESRGAN
            # The `outscale` parameter of `enhance` method in RealESRGANer is what controls final scale.
            # Ensure the loaded model supports the desired scale or that RealESRGANer handles arbitrary scales.
            logger.debug(f"ESRGAN: Calling upsampler.enhance with outscale={scale_factor}, face_enhance={enhance_face}.")
            output_cv, _ = self.upsampler.enhance(
                img_cv,
                outscale=scale_factor,
                face_enhance=enhance_face
            )
            
            # Convert back to PIL
            output_rgb = cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(output_rgb)
            logger.info(f"Real-ESRGAN upscaling successful. Original size: {original_size}, New size: {result_image.size}")
            
            return result_image
            
        except RuntimeError as e: # Catch PyTorch/CUDA errors
            logger.error(f"RuntimeError during Real-ESRGAN processing: {e}", exc_info=True)
            logger.warning("Falling back to CV-based upscaling due to ESRGAN RuntimeError.")
            return self._fallback_upscale(image, scale_factor)
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error during Real-ESRGAN processing: {e}", exc_info=True)
            logger.warning("Falling back to CV-based upscaling due to unexpected ESRGAN error.")
            return self._fallback_upscale(image, scale_factor)
    
    def _fallback_upscale(
        self, 
        image: Image.Image, 
        scale_factor: int = 4
    ) -> Image.Image:
        """Fallback upscaling using PIL and OpenCV with sharpening."""
        logger.info(f"Performing fallback upscaling. Target scale: {scale_factor}")
        original_size = image.size
        try:
            new_width = image.width * scale_factor
            new_height = image.height * scale_factor
            logger.debug(f"Fallback: Resizing image from {original_size} to ({new_width}, {new_height}) using Lanczos.")
            
            upscaled_pil = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Apply some sharpening using OpenCV for better control
            img_cv = cv2.cvtColor(np.array(upscaled_pil), cv2.COLOR_RGB2BGR)
            
            logger.debug("Fallback: Applying Gaussian blur and unsharp masking.")
            gaussian = cv2.GaussianBlur(img_cv, (0,0), sigmaX=2.0, sigmaY=2.0) # sigma values can be tuned
            # Parameters for addWeighted: src1, alpha, src2, beta, gamma
            unsharp_mask = cv2.addWeighted(img_cv, 1.5, gaussian, -0.5, 0)
            
            result_rgb = cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(result_rgb)
            logger.info(f"Fallback upscaling successful. New size: {result_image.size}")
            
            return result_image
            
        except cv2.error as e: # Specific OpenCV errors
            logger.error(f"OpenCV error during fallback upscaling: {e}", exc_info=True)
        except Exception as e: # Other unexpected errors (e.g., PIL errors)
            logger.error(f"Unexpected error during fallback upscaling: {e}", exc_info=True)

        # If any error occurs in fallback's main path, try a simple resize as a last resort.
        logger.warning("Fallback upscaling with sharpening failed. Attempting simple resize.")
        try:
            new_width_lr = image.width * scale_factor
            new_height_lr = image.height * scale_factor
            return image.resize((new_width_lr, new_height_lr), Image.Resampling.LANCZOS)
        except Exception as final_fallback_e:
            logger.critical(f"Critical error in fallback's simple resize: {final_fallback_e}", exc_info=True)
            raise final_fallback_e # Or return original image if preferred
    
    def enhance_quality(
        self, 
        image: Image.Image,
        denoise: bool = True,
        sharpen: bool = True
    ) -> Image.Image:
        """
        Enhance image quality using various CV techniques (no upscaling).
        """
        logger.info(f"Request to enhance quality for image of size {image.size}. Denoise: {denoise}, Sharpen: {sharpen}.")
        original_mode = image.mode
        try:
            # Convert to BGR for OpenCV processing
            if image.mode == 'RGBA':
                image_rgb = image.convert('RGB')
                logger.debug("EnhanceQuality: Converted RGBA to RGB.")
            elif image.mode != 'RGB':
                image_rgb = image.convert('RGB')
                logger.debug(f"EnhanceQuality: Converted {image.mode} to RGB.")
            else:
                image_rgb = image

            img_cv = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
            
            if denoise:
                logger.debug("EnhanceQuality: Applying bilateral filter for denoising.")
                img_cv = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75) # Parameters can be tuned
            
            if sharpen:
                logger.debug("EnhanceQuality: Applying sharpening kernel.")
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]]) # Standard sharpening kernel
                img_cv = cv2.filter2D(img_cv, ddepth=-1, kernel=kernel) # ddepth=-1 maintains original depth
            
            # Color enhancement using CLAHE on L channel of LAB color space
            logger.debug("EnhanceQuality: Applying CLAHE for color/contrast enhancement.")
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB) # Convert to LAB
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=settings.CLAHE_CLIP_LIMIT, tileGridSize=(settings.CLAHE_TILE_GRID_SIZE, settings.CLAHE_TILE_GRID_SIZE))
            cl = clahe.apply(l_channel) # Apply CLAHE to L-channel
            
            merged_lab = cv2.merge((cl, a_channel, b_channel)) # Merge CLAHE L-channel with original A and B channels
            img_cv_enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR) # Convert back to BGR
            
            # Convert back to PIL
            result_rgb = cv2.cvtColor(img_cv_enhanced, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(result_rgb)
            
            # If original was RGBA, try to re-apply alpha or handle appropriately
            if original_mode == 'RGBA' and hasattr(image, 'split'):
                try:
                    _, _, _, alpha = image.split()
                    if alpha:
                        result_image.putalpha(alpha)
                        logger.debug("EnhanceQuality: Re-applied original alpha channel.")
                except Exception as e_alpha:
                    logger.warning(f"EnhanceQuality: Could not re-apply alpha channel: {e_alpha}", exc_info=True)

            logger.info("Quality enhancement successful.")
            return result_image
            
        except cv2.error as e:
            logger.error(f"OpenCV error during quality enhancement: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during quality enhancement: {e}", exc_info=True)

        logger.warning("Quality enhancement failed. Returning original image.")
        return image # Return original on failure
    
    def batch_upscale(
        self, 
        images: list[Image.Image], 
        scale_factor: int = 4,
        enhance_face: bool = False # Added enhance_face for consistency
    ) -> list[Image.Image]:
        """
        Upscale multiple images in batch.
        """
        logger.info(f"Request to batch upscale {len(images)} images. Scale: {scale_factor}, Enhance Face: {enhance_face}.")
        if not self.model_loaded: # Load model if not already loaded
            logger.info("ESRGAN model not loaded for batch_upscale. Calling load_model().")
            self.load_model()

        results = []
        for i, image in enumerate(images):
            logger.debug(f"Batch upscaling image {i+1}/{len(images)}.")
            try:
                # Pass enhance_face to upscale_image, which then passes to _esrgan_upscale
                upscaled_image = self.upscale_image(image, scale_factor, enhance_face)
                results.append(upscaled_image)
            except Exception as e:
                logger.error(f"Failed to upscale image {i+1} in batch: {e}", exc_info=True)
                results.append(image)  # Return original image if processing for this one failed
        
        logger.info(f"Batch upscaling completed. Processed {len(results)} images.")
        return results
    
    def unload_model(self):
        """Unload model to free memory."""
        logger.info(f"Attempting to unload {self.MODEL_NAME} model...")
        if self.upsampler is not None:
            try:
                # The actual model is within self.upsampler.model
                # Moving it to CPU before del is good practice if it's a large torch module.
                if hasattr(self.upsampler, 'model') and self.upsampler.model is not None and \
                   self.device.startswith("cuda") and hasattr(self.upsampler.model, 'to'):
                    try:
                        self.upsampler.model.to('cpu')
                        logger.debug(f"{self.MODEL_NAME}'s internal model moved to CPU.")
                    except Exception as e_cpu:
                        logger.warning(f"Could not move {self.MODEL_NAME}'s internal model to CPU: {e_cpu}", exc_info=True)

                del self.upsampler
                self.upsampler = None
                logger.debug(f"{self.MODEL_NAME} upsampler object deleted.")

                if torch.cuda.is_available():
                    logger.debug(f"Clearing CUDA cache after unloading {self.MODEL_NAME} model.")
                    torch.cuda.empty_cache()

                self.model_loaded = False
                logger.info(f"{self.MODEL_NAME} model unloaded successfully.")
            except Exception as e:
                logger.error(f"Error during {self.MODEL_NAME} model unloading: {e}", exc_info=True)
                self.upsampler = None
                self.model_loaded = False
        else:
            logger.info(f"{self.MODEL_NAME} model was not loaded, no unload action needed.")

        if self.upsampler is None:
            self.model_loaded = False