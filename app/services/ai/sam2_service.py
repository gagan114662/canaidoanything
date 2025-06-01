import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional, List, Tuple
import logging
from app.core.config import settings

# Note: SAM2 integration - you may need to install sam2 package or use transformers
try:
    from transformers import Sam2Model, Sam2Processor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    # Ensure this warning is not duplicated if logger is configured at root
    # logging.warning("SAM2 not available. Install sam2 package for background removal.")

logger = logging.getLogger(__name__) # This should be at module level

if not SAM2_AVAILABLE: # Log once at module load if not available
    logger.warning("SAM2 HuggingFace package not available. SAM2Service will operate in fallback-only mode.")

from app.core.exceptions import ModelLoadError # Import custom exception

class SAM2Service:
    """Service for SAM 2 (Segment Anything Model 2) background removal and segmentation"""
    
    MODEL_NAME = "SAM2" # Class attribute for model name

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_loaded = False # Tracks if the actual SAM2 model is loaded
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"SAM2Service initialized. Device set to: {self.device}. SAM2 Package Available: {SAM2_AVAILABLE}")
        
    def load_model(self):
        """Load the SAM 2 model."""
        if self.model_loaded: # This implies SAM2_AVAILABLE and self.model is not None
            logger.info(f"{self.MODEL_NAME} model is already loaded and ready.")
            return

        if not SAM2_AVAILABLE:
            logger.warning(f"{self.MODEL_NAME} HuggingFace package not available. Cannot load model. Service will use fallbacks.")
            # self.model_loaded remains False, indicating the primary model isn't loaded.
            # Fallback methods will be used by default if called.
            return # Cannot proceed with loading the SAM2 model
            
        model_path = settings.SAM2_MODEL_PATH
        logger.info(f"Loading {self.MODEL_NAME} model from {model_path} to device {self.device}...")

        try:
            self.processor = Sam2Processor.from_pretrained(
                model_path,
                use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None
            )
            logger.info(f"{self.MODEL_NAME} processor loaded successfully from {model_path}.")
            
            self.model = Sam2Model.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None
            )
            self.model = self.model.to(self.device)
            self.model_loaded = True # Actual SAM2 model is loaded
            logger.info(f"{self.MODEL_NAME} model loaded successfully from {model_path} and moved to {self.device}.")
            
        except FileNotFoundError as e: # More specific for local path issues
            logger.error(f"{self.MODEL_NAME} model files not found at {model_path}: {e}", exc_info=True)
            self.model_loaded = False
            raise ModelLoadError(model_name=self.MODEL_NAME, original_exception=e, path=model_path)
        except RuntimeError as e: # Catches HuggingFace Hub errors (like 401, 404 for model ID) or PyTorch errors
            logger.error(f"Runtime error loading {self.MODEL_NAME} model from {model_path} (e.g. HF Hub issue, CUDA OOM): {e}", exc_info=True)
            self.model_loaded = False
            raise ModelLoadError(model_name=self.MODEL_NAME, original_exception=e, path=model_path)
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"An unexpected error occurred while loading {self.MODEL_NAME} model from {model_path}: {e}", exc_info=True)
            self.model_loaded = False
            raise ModelLoadError(model_name=self.MODEL_NAME, original_exception=e, path=model_path)
    
    def is_model_loaded(self) -> bool:
        """Check if the actual SAM2 model is loaded and usable."""
        # Considers SAM2 package availability and successful model/processor initialization.
        return SAM2_AVAILABLE and self.model is not None and self.processor is not None and self.model_loaded
    
    def remove_background(
        self, 
        image: Image.Image,
        return_mask: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """
        Remove background from image using SAM 2 or fallback method
        
        Args:
            image: Input image
            return_mask: Whether to return the mask along with the result
            
        Returns:
            Image with transparent background, optionally with mask
        """
        # This check ensures that if load_model() was called but failed (e.g. model file not found),
        # and self.model_loaded became false, we don't try to use a non-existent model.
        if not self.is_model_loaded(): # Use the enhanced is_model_loaded()
            logger.warning(f"{self.MODEL_NAME} model not available or not loaded. Attempting to load/re-load for remove_background.")
            try:
                self.load_model() # Attempt to load
                if not self.is_model_loaded(): # Check again
                    logger.error(f"{self.MODEL_NAME} model failed to load. Using fallback for remove_background.")
                    return self._fallback_background_removal(image, return_mask)
            except ModelLoadError: # Catch specific load error
                 logger.error(f"{self.MODEL_NAME} model failed to load on demand. Using fallback for remove_background.")
                 return self._fallback_background_removal(image, return_mask)
        
        logger.info(f"Request to remove background for image size {image.size}, mode {image.mode}.")
        try:
            # self.is_model_loaded() should be true here if we didn't go to fallback.
            logger.debug(f"Attempting background removal with {self.MODEL_NAME} model.")
            return self._sam2_background_removal(image, return_mask)
                
        except Exception as e: # Catch errors from _sam2_background_removal itself
            logger.error(f"Error during SAM2 background removal process: {e}", exc_info=True)
            logger.warning("Falling back to CV-based segmentation due to error in SAM2 processing.")
            try:
                if image.mode != "RGBA":
                    image = image.convert("RGBA")
                # Create a dummy mask (fully opaque) if needed for consistency
                dummy_mask = Image.new('L', image.size, 255)
                return (image, dummy_mask) if return_mask else image
            except Exception as fallback_e:
                logger.error(f"Error converting image to RGBA in fallback: {fallback_e}", exc_info=True)
                # If even conversion fails, this is tricky. Client might expect an Image object.
                # Simplest is to raise, or return None if API handles it. For now, let it raise.
                raise fallback_e # Or return a placeholder error image / None
    
    def _sam2_background_removal(
        self, 
        image: Image.Image, 
        return_mask: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """Background removal using SAM 2."""
        logger.info("Performing background removal with SAM2.")
        original_size = image.size
        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                logger.debug(f"SAM2: Input image mode is {image.mode}, converting to RGB.")
                image_rgb = image.convert("RGB")
            else:
                image_rgb = image
            
            # Process image for SAM2
            logger.debug("SAM2: Processing image with SAM2Processor.")
            inputs = self.processor(image_rgb, return_tensors="pt").to(self.device)
            
            # Generate mask
            logger.debug("SAM2: Generating masks with SAM2Model.")
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get the best mask
            # Assuming batch size is 1. masks shape: (batch_size, num_masks, height, width)
            masks = outputs.pred_masks[0].cpu().numpy() # Get masks for the first image in batch
            logger.debug(f"SAM2: Predicted masks shape: {masks.shape}")

            # Select the "best" mask - this might need more sophisticated logic
            # For now, let's assume the first mask is the primary one or sum them up if appropriate
            # Or use segmentation points if provided in a more advanced scenario
            if masks.ndim == 3 and masks.shape[0] > 0: # (num_masks, height, width)
                # Example: take the first mask, or sum masks if they represent different parts of one object
                mask_array = masks[0] # Taking the first mask by default
                # mask_array = np.sum(masks, axis=0) > 0.5 # Alternative: combine masks
            elif masks.ndim == 2: # (height, width)
                mask_array = masks
            else:
                logger.error(f"SAM2: Unexpected mask shape: {masks.shape}. Using fallback.")
                return self._fallback_background_removal(image, return_mask)

            # Normalize mask to 0-1 range if it's not already (e.g. if it's logits)
            if mask_array.dtype != np.uint8: # If not binary, assume logits/probabilities
                 mask_array = (mask_array > 0.5).astype(np.float32) # Binarize if needed

            # Convert mask to PIL Image
            logger.debug("SAM2: Converting mask to PIL Image.")
            mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')

            # Resize mask to original image size
            if mask_pil.size != original_size:
                logger.debug(f"SAM2: Resizing mask from {mask_pil.size} to original image size {original_size}.")
                mask_pil = mask_pil.resize(original_size, Image.Resampling.LANCZOS)
            
            # Apply mask to create transparent background
            image_rgba = image.convert("RGBA") # Ensure original image (before RGB conversion for model) is used for alpha
            image_rgba.putalpha(mask_pil)
            logger.info("SAM2 background removal successful.")
            
            return (image_rgba, mask_pil) if return_mask else image_rgba
            
        except RuntimeError as e: # Catch PyTorch/diffusers errors
            logger.error(f"RuntimeError during SAM2 processing: {e}", exc_info=True)
            logger.warning("Falling back to CV-based background removal due to SAM2 RuntimeError.")
            return self._fallback_background_removal(image, return_mask)
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error during SAM2 processing: {e}", exc_info=True)
            logger.warning("Falling back to CV-based background removal due to unexpected SAM2 error.")
            return self._fallback_background_removal(image, return_mask)
    
    def _fallback_background_removal(
        self, 
        image: Image.Image, 
        return_mask: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """Fallback background removal using traditional computer vision (GrabCut)."""
        logger.info("Performing fallback background removal using GrabCut.")
        original_mode = image.mode
        original_size = image.size
        try:
            # Convert PIL to OpenCV
            if image.mode == 'RGBA': # GrabCut works best on RGB
                img_rgb = image.convert('RGB')
                logger.debug("Fallback: Converted RGBA to RGB for GrabCut.")
            elif image.mode != 'RGB':
                img_rgb = image.convert('RGB')
                logger.debug(f"Fallback: Converted {image.mode} to RGB for GrabCut.")
            else:
                img_rgb = image

            img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            
            # Create mask using GrabCut algorithm
            mask = np.zeros(img_cv.shape[:2], np.uint8) # Mask for GrabCut
            bgd_model = np.zeros((1, 65), np.float64)    # Background model
            fgd_model = np.zeros((1, 65), np.float64)    # Foreground model
            
            # Define rectangle around the object (slightly smaller than image to get definite background)
            height, width = img_cv.shape[:2]
            rect_margin = int(min(width, height) * 0.05) # 5% margin
            rect = (rect_margin, rect_margin, width - 2*rect_margin, height - 2*rect_margin)
            logger.debug(f"Fallback: Defined GrabCut rectangle: {rect}")

            # Apply GrabCut
            logger.debug("Fallback: Applying GrabCut...")
            cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT) # 5 iterations
            
            # Create final mask where 0 and 2 are background, 1 and 3 are foreground
            final_mask_cv = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
            
            # Apply morphological operations to clean up mask
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            logger.debug(f"Fallback: Applying morphological closing and opening with kernel size {kernel_size}.")
            final_mask_cv = cv2.morphologyEx(final_mask_cv, cv2.MORPH_CLOSE, kernel, iterations=2)
            final_mask_cv = cv2.morphologyEx(final_mask_cv, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Convert mask to PIL
            mask_pil = Image.fromarray(final_mask_cv * 255, mode='L')
            
            # Apply mask to the original image (preserving original mode if not RGB for RGBA conversion)
            if original_mode != "RGBA":
                image_rgba = image.convert("RGBA") # Use original image for conversion
            else:
                image_rgba = image.copy() # If already RGBA, copy to avoid modifying original

            image_rgba.putalpha(mask_pil)
            logger.info("Fallback background removal successful.")
            
            return (image_rgba, mask_pil) if return_mask else image_rgba

        except cv2.error as e: # Specific OpenCV errors
            logger.error(f"OpenCV error during fallback background removal: {e}", exc_info=True)
        except Exception as e: # Other unexpected errors
            logger.error(f"Unexpected error during fallback background removal: {e}", exc_info=True)

        # If any error occurs in fallback, return original image with an alpha channel
        # and a fully opaque dummy mask if requested.
        logger.warning("Fallback background removal failed. Returning original image with alpha (if possible).")
        try:
            if image.mode != "RGBA":
                image_rgba_fail = image.convert("RGBA")
            else:
                image_rgba_fail = image.copy()
            dummy_mask_fail = Image.new('L', image.size, 255) # Opaque mask
            return (image_rgba_fail, dummy_mask_fail) if return_mask else image_rgba_fail
        except Exception as final_fallback_e:
            logger.critical(f"Critical error in fallback's own error handling: {final_fallback_e}", exc_info=True)
            raise final_fallback_e # Or return None or a placeholder
    
    def segment_objects(
        self, 
        image: Image.Image, 
        points: Optional[List[Tuple[int, int]]] = None # Points are not used yet
    ) -> List[Image.Image]:
        """
        Segment multiple objects in the image.
        Currently, this is a basic implementation that returns the primary object mask.
        It could be extended to use input points for interactive segmentation with SAM2.
        """
        logger.info(f"Request to segment objects in image of size {image.size}. Points: {points}")
        if not self.model_loaded: # Ensure model loading is attempted
            logger.info("SAM2 model not yet loaded for segment_objects. Calling load_model().")
            self.load_model()
        
        # This basic version just returns the mask from background removal.
        # A more advanced version would use 'points' with SAM2 for interactive segmentation.
        try:
            # Use the main remove_background logic which already chooses SAM2 or fallback
            result_image, mask_image = self.remove_background(image, return_mask=True)
            logger.info("Object segmentation (primary object) successful using remove_background logic.")
            return [mask_image]
        except Exception as e:
            logger.error(f"Error during segment_objects: {e}", exc_info=True)
            # Fallback: return a list with a fully opaque mask
            logger.warning("segment_objects failed, returning a fully opaque mask.")
            return [Image.new('L', image.size, 255)]
    
    def unload_model(self):
        """Unload model to free memory."""
        logger.info(f"Attempting to unload {self.MODEL_NAME} model...")
        if self.model is not None or self.processor is not None:
            try:
                if self.model is not None and self.device == "cuda" and hasattr(self.model, 'to'):
                    try:
                        self.model.to('cpu')
                        logger.debug(f"{self.MODEL_NAME} model moved to CPU before unloading.")
                    except Exception as e_cpu:
                        logger.warning(f"Could not move {self.MODEL_NAME} model to CPU before unloading: {e_cpu}", exc_info=True)

                del self.model
                del self.processor
                self.model = None
                self.processor = None

                if torch.cuda.is_available():
                    logger.debug(f"Clearing CUDA cache after unloading {self.MODEL_NAME} model.")
                    torch.cuda.empty_cache()

                self.model_loaded = False # Set after successful unload
                logger.info(f"{self.MODEL_NAME} model unloaded successfully.")
            except Exception as e:
                logger.error(f"Error during {self.MODEL_NAME} model unloading: {e}", exc_info=True)
                self.model = None # Ensure attributes are cleared
                self.processor = None
                self.model_loaded = False
        else:
            logger.info(f"{self.MODEL_NAME} model and processor were already None or not loaded. No unload action needed.")

        # Final check to ensure model_loaded is consistent
        if self.model is None and self.processor is None:
            self.model_loaded = False