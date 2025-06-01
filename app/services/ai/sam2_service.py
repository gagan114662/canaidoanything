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
    logging.warning("SAM2 not available. Install sam2 package for background removal.")

logger = logging.getLogger(__name__)

class SAM2Service:
    """Service for SAM 2 (Segment Anything Model 2) background removal and segmentation"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the SAM 2 model"""
        if not SAM2_AVAILABLE:
            logger.warning("SAM2 not available. Using fallback background removal.")
            self.model_loaded = True
            return
            
        try:
            logger.info("Loading SAM 2 model...")
            
            self.processor = Sam2Processor.from_pretrained(
                settings.SAM2_MODEL_PATH,
                use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None
            )
            
            self.model = Sam2Model.from_pretrained(
                settings.SAM2_MODEL_PATH,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None
            )
            
            self.model = self.model.to(self.device)
            self.model_loaded = True
            logger.info("SAM 2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {str(e)}")
            # Fallback to basic background removal
            self.model_loaded = True
            logger.info("Using fallback background removal method")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded
    
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
        if not self.model_loaded:
            self.load_model()
        
        try:
            if SAM2_AVAILABLE and self.model is not None:
                return self._sam2_background_removal(image, return_mask)
            else:
                return self._fallback_background_removal(image, return_mask)
                
        except Exception as e:
            logger.error(f"Error in background removal: {str(e)}")
            # Return original image with alpha channel if removal fails
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            return (image, image) if return_mask else image
    
    def _sam2_background_removal(
        self, 
        image: Image.Image, 
        return_mask: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """Background removal using SAM 2"""
        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Process image for SAM2
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate mask (assuming we want the main object)
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get the best mask (you may need to adjust this based on SAM2 API)
            masks = outputs.pred_masks.cpu().numpy()
            
            # Select the largest mask (assuming it's the main object)
            if len(masks.shape) > 2:
                # Take the mask with highest confidence or largest area
                mask = masks[0, 0] if masks.shape[0] > 0 else masks[0]
            else:
                mask = masks
            
            # Convert mask to PIL Image
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            mask_pil = mask_pil.resize(image.size, Image.Resampling.LANCZOS)
            
            # Apply mask to create transparent background
            image_rgba = image.convert("RGBA")
            image_rgba.putalpha(mask_pil)
            
            return (image_rgba, mask_pil) if return_mask else image_rgba
            
        except Exception as e:
            logger.error(f"SAM2 processing failed: {str(e)}")
            return self._fallback_background_removal(image, return_mask)
    
    def _fallback_background_removal(
        self, 
        image: Image.Image, 
        return_mask: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """Fallback background removal using traditional computer vision"""
        try:
            # Convert PIL to OpenCV
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create mask using GrabCut algorithm
            mask = np.zeros(img_cv.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Define rectangle around the object (assuming center object)
            height, width = img_cv.shape[:2]
            rect = (int(width*0.1), int(height*0.1), 
                   int(width*0.8), int(height*0.8))
            
            # Apply GrabCut
            cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create final mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            # Convert mask to PIL
            mask_pil = Image.fromarray(mask2 * 255, mode='L')
            
            # Apply mask to image
            image_rgba = image.convert("RGBA")
            image_rgba.putalpha(mask_pil)
            
            return (image_rgba, mask_pil) if return_mask else image_rgba
            
        except Exception as e:
            logger.error(f"Fallback background removal failed: {str(e)}")
            # Return original image with alpha channel
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            return (image, image) if return_mask else image
    
    def segment_objects(
        self, 
        image: Image.Image, 
        points: Optional[List[Tuple[int, int]]] = None
    ) -> List[Image.Image]:
        """
        Segment multiple objects in the image
        
        Args:
            image: Input image
            points: Optional list of (x, y) points to guide segmentation
            
        Returns:
            List of segmented object masks
        """
        if not self.model_loaded:
            self.load_model()
        
        # For now, return single mask (can be extended for multi-object segmentation)
        mask_result = self.remove_background(image, return_mask=True)
        if isinstance(mask_result, tuple):
            return [mask_result[1]]
        else:
            # Create a simple mask
            return [Image.new('L', image.size, 255)]
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SAM2 model unloaded")