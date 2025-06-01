import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Tuple, Optional, Union
import io
import base64

def load_image(image_path: str) -> Image.Image:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object
    """
    try:
        image = Image.open(image_path)
        # Convert to RGB if necessary
        if image.mode in ['RGBA', 'P']:
            # Create white background for transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")

def save_image(image: Image.Image, output_path: str, quality: int = 95) -> bool:
    """
    Save PIL Image to file
    
    Args:
        image: PIL Image object
        output_path: Path to save the image
        quality: JPEG quality (1-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure RGB mode for JPEG
        if image.mode in ['RGBA', 'P']:
            # Create white background for transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(output_path, 'JPEG', quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f"Failed to save image: {str(e)}")
        return False

def resize_image(image: Image.Image, max_size: int, maintain_aspect: bool = True) -> Image.Image:
    """
    Resize image to fit within max_size while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension (width or height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    try:
        if maintain_aspect:
            # Calculate new size maintaining aspect ratio
            width, height = image.size
            
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
            
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    
    except Exception as e:
        print(f"Failed to resize image: {str(e)}")
        return image

def crop_to_square(image: Image.Image) -> Image.Image:
    """
    Crop image to square aspect ratio (center crop)
    
    Args:
        image: PIL Image object
        
    Returns:
        Square PIL Image
    """
    try:
        width, height = image.size
        size = min(width, height)
        
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        
        return image.crop((left, top, right, bottom))
    
    except Exception as e:
        print(f"Failed to crop image: {str(e)}")
        return image

def enhance_image(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0
) -> Image.Image:
    """
    Enhance image with various adjustments
    
    Args:
        image: PIL Image object
        brightness: Brightness factor (1.0 = no change)
        contrast: Contrast factor (1.0 = no change)
        saturation: Saturation factor (1.0 = no change)
        sharpness: Sharpness factor (1.0 = no change)
        
    Returns:
        Enhanced PIL Image
    """
    try:
        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
        
        return image
    
    except Exception as e:
        print(f"Failed to enhance image: {str(e)}")
        return image

def apply_blur(image: Image.Image, radius: float = 1.0) -> Image.Image:
    """
    Apply Gaussian blur to image
    
    Args:
        image: PIL Image object
        radius: Blur radius
        
    Returns:
        Blurred PIL Image
    """
    try:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    except Exception as e:
        print(f"Failed to apply blur: {str(e)}")
        return image

def image_to_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """
    Convert PIL Image to base64 string
    
    Args:
        image: PIL Image object
        format: Image format (JPEG, PNG, etc.)
        
    Returns:
        Base64 encoded string
    """
    try:
        buffer = io.BytesIO()
        
        # Ensure RGB mode for JPEG
        if format.upper() == 'JPEG' and image.mode in ['RGBA', 'P']:
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        
        image.save(buffer, format=format, quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    except Exception as e:
        print(f"Failed to convert image to base64: {str(e)}")
        return ""

def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(img_data))
        return image.convert('RGB')
    
    except Exception as e:
        raise ValueError(f"Failed to convert base64 to image: {str(e)}")

def get_image_stats(image: Image.Image) -> dict:
    """
    Get statistics about an image
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image statistics
    """
    try:
        # Convert to numpy array for calculations
        img_array = np.array(image)
        
        stats = {
            'width': image.size[0],
            'height': image.size[1],
            'mode': image.mode,
            'channels': len(img_array.shape) if len(img_array.shape) > 2 else 1,
            'mean_brightness': np.mean(img_array),
            'std_brightness': np.std(img_array),
            'min_value': np.min(img_array),
            'max_value': np.max(img_array)
        }
        
        # Color statistics for RGB images
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            stats.update({
                'mean_red': np.mean(img_array[:, :, 0]),
                'mean_green': np.mean(img_array[:, :, 1]),
                'mean_blue': np.mean(img_array[:, :, 2])
            })
        
        return stats
    
    except Exception as e:
        return {'error': str(e)}

def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (256, 256)) -> Image.Image:
    """
    Create thumbnail of image
    
    Args:
        image: PIL Image object
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail PIL Image
    """
    try:
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail
    
    except Exception as e:
        print(f"Failed to create thumbnail: {str(e)}")
        return image

def pad_image_to_square(image: Image.Image, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Pad image to make it square
    
    Args:
        image: PIL Image object
        background_color: RGB color for padding
        
    Returns:
        Square PIL Image with padding
    """
    try:
        width, height = image.size
        size = max(width, height)
        
        # Create new square image with background color
        new_image = Image.new('RGB', (size, size), background_color)
        
        # Paste original image in center
        paste_x = (size - width) // 2
        paste_y = (size - height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image
    
    except Exception as e:
        print(f"Failed to pad image: {str(e)}")
        return image

# New imports for apply_lut
from PIL import ImageOps
import logging
logger = logging.getLogger(__name__)

def apply_lut(image: Image.Image, lut_path: str) -> Image.Image:
    logger.info(f"Attempting to apply LUT: {lut_path} to image.")
    # Placeholder: Actual LUT application logic would go here.
    # For now, let's simulate a simple color effect if LUT path is "mock_lut_path.cube"
    if "mock_lut_path.cube" in lut_path:
        try:
            # Example: Apply a simple inversion as a mock
            if image.mode == 'RGBA':
                # Separate alpha channel
                alpha = image.split()[-1]
                image_rgb = image.convert('RGB')

                image_rgb = ImageOps.invert(image_rgb)
                image_rgb.putalpha(alpha) # Reapply alpha channel
                logger.info(f"Applied mock LUT effect (inversion with alpha) for {lut_path}")
                return image_rgb
            else:
                # For RGB images or others, convert to RGB first then invert
                image_rgb = image.convert('RGB')
                image_rgb = ImageOps.invert(image_rgb)
                logger.info(f"Applied mock LUT effect (inversion) for {lut_path}")
                return image_rgb

        except Exception as e:
            logger.error(f"Mock LUT application error: {e}")
            return image # Return original on error

    logger.warning(f"LUT file {lut_path} not found or not a mock LUT. Returning original image.")
    return image

def ai_smart_sharpen(image: Image.Image, intensity: float = 0.5) -> Image.Image:
    logger.info(f"Attempting AI smart sharpening with intensity: {intensity}")
    # Placeholder: Actual AI sharpening logic.
    # Mock: Apply a standard sharpening filter for now, intensity might not be fully used by mock.
    try:
        # ImageFilter is already imported at the top of the file
        # from PIL import ImageFilter

        # Ensure intensity has some effect, even if basic
        # For UnsharpMask, radius affects the blurriness of the mask, percent is strength of effect
        # Let's map intensity (0-1) to radius (0.5-2.5) and percent (100-200)
        radius = 0.5 + (intensity * 2.0)
        percent = 100 + int(intensity * 100)
        threshold = 3 # Default threshold

        sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        logger.info(f"Applied mock AI smart sharpening (UnsharpMask radius={radius:.1f}, percent={percent}).")
        return sharpened_image
    except Exception as e:
        logger.error(f"Mock AI smart sharpening error: {e}")
        return image

def ai_smart_denoise(image: Image.Image, strength: float = 0.5) -> Image.Image:
    logger.info(f"Attempting AI smart denoising with strength: {strength}")
    # Placeholder: Actual AI denoising logic.
    # Mock: Apply a simple blur for now as a stand-in for denoising.
    try:
        # ImageFilter is already imported at the top of the file
        # from PIL import ImageFilter

        # Ensure strength has some effect
        # For GaussianBlur, radius is the blur radius.
        # Map strength (0-1) to radius (0-2)
        radius = strength * 2.0

        if radius == 0: # No denoising if strength is 0
            logger.info("Mock AI smart denoising strength is 0, returning original image.")
            return image

        denoised_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        logger.info(f"Applied mock AI smart denoising (GaussianBlur radius={radius:.1f}).")
        return denoised_image
    except Exception as e:
        logger.error(f"Mock AI smart denoising error: {e}")
        return image

def get_saliency_map(image: Image.Image) -> np.ndarray:
    logger.info("Attempting to generate saliency map.")
    # Placeholder: Actual AI saliency detection.
    # Mock: Create a gradient saliency map (e.g., brighter towards the center or a corner).
    try:
        img_gray = image.convert('L')
        width, height = img_gray.size
        saliency_array = np.zeros((height, width), dtype=np.uint8)
        # Create a mock map where the center is most salient
        for r_idx in range(height): # Renamed r to r_idx to avoid conflict if np.r_ is used
            for c_idx in range(width): # Renamed c to c_idx
                dist_to_center = np.sqrt(((r_idx - height/2)**2) + ((c_idx - width/2)**2))
                max_dist = np.sqrt(((height/2)**2) + ((width/2)**2))
                if max_dist == 0: # Avoid division by zero for 1x1 image or similar
                    saliency_value = 255
                else:
                    saliency_value = 255 * (1 - (dist_to_center / max_dist))
                saliency_array[r_idx, c_idx] = int(saliency_value)
        logger.info("Generated mock saliency map (center-biased).")
        return saliency_array
    except Exception as e:
        logger.error(f"Mock saliency map generation error: {e}")
        # Return a flat saliency map on error
        return np.full((image.height, image.width), 128, dtype=np.uint8)

def ai_enhance_fabric_texture(image: Image.Image, intensity: float = 0.5) -> Image.Image:
    logger.info(f"Attempting AI fabric texture enhancement with intensity: {intensity}")
    # Placeholder: Actual AI fabric texture enhancement.
    # Mock: Apply a combination of sharpening and contrast.
    try:
        # Sharpen
        radius_sharpen = 1 + intensity # Max radius 2 for UnsharpMask
        temp_image = image.filter(ImageFilter.UnsharpMask(radius=radius_sharpen, percent=150, threshold=3))
        # Increase contrast
        contrast_factor = 1.0 + (0.2 * intensity) # Max 1.2 contrast
        enhancer = ImageEnhance.Contrast(temp_image)
        enhanced_image = enhancer.enhance(contrast_factor)
        logger.info(f"Applied mock AI fabric texture enhancement (UnsharpMask radius={radius_sharpen:.1f}, Contrast factor={contrast_factor:.2f}).")
        return enhanced_image
    except Exception as e:
        logger.error(f"Mock AI fabric texture enhancement error: {e}")
        return image

def ai_enhance_facial_details(image: Image.Image, intensity: float = 0.5) -> Image.Image:
    logger.info(f"Attempting AI facial detail enhancement with intensity: {intensity}")
    # Placeholder: Actual AI facial detail enhancement (e.g., eyes, lips).
    # Mock: Apply a mild sharpening.
    try:
        # Sharpen slightly differently for faces
        radius_face = 0.5 + intensity * 0.5 # Max radius 1.0 for UnsharpMask
        percent_face = 110 + int(intensity * 40) # Max percent 150
        threshold_face = 2 + int(intensity) # Max threshold 3
        enhanced_image = image.filter(ImageFilter.UnsharpMask(radius=radius_face, percent=percent_face, threshold=threshold_face))
        logger.info(f"Applied mock AI facial detail enhancement (UnsharpMask radius={radius_face:.1f}, percent={percent_face}).")
        return enhanced_image
    except Exception as e:
        logger.error(f"Mock AI facial detail enhancement error: {e}")
        return image

def ai_enhance_accessory_details(image: Image.Image, intensity: float = 0.5) -> Image.Image:
    logger.info(f"Attempting AI accessory detail enhancement with intensity: {intensity}")
    # Placeholder: Actual AI accessory detail enhancement (e.g., jewelry).
    # Mock: Apply slight contrast and sharpness.
    try:
        contrast_factor = 1.0 + (0.1 * intensity) # Max 1.1 contrast
        enhancer = ImageEnhance.Contrast(image)
        temp_image = enhancer.enhance(contrast_factor)

        radius_accessory = 0.5 + intensity * 0.5 # Max radius 1.0
        percent_accessory = 120 + int(intensity * 30) # Max percent 150
        enhanced_image = temp_image.filter(ImageFilter.UnsharpMask(radius=radius_accessory, percent=percent_accessory, threshold=3))
        logger.info(f"Applied mock AI accessory detail enhancement (Contrast factor={contrast_factor:.2f}, UnsharpMask radius={radius_accessory:.1f}).")
        return enhanced_image
    except Exception as e:
        logger.error(f"Mock AI accessory detail enhancement error: {e}")
        return image