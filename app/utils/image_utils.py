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