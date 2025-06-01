import pytest
from unittest.mock import patch, MagicMock
from PIL import Image, UnidentifiedImageError # For UnidentifiedImageError
import numpy as np

from app.utils.image_utils import (
    load_image,
    save_image,
    resize_image,
    enhance_image,
    convert_to_rgb,
    convert_to_rgba,
    # create_image_grid # Testing this is more complex, might skip if time is short
)
# Mock the logger used in image_utils to prevent actual logging during tests
@pytest.fixture(autouse=True)
def mock_image_utils_logger():
    with patch('app.utils.image_utils.logger') as mock_log:
        yield mock_log

# --- Tests for load_image ---

@patch('app.utils.image_utils.Image.open')
def test_load_image_success(mock_image_open, mock_image_utils_logger):
    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    filepath = "fake/path/image.jpg"
    img = load_image(filepath)

    mock_image_open.assert_called_once_with(filepath)
    assert img == mock_img_instance
    mock_image_utils_logger.info.assert_any_call(f"Image loaded successfully from {filepath}")

@patch('app.utils.image_utils.Image.open', side_effect=FileNotFoundError("File not found"))
def test_load_image_file_not_found(mock_image_open, mock_image_utils_logger):
    filepath = "fake/nonexistent/image.jpg"
    img = load_image(filepath)

    assert img is None
    mock_image_utils_logger.error.assert_called_once_with(
        f"Error loading image from {filepath}: File not found", exc_info=True
    )

@patch('app.utils.image_utils.Image.open', side_effect=UnidentifiedImageError("Cannot identify image file"))
def test_load_image_unidentified_image_error(mock_image_open, mock_image_utils_logger):
    filepath = "fake/corrupted/image.jpg"
    img = load_image(filepath)

    assert img is None
    mock_image_utils_logger.error.assert_called_once_with(
        f"Error loading image from {filepath}: Cannot identify image file", exc_info=True
    )

@patch('app.utils.image_utils.Image.open', side_effect=Exception("Generic error"))
def test_load_image_generic_exception(mock_image_open, mock_image_utils_logger):
    filepath = "fake/error/image.jpg"
    img = load_image(filepath)

    assert img is None
    mock_image_utils_logger.error.assert_called_once_with(
        f"Unexpected error loading image from {filepath}: Generic error", exc_info=True
    )

# --- Tests for save_image ---

def test_save_image_success(tmp_path, mock_image_utils_logger):
    mock_img = MagicMock(spec=Image.Image)
    filepath = tmp_path / "test_save.jpg"

    result = save_image(mock_img, str(filepath))

    assert result is True
    mock_img.save.assert_called_once_with(str(filepath))
    mock_image_utils_logger.info.assert_any_call(f"Image saved successfully to {str(filepath)}")

def test_save_image_io_error(tmp_path, mock_image_utils_logger):
    mock_img = MagicMock(spec=Image.Image)
    mock_img.save.side_effect = IOError("Disk full")
    filepath = tmp_path / "test_io_error.jpg"

    result = save_image(mock_img, str(filepath))

    assert result is False
    mock_image_utils_logger.error.assert_called_once_with(
        f"IOError saving image to {str(filepath)}: Disk full", exc_info=True
    )

def test_save_image_generic_exception(tmp_path, mock_image_utils_logger):
    mock_img = MagicMock(spec=Image.Image)
    mock_img.save.side_effect = Exception("Generic save error")
    filepath = tmp_path / "test_generic_error.jpg"

    result = save_image(mock_img, str(filepath))

    assert result is False
    mock_image_utils_logger.error.assert_called_once_with(
        f"Unexpected error saving image to {str(filepath)}: Generic save error", exc_info=True
    )

# --- Tests for resize_image ---

def test_resize_image_larger_than_max(mock_image_utils_logger):
    mock_img = MagicMock(spec=Image.Image)
    mock_img.size = (2000, 1000) # width, height
    max_dimension = 500

    # Mock the resize method on the image instance
    resized_mock_img = MagicMock(spec=Image.Image)
    resized_mock_img.size = (500, 250) # Expected size
    mock_img.resize.return_value = resized_mock_img

    result_img = resize_image(mock_img, max_dimension)

    mock_img.resize.assert_called_once_with((500, 250), Image.Resampling.LANCZOS)
    assert result_img == resized_mock_img
    mock_image_utils_logger.info.assert_any_call(f"Resized image from (2000, 1000) to (500, 250) with max_dimension {max_dimension}")

def test_resize_image_smaller_than_max(mock_image_utils_logger):
    mock_img = MagicMock(spec=Image.Image)
    mock_img.size = (400, 300)
    max_dimension = 500

    result_img = resize_image(mock_img, max_dimension)

    mock_img.resize.assert_not_called() # Should not be called if smaller
    assert result_img == mock_img # Should return original image
    mock_image_utils_logger.info.assert_any_call("Image size (400, 300) is within max_dimension 500. No resize needed.")

def test_resize_image_zero_dimension_error(mock_image_utils_logger):
    mock_img = MagicMock(spec=Image.Image)
    mock_img.size = (0, 100) # Invalid size
    max_dimension = 500

    with pytest.raises(ValueError): # Or check if it returns original, depending on desired handling
        resize_image(mock_img, max_dimension)
    # If it's meant to log an error and return original:
    # result_img = resize_image(mock_img, max_dimension)
    # assert result_img == mock_img
    # mock_image_utils_logger.error.assert_called_once()


# --- Tests for convert_to_rgb ---
def test_convert_to_rgb_already_rgb():
    mock_img = MagicMock(spec=Image.Image)
    mock_img.mode = "RGB"
    assert convert_to_rgb(mock_img) == mock_img
    mock_img.convert.assert_not_called()

def test_convert_to_rgb_needs_conversion():
    mock_img = MagicMock(spec=Image.Image)
    mock_img.mode = "RGBA"
    mock_converted_img = MagicMock(spec=Image.Image)
    mock_img.convert.return_value = mock_converted_img

    assert convert_to_rgb(mock_img) == mock_converted_img
    mock_img.convert.assert_called_once_with("RGB")

# --- Tests for convert_to_rgba ---
def test_convert_to_rgba_already_rgba():
    mock_img = MagicMock(spec=Image.Image)
    mock_img.mode = "RGBA"
    assert convert_to_rgba(mock_img) == mock_img
    mock_img.convert.assert_not_called()

def test_convert_to_rgba_needs_conversion():
    mock_img = MagicMock(spec=Image.Image)
    mock_img.mode = "RGB"
    mock_converted_img = MagicMock(spec=Image.Image)
    mock_img.convert.return_value = mock_converted_img

    assert convert_to_rgba(mock_img) == mock_converted_img
    mock_img.convert.assert_called_once_with("RGBA")

# --- Tests for enhance_image (basic check) ---
@patch('app.utils.image_utils.ImageEnhance.Brightness')
@patch('app.utils.image_utils.ImageEnhance.Contrast')
@patch('app.utils.image_utils.ImageEnhance.Sharpness')
def test_enhance_image(mock_sharpness_enh, mock_contrast_enh, mock_brightness_enh, mock_image_utils_logger):
    mock_img = MagicMock(spec=Image.Image)

    # Mock enhancer objects and their enhance methods
    mock_b_inst = MagicMock(); mock_brightness_enh.return_value = mock_b_inst
    mock_c_inst = MagicMock(); mock_contrast_enh.return_value = mock_c_inst
    mock_s_inst = MagicMock(); mock_sharpness_enh.return_value = mock_s_inst

    mock_b_inst.enhance.return_value = mock_img # Simulate chain
    mock_c_inst.enhance.return_value = mock_img
    mock_s_inst.enhance.return_value = mock_img

    result = enhance_image(mock_img, brightness=1.1, contrast=1.2, sharpness=1.3)

    mock_brightness_enh.assert_called_once_with(mock_img)
    mock_b_inst.enhance.assert_called_once_with(1.1)

    mock_contrast_enh.assert_called_once_with(mock_img) # Called with result of brightness
    mock_c_inst.enhance.assert_called_once_with(1.2)

    mock_sharpness_enh.assert_called_once_with(mock_img) # Called with result of contrast
    mock_s_inst.enhance.assert_called_once_with(1.3)

    assert result == mock_img # Final result of chained mocks
    mock_image_utils_logger.debug.assert_any_call("Applying brightness enhancement with factor 1.1")
    mock_image_utils_logger.debug.assert_any_call("Applying contrast enhancement with factor 1.2")
    mock_image_utils_logger.debug.assert_any_call("Applying sharpness enhancement with factor 1.3")

# Note: Testing create_image_grid would be more involved, requiring multiple (mock) images
# and potentially checks on the final grid dimensions or how `paste` was called.
# For this exercise, focusing on the more common utilities.
