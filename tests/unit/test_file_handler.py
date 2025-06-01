import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import shutil

from fastapi import UploadFile, HTTPException
from app.utils.file_handler import validate_file, save_upload_file, get_file_info
from app.core.config import settings # To get MAX_FILE_SIZE and ALLOWED_EXTENSIONS

@pytest.fixture
def mock_upload_file():
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_image.jpg"
    mock_file.content_type = "image/jpeg"
    # Simulate file size by mocking seek and tell if necessary for validate_file,
    # or by directly setting a 'size' attribute if your validate_file inspects that.
    # For this example, let's assume validate_file might use a 'size' like attribute
    # or we can mock the file's seek/tell for a more realistic size check.

    # To simulate size for `validate_file` if it reads the file:
    # mock_file.file.seek.return_value = 0
    # mock_file.file.tell.side_effect = [settings.MAX_FILE_SIZE - 1, 0] # Valid size then reset for read

    # Simpler: If validate_file directly accesses a 'size' attribute if available on UploadFile
    # (FastAPI's UploadFile has a 'size' attribute if SpooledTemporaryFile is used)
    # For testing, we can add it to the mock.
    mock_file.size = settings.MAX_FILE_SIZE -100 # Valid size
    return mock_file

# --- Tests for validate_file ---

def test_validate_file_valid(mock_upload_file):
    assert validate_file(mock_upload_file) is True

def test_validate_file_invalid_extension(mock_upload_file, monkeypatch):
    monkeypatch.setattr(settings, 'ALLOWED_EXTENSIONS', ["png", "gif"])
    mock_upload_file.filename = "test_image.jpg" # jpg is not in new allowed list

    with pytest.raises(HTTPException) as exc_info:
        validate_file(mock_upload_file)
    assert exc_info.value.status_code == 400
    assert "Invalid file extension" in exc_info.value.detail

def test_validate_file_invalid_content_type(mock_upload_file, monkeypatch):
    monkeypatch.setattr(settings, 'ALLOWED_EXTENSIONS', ["jpg", "jpeg", "png"]) # Ensure jpg is allowed for this test
    mock_upload_file.content_type = "text/plain"

    with pytest.raises(HTTPException) as exc_info:
        validate_file(mock_upload_file)
    assert exc_info.value.status_code == 400
    assert "Invalid content type" in exc_info.value.detail

def test_validate_file_too_large(mock_upload_file):
    mock_upload_file.size = settings.MAX_FILE_SIZE + 100 # Exceeds max size

    with pytest.raises(HTTPException) as exc_info:
        validate_file(mock_upload_file)
    assert exc_info.value.status_code == 413 # Payload Too Large
    assert "File size exceeds limit" in exc_info.value.detail

def test_validate_file_empty_filename(mock_upload_file):
    mock_upload_file.filename = ""
    with pytest.raises(HTTPException) as exc_info:
        validate_file(mock_upload_file)
    assert exc_info.value.status_code == 400
    assert "Invalid filename" in exc_info.value.detail


# --- Tests for save_upload_file ---

@pytest.mark.asyncio
async def test_save_upload_file_success(mock_upload_file, tmp_path, monkeypatch):
    # tmp_path is a pytest fixture providing a temporary directory path (Path object)
    # Mock settings.UPLOAD_DIR to use the temporary path
    monkeypatch.setattr(settings, 'UPLOAD_DIR', str(tmp_path))

    task_id = "test_task_123"

    # Mock aiofiles.open for async file operations
    mock_aio_open = mock_open()
    # When aiofiles.open is called as a context manager, __aenter__ should return a mock file handle
    # and that handle needs an async write method.
    mock_async_file_handle = MagicMock()
    mock_async_file_handle.write = MagicMock(return_value=None) # Simulate async write

    # Patch 'aiofiles.open' in the 'app.utils.file_handler' module
    with patch('app.utils.file_handler.aiofiles.open', mock_aio_open) as patched_aio_open:
        # Configure the __aenter__ method of the mock returned by aiofiles.open()
        # to return an object that has an async write method.
        async def dummy_aenter(): return mock_async_file_handle
        patched_aio_open.return_value.__aenter__ = dummy_aenter

        # Mock shutil.copyfileobj - though with async save, this might not be directly used
        # if we're reading and writing chunk by chunk.
        # For this test, assuming save_upload_file uses file.read() and async write.
        mock_upload_file.read.return_value = b"file content"
        mock_upload_file.seek.return_value = None # Mock seek if called

        saved_path = await save_upload_file(mock_upload_file, task_id)

        # Construct expected path
        filename = mock_upload_file.filename
        expected_dir = tmp_path / task_id
        expected_path = expected_dir / filename

        assert saved_path == expected_path
        assert expected_dir.exists() # Check if directory was created

        # Check aiofiles.open was called correctly
        patched_aio_open.assert_called_once_with(expected_path, "wb")
        # Check that the async file handle's write was called
        mock_async_file_handle.write.assert_called_once_with(b"file content")
        mock_upload_file.read.assert_called_once()


@pytest.mark.asyncio
async def test_save_upload_file_io_error(mock_upload_file, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(settings, 'UPLOAD_DIR', str(tmp_path))
    task_id = "test_io_error"

    # Mock aiofiles.open to raise IOError on write
    mock_aio_open = mock_open()
    mock_async_file_handle = MagicMock()
    mock_async_file_handle.write.side_effect = IOError("Disk full")

    with patch('app.utils.file_handler.aiofiles.open', mock_aio_open) as patched_aio_open:
        async def dummy_aenter(): return mock_async_file_handle
        patched_aio_open.return_value.__aenter__ = dummy_aenter

        mock_upload_file.read.return_value = b"file content"

        with pytest.raises(HTTPException) as exc_info:
            await save_upload_file(mock_upload_file, task_id)

        assert exc_info.value.status_code == 500
        assert "Could not save file" in exc_info.value.detail
        # Check logger was called (if file_handler.py has logging for this)
        # Example: assert "Disk full" in caplog.text

# --- Tests for get_file_info ---

def test_get_file_info(mock_upload_file):
    mock_upload_file.filename = "example.PNG"
    mock_upload_file.size = 1024 * 500 # 500KB

    info = get_file_info(mock_upload_file)

    assert info["filename"] == "example.PNG"
    assert info["content_type"] == "image/jpeg" # from fixture
    assert info["size_kb"] == 500.0
    assert info["extension"] == "png" # Should be lowercased

def test_get_file_info_no_extension(mock_upload_file):
    mock_upload_file.filename = "file_without_extension"
    info = get_file_info(mock_upload_file)
    assert info["extension"] == ""
    assert info["filename"] == "file_without_extension"

def test_get_file_info_empty_filename(mock_upload_file):
    mock_upload_file.filename = ""
    info = get_file_info(mock_upload_file)
    assert info["extension"] == ""
    assert info["filename"] == ""

# TODO: Add tests for image_utils.py
# - load_image (mock Image.open, test FileNotFoundError, generic Exception)
# - save_image (mock Image object's save method, test IOError, generic Exception)
# - resize_image (test resizing logic, aspect ratio preservation if applicable)
# - enhance_image (mock PIL.ImageEnhance operations, check factors are applied)
# - convert_to_rgb, convert_to_rgba (check mode changes)
# - create_image_grid (more complex, might need actual small images or mocked ones)
