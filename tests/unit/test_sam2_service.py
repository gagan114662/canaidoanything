import pytest
from unittest.mock import patch, MagicMock
import torch

# Assuming SAM2_AVAILABLE might be True or False depending on test environment setup
# For specific tests, we'll patch it.
from app.services.ai.sam2_service import SAM2Service, SAM2_AVAILABLE as ACTUAL_SAM2_AVAILABLE
from app.core.config import settings
from app.core.exceptions import ModelLoadError

# Mock the logger used in SAM2Service to prevent actual logging during tests
@pytest.fixture(autouse=True)
def mock_sam2_service_logger():
    with patch('app.services.ai.sam2_service.logger') as mock_log:
        yield mock_log

@pytest.fixture
def sam2_service_instance(monkeypatch):
    # Test with SAM2_AVAILABLE = True by default for most loading tests
    monkeypatch.setattr('app.services.ai.sam2_service.SAM2_AVAILABLE', True)
    service = SAM2Service()
    # Reset device based on actual test environment for some tests
    service.device = "cuda" if torch.cuda.is_available() else "cpu"
    return service

@patch('app.services.ai.sam2_service.Sam2Processor')
@patch('app.services.ai.sam2_service.Sam2Model')
def test_load_model_success(mock_sam2model_cls, mock_sam2processor_cls, sam2_service_instance, mock_sam2_service_logger):
    mock_processor_instance = MagicMock()
    mock_model_instance = MagicMock()
    mock_sam2processor_cls.from_pretrained.return_value = mock_processor_instance
    mock_sam2model_cls.from_pretrained.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance

    sam2_service_instance.load_model()

    mock_sam2processor_cls.from_pretrained.assert_called_once_with(
        settings.SAM2_MODEL_PATH,
        use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None
    )
    mock_sam2model_cls.from_pretrained.assert_called_once_with(
        settings.SAM2_MODEL_PATH,
        torch_dtype=torch.float16 if sam2_service_instance.device == "cuda" else torch.float32,
        use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None
    )
    mock_model_instance.to.assert_called_once_with(sam2_service_instance.device)

    assert sam2_service_instance.model_loaded
    assert sam2_service_instance.processor == mock_processor_instance
    assert sam2_service_instance.model == mock_model_instance
    mock_sam2_service_logger.info.assert_any_call(
        f"{SAM2Service.MODEL_NAME} model loaded successfully from {settings.SAM2_MODEL_PATH} and moved to {sam2_service_instance.device}."
    )

def test_load_model_sam2_not_available(monkeypatch, mock_sam2_service_logger):
    monkeypatch.setattr('app.services.ai.sam2_service.SAM2_AVAILABLE', False)
    service = SAM2Service() # Re-initialize to pick up patched SAM2_AVAILABLE
    service.load_model()

    assert not service.model_loaded # model_loaded should be False if primary model cannot be loaded
    mock_sam2_service_logger.warning.assert_any_call(
        f"{SAM2Service.MODEL_NAME} HuggingFace package not available. Cannot load model. Service will use fallbacks."
    )

@patch('app.services.ai.sam2_service.Sam2Processor')
def test_load_model_raises_model_load_error_on_processor_failure(mock_sam2processor_cls, sam2_service_instance, mock_sam2_service_logger):
    mock_sam2processor_cls.from_pretrained.side_effect = RuntimeError("Failed to load processor")

    with pytest.raises(ModelLoadError) as exc_info:
        sam2_service_instance.load_model()

    assert exc_info.value.model_name == SAM2Service.MODEL_NAME
    assert isinstance(exc_info.value.original_exception, RuntimeError)
    assert not sam2_service_instance.model_loaded
    mock_sam2_service_logger.error.assert_any_call(
        f"Runtime error loading {SAM2Service.MODEL_NAME} model from {settings.SAM2_MODEL_PATH} (e.g. HF Hub issue, CUDA OOM): Failed to load processor",
        exc_info=True
    )

@patch('app.services.ai.sam2_service.Sam2Processor') # Mock processor to succeed
@patch('app.services.ai.sam2_service.Sam2Model')
def test_load_model_raises_model_load_error_on_model_failure(mock_sam2model_cls, mock_sam2processor_cls, sam2_service_instance, mock_sam2_service_logger):
    mock_sam2processor_cls.from_pretrained.return_value = MagicMock() # Processor loads fine
    mock_sam2model_cls.from_pretrained.side_effect = FileNotFoundError("Model weights not found")

    with pytest.raises(ModelLoadError) as exc_info:
        sam2_service_instance.load_model()

    assert exc_info.value.model_name == SAM2Service.MODEL_NAME
    assert isinstance(exc_info.value.original_exception, FileNotFoundError)
    assert not sam2_service_instance.model_loaded
    mock_sam2_service_logger.error.assert_any_call(
        f"{SAM2Service.MODEL_NAME} model files not found at {settings.SAM2_MODEL_PATH}: Model weights not found",
        exc_info=True
    )

def test_is_model_loaded(sam2_service_instance, monkeypatch):
    assert not sam2_service_instance.is_model_loaded() # Initially false

    # Simulate successful load
    sam2_service_instance.model = MagicMock()
    sam2_service_instance.processor = MagicMock()
    sam2_service_instance.model_loaded = True
    # SAM2_AVAILABLE is True due to fixture setup
    assert sam2_service_instance.is_model_loaded()

    # Simulate SAM2_AVAILABLE is False
    monkeypatch.setattr('app.services.ai.sam2_service.SAM2_AVAILABLE', False)
    assert not sam2_service_instance.is_model_loaded() # Should be false even if other flags are true

@patch('torch.cuda.is_available')
@patch('torch.cuda.empty_cache')
def test_unload_model_cuda(mock_empty_cache, mock_is_available, sam2_service_instance, mock_sam2_service_logger):
    mock_is_available.return_value = True
    sam2_service_instance.device = "cuda"
    sam2_service_instance.model = MagicMock()
    sam2_service_instance.processor = MagicMock()
    sam2_service_instance.model_loaded = True

    sam2_service_instance.unload_model()

    assert sam2_service_instance.model is None
    assert sam2_service_instance.processor is None
    assert not sam2_service_instance.model_loaded
    mock_empty_cache.assert_called_once()
    mock_sam2_service_logger.info.assert_any_call(f"{SAM2Service.MODEL_NAME} model unloaded successfully.")

@patch('app.services.ai.sam2_service.SAM2Service._fallback_background_removal')
def test_remove_background_uses_fallback_when_sam2_unavailable(mock_fallback_method, monkeypatch, mock_sam2_service_logger):
    monkeypatch.setattr('app.services.ai.sam2_service.SAM2_AVAILABLE', False)
    # Re-initialize to ensure SAM2_AVAILABLE is False during __init__ and subsequent calls
    service = SAM2Service()

    mock_image = MagicMock(spec=Image.Image)
    mock_image.mode = "RGB"
    mock_image.size = (100,100)

    service.remove_background(mock_image)

    mock_fallback_method.assert_called_once_with(mock_image, False)
    mock_sam2_service_logger.warning.assert_any_call(
        f"{SAM2Service.MODEL_NAME} HuggingFace package not available. Cannot load model. Service will use fallbacks."
    )
    # Check for the warning inside remove_background
    mock_sam2_service_logger.warning.assert_any_call(
         f"{SAM2Service.MODEL_NAME} model not available or not loaded. Attempting to load/re-load for remove_background."
    )


@patch('app.services.ai.sam2_service.SAM2Service._fallback_background_removal')
@patch('app.services.ai.sam2_service.Sam2Processor')
@patch('app.services.ai.sam2_service.Sam2Model')
def test_remove_background_uses_fallback_on_model_load_failure(
    mock_sam2model_cls, mock_sam2processor_cls, mock_fallback_method,
    sam2_service_instance, mock_sam2_service_logger
):
    # Simulate model loading failing by making from_pretrained raise ModelLoadError
    mock_sam2processor_cls.from_pretrained.side_effect = ModelLoadError("SAM2", RuntimeError("Test Load Fail"), "path")

    mock_image = MagicMock(spec=Image.Image)
    mock_image.mode = "RGB"
    mock_image.size = (100,100)

    sam2_service_instance.remove_background(mock_image) # This will attempt to load_model internally

    mock_fallback_method.assert_called_once_with(mock_image, False)
    mock_sam2_service_logger.error.assert_any_call(
        f"{SAM2Service.MODEL_NAME} model failed to load on demand. Using fallback for remove_background."
    )

# Restore ACTUAL_SAM2_AVAILABLE for other test modules if they run in the same session.
# This is more of a safeguard, pytest typically isolates modules.
@pytest.fixture(scope="module", autouse=True)
def restore_sam2_available_after_tests(monkeypatch):
    yield
    monkeypatch.setattr('app.services.ai.sam2_service.SAM2_AVAILABLE', ACTUAL_SAM2_AVAILABLE)
