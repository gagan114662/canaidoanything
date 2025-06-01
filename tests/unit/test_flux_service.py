import pytest
from unittest.mock import patch, MagicMock
import torch

from app.services.ai.flux_service import FluxService
from app.core.config import settings
from app.core.exceptions import ModelLoadError

# Mock the logger used in FluxService to prevent actual logging during tests
@pytest.fixture(autouse=True)
def mock_service_logger():
    with patch('app.services.ai.flux_service.logger') as mock_log:
        yield mock_log

@pytest.fixture
def flux_service_instance(monkeypatch):
    # Ensure settings are somewhat controlled for tests if necessary
    # monkeypatch.setattr(settings, 'FLUX_MODEL_PATH', 'fake/flux-model')
    # monkeypatch.setattr(settings, 'HF_TOKEN', None)
    return FluxService()

def test_flux_service_initialization(flux_service_instance):
    assert not flux_service_instance.model_loaded
    assert flux_service_instance.pipeline is None
    # Device check depends on test environment's CUDA availability,
    # So we check it's either 'cuda' or 'cpu' as determined by torch.
    assert flux_service_instance.device in ["cuda", "cpu"]

@patch('app.services.ai.flux_service.FluxPipeline')
def test_load_model_success(mock_flux_pipeline_cls, flux_service_instance, mock_service_logger):
    mock_pipeline_instance = MagicMock()
    mock_flux_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance
    mock_pipeline_instance.to.return_value = mock_pipeline_instance # Ensure .to() returns the mock

    flux_service_instance.load_model()

    mock_flux_pipeline_cls.from_pretrained.assert_called_once_with(
        settings.FLUX_MODEL_PATH,
        torch_dtype=torch.float16 if flux_service_instance.device == "cuda" else torch.float32,
        use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None
    )
    mock_pipeline_instance.to.assert_called_once_with(flux_service_instance.device)
    if flux_service_instance.device == "cuda":
        mock_pipeline_instance.enable_model_cpu_offload.assert_called_once()
        mock_pipeline_instance.enable_attention_slicing.assert_called_once()

    assert flux_service_instance.model_loaded
    assert flux_service_instance.pipeline == mock_pipeline_instance
    mock_service_logger.info.assert_any_call(
        f"{FluxService.MODEL_NAME} model loaded successfully from {settings.FLUX_MODEL_PATH} on {flux_service_instance.device}."
    )

@patch('app.services.ai.flux_service.FluxPipeline')
def test_load_model_already_loaded(mock_flux_pipeline_cls, flux_service_instance, mock_service_logger):
    flux_service_instance.model_loaded = True # Simulate model already loaded
    flux_service_instance.pipeline = MagicMock() # Simulate pipeline exists

    flux_service_instance.load_model()

    mock_flux_pipeline_cls.from_pretrained.assert_not_called() # Should not attempt to load again
    mock_service_logger.info.assert_called_with(f"{FluxService.MODEL_NAME} model is already loaded.")

@patch('app.services.ai.flux_service.FluxPipeline')
def test_load_model_raises_file_not_found(mock_flux_pipeline_cls, flux_service_instance, mock_service_logger):
    mock_flux_pipeline_cls.from_pretrained.side_effect = FileNotFoundError("Model not found")

    with pytest.raises(ModelLoadError) as exc_info:
        flux_service_instance.load_model()

    assert exc_info.value.model_name == FluxService.MODEL_NAME
    assert isinstance(exc_info.value.original_exception, FileNotFoundError)
    assert not flux_service_instance.model_loaded
    mock_service_logger.error.assert_called_once()

@patch('app.services.ai.flux_service.FluxPipeline')
def test_load_model_raises_runtime_error(mock_flux_pipeline_cls, flux_service_instance, mock_service_logger):
    mock_flux_pipeline_cls.from_pretrained.side_effect = RuntimeError("CUDA OOM")

    with pytest.raises(ModelLoadError) as exc_info:
        flux_service_instance.load_model()

    assert exc_info.value.model_name == FluxService.MODEL_NAME
    assert isinstance(exc_info.value.original_exception, RuntimeError)
    assert not flux_service_instance.model_loaded
    mock_service_logger.error.assert_called_once()

@patch('app.services.ai.flux_service.FluxPipeline')
def test_load_model_raises_generic_exception(mock_flux_pipeline_cls, flux_service_instance, mock_service_logger):
    mock_flux_pipeline_cls.from_pretrained.side_effect = Exception("Some other error")

    with pytest.raises(ModelLoadError) as exc_info:
        flux_service_instance.load_model()

    assert exc_info.value.model_name == FluxService.MODEL_NAME
    assert isinstance(exc_info.value.original_exception, Exception)
    assert not flux_service_instance.model_loaded
    mock_service_logger.error.assert_called_once()

def test_is_model_loaded(flux_service_instance):
    assert not flux_service_instance.is_model_loaded() # Initially false
    flux_service_instance.model_loaded = True
    assert flux_service_instance.is_model_loaded()

@patch('torch.cuda.is_available')
@patch('torch.cuda.empty_cache')
def test_unload_model_cuda(mock_empty_cache, mock_is_available, flux_service_instance, mock_service_logger):
    mock_is_available.return_value = True
    flux_service_instance.device = "cuda" # Ensure device is cuda for this test
    flux_service_instance.pipeline = MagicMock()
    flux_service_instance.model_loaded = True

    flux_service_instance.unload_model()

    assert flux_service_instance.pipeline is None
    assert not flux_service_instance.model_loaded
    mock_empty_cache.assert_called_once()
    mock_service_logger.info.assert_any_call(f"{FluxService.MODEL_NAME} model unloaded successfully.")

@patch('torch.cuda.is_available')
@patch('torch.cuda.empty_cache')
def test_unload_model_cpu(mock_empty_cache, mock_is_available, flux_service_instance, mock_service_logger):
    mock_is_available.return_value = False # Simulate CPU only
    flux_service_instance.device = "cpu"   # Ensure device is cpu
    flux_service_instance.pipeline = MagicMock()
    flux_service_instance.model_loaded = True

    flux_service_instance.unload_model()

    assert flux_service_instance.pipeline is None
    assert not flux_service_instance.model_loaded
    mock_empty_cache.assert_not_called() # Should not be called on CPU
    mock_service_logger.info.assert_any_call(f"{FluxService.MODEL_NAME} model unloaded successfully.")

def test_unload_model_not_loaded(flux_service_instance, mock_service_logger):
    flux_service_instance.pipeline = None
    flux_service_instance.model_loaded = False

    flux_service_instance.unload_model()
    mock_service_logger.info.assert_any_call(f"{FluxService.MODEL_NAME} model was not loaded, no unload action needed.")
    assert not flux_service_instance.model_loaded # Should remain false
