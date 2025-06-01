import pytest
from unittest.mock import patch, MagicMock
import torch
from PIL import Image

from app.services.ai.esrgan_service import ESRGANService, ESRGAN_AVAILABLE as ACTUAL_ESRGAN_AVAILABLE
from app.core.config import settings
from app.core.exceptions import ModelLoadError

@pytest.fixture(autouse=True)
def mock_esrgan_service_logger():
    with patch('app.services.ai.esrgan_service.logger') as mock_log:
        yield mock_log

@pytest.fixture
def esrgan_service_instance(monkeypatch):
    monkeypatch.setattr('app.services.ai.esrgan_service.ESRGAN_AVAILABLE', True)
    service = ESRGANService()
    service.device = "cuda" if torch.cuda.is_available() else "cpu"
    return service

# Mock RRDBNet and RealESRGANer for all tests in this module
@pytest.fixture(autouse=True)
def mock_realesrgan_dependencies():
    with patch('app.services.ai.esrgan_service.RRDBNet') as mock_rrdbnet, \
         patch('app.services.ai.esrgan_service.RealESRGANer') as mock_realesrganer:
        # Configure default return values if needed for most tests
        mock_rrdbnet.return_value = MagicMock()
        mock_realesrganer.return_value = MagicMock()
        yield mock_rrdbnet, mock_realesrganer


def test_esrgan_service_initialization(esrgan_service_instance):
    assert not esrgan_service_instance.model_loaded
    assert esrgan_service_instance.upsampler is None
    assert esrgan_service_instance.device in ["cuda", "cpu"]

def test_load_model_success(esrgan_service_instance, mock_realesrgan_dependencies, mock_esrgan_service_logger):
    mock_rrdbnet_cls, mock_realesrganer_cls = mock_realesrgan_dependencies
    mock_rrdbnet_instance = mock_rrdbnet_cls.return_value
    mock_realesrganer_instance = mock_realesrganer_cls.return_value

    esrgan_service_instance.load_model()

    mock_rrdbnet_cls.assert_called_once_with(
        num_in_ch=settings.ESRGAN_RRDB_NUM_IN_CH,
        num_out_ch=settings.ESRGAN_RRDB_NUM_OUT_CH,
        num_feat=settings.ESRGAN_RRDB_NUM_FEAT,
        num_block=settings.ESRGAN_RRDB_NUM_BLOCK,
        num_grow_ch=settings.ESRGAN_RRDB_NUM_GROW_CH,
        scale=settings.ESRGAN_RRDB_SCALE
    )

    expected_gpu_id = None
    if esrgan_service_instance.device.startswith("cuda"):
        expected_gpu_id = int(esrgan_service_instance.device.split(":")[1]) if ":" in esrgan_service_instance.device else 0

    mock_realesrganer_cls.assert_called_once_with(
        scale=settings.ESRGAN_DEFAULT_SCALE_PARAM,
        model_path=settings.ESRGAN_MODEL_PATH,
        model=mock_rrdbnet_instance,
        tile=settings.ESRGAN_TILE_SIZE,
        tile_pad=settings.ESRGAN_TILE_PAD,
        pre_pad=settings.ESRGAN_PRE_PAD,
        half=(esrgan_service_instance.device != "cpu"),
        gpu_id=expected_gpu_id
    )

    assert esrgan_service_instance.model_loaded
    assert esrgan_service_instance.upsampler == mock_realesrganer_instance
    mock_esrgan_service_logger.info.assert_any_call(
        f"{ESRGANService.MODEL_NAME} model loaded successfully from {settings.ESRGAN_MODEL_PATH} on {esrgan_service_instance.device} (RealESRGANer gpu_id: {expected_gpu_id})."
    )

def test_load_model_esrgan_not_available(monkeypatch, mock_esrgan_service_logger):
    monkeypatch.setattr('app.services.ai.esrgan_service.ESRGAN_AVAILABLE', False)
    service = ESRGANService() # Re-initialize
    service.load_model()

    # model_loaded might be True if fallback is considered a "loaded" state by __init__ or load_model
    # but is_model_loaded() should be False for the actual ESRGAN model
    assert not service.is_model_loaded()
    mock_esrgan_service_logger.warning.assert_any_call(
        f"{ESRGANService.MODEL_NAME} package not available. Service will rely on fallback upscaling methods."
    )

@patch('app.services.ai.esrgan_service.RRDBNet', side_effect=RuntimeError("RRDBNet init failed"))
def test_load_model_raises_model_load_error_on_rrdbnet_failure(mock_rrdbnet_init, esrgan_service_instance, mock_esrgan_service_logger):
    with pytest.raises(ModelLoadError) as exc_info:
        esrgan_service_instance.load_model()

    assert exc_info.value.model_name == ESRGANService.MODEL_NAME
    assert isinstance(exc_info.value.original_exception, RuntimeError)
    assert not esrgan_service_instance.model_loaded
    mock_esrgan_service_logger.error.assert_any_call(
        f"An unexpected error occurred while loading {ESRGANService.MODEL_NAME} model: RRDBNet init failed", exc_info=True
    )


@patch('app.services.ai.esrgan_service.RealESRGANer', side_effect=FileNotFoundError("Model .pth not found"))
def test_load_model_raises_model_load_error_on_realesrganer_failure(mock_realesrganer_init, esrgan_service_instance, mock_esrgan_service_logger):
    # RRDBNet is mocked by fixture to succeed
    with pytest.raises(ModelLoadError) as exc_info:
        esrgan_service_instance.load_model()

    assert exc_info.value.model_name == ESRGANService.MODEL_NAME
    assert isinstance(exc_info.value.original_exception, FileNotFoundError)
    assert not esrgan_service_instance.model_loaded
    mock_esrgan_service_logger.error.assert_any_call(
        f"{ESRGANService.MODEL_NAME} model file not found at {settings.ESRGAN_MODEL_PATH}: Model .pth not found", exc_info=True
    )


def test_is_model_loaded(esrgan_service_instance, monkeypatch):
    assert not esrgan_service_instance.is_model_loaded() # Initially false

    # Simulate successful load
    esrgan_service_instance.upsampler = MagicMock()
    esrgan_service_instance.model_loaded = True
    # ESRGAN_AVAILABLE is True due to fixture setup
    assert esrgan_service_instance.is_model_loaded()

    monkeypatch.setattr('app.services.ai.esrgan_service.ESRGAN_AVAILABLE', False)
    assert not esrgan_service_instance.is_model_loaded()


@patch('torch.cuda.is_available')
@patch('torch.cuda.empty_cache')
def test_unload_model_cuda(mock_empty_cache, mock_is_available, esrgan_service_instance, mock_esrgan_service_logger):
    mock_is_available.return_value = True
    esrgan_service_instance.device = "cuda"
    esrgan_service_instance.upsampler = MagicMock()
    # Mock the internal model within upsampler if your unload tries to interact with it
    esrgan_service_instance.upsampler.model = MagicMock()
    esrgan_service_instance.model_loaded = True

    esrgan_service_instance.unload_model()

    assert esrgan_service_instance.upsampler is None
    assert not esrgan_service_instance.model_loaded
    mock_empty_cache.assert_called_once()
    mock_esrgan_service_logger.info.assert_any_call(f"{ESRGANService.MODEL_NAME} model unloaded successfully.")

@patch('app.services.ai.esrgan_service.ESRGANService._fallback_upscale')
def test_upscale_image_uses_fallback_when_esrgan_unavailable(mock_fallback_method, monkeypatch, mock_esrgan_service_logger):
    monkeypatch.setattr('app.services.ai.esrgan_service.ESRGAN_AVAILABLE', False)
    service = ESRGANService() # Re-initialize

    mock_image = MagicMock(spec=Image.Image)
    mock_image.mode = "RGB"; mock_image.size = (100,100)

    service.upscale_image(mock_image, scale_factor=4)

    mock_fallback_method.assert_called_once_with(mock_image, 4)
    mock_esrgan_service_logger.warning.assert_any_call(
        f"{ESRGANService.MODEL_NAME} package not available. Service will rely on fallback upscaling methods."
    )
    mock_esrgan_service_logger.error.assert_any_call(
        f"{ESRGANService.MODEL_NAME} failed to load. Using fallback for upscale_image."
    )


@pytest.fixture(scope="module", autouse=True)
def restore_esrgan_available_after_tests(monkeypatch):
    yield
    monkeypatch.setattr('app.services.ai.esrgan_service.ESRGAN_AVAILABLE', ACTUAL_ESRGAN_AVAILABLE)
