import pytest
from unittest.mock import patch, MagicMock
import torch
from PIL import Image
import numpy as np

# Import service and its availability flags
from app.services.ai.model_enhancement_service import (
    ModelEnhancementService,
    GFPGAN_AVAILABLE as ACTUAL_GFPGAN_AVAILABLE,
    MEDIAPIPE_AVAILABLE as ACTUAL_MEDIAPIPE_AVAILABLE,
    INSIGHTFACE_AVAILABLE as ACTUAL_INSIGHTFACE_AVAILABLE
)
from app.core.config import settings
# Assuming ModelLoadError might be raised by sub-loaders if they were refactored for it.
# For now, test based on current behavior (warnings, None attributes).

@pytest.fixture(autouse=True)
def mock_mes_logger(): # MES for ModelEnhancementService
    with patch('app.services.ai.model_enhancement_service.logger') as mock_log:
        yield mock_log

@pytest.fixture
def mes_service_instance(monkeypatch):
    # By default, assume all packages are available for most tests
    monkeypatch.setattr('app.services.ai.model_enhancement_service.GFPGAN_AVAILABLE', True)
    monkeypatch.setattr('app.services.ai.model_enhancement_service.MEDIAPIPE_AVAILABLE', True)
    monkeypatch.setattr('app.services.ai.model_enhancement_service.INSIGHTFACE_AVAILABLE', True)
    service = ModelEnhancementService()
    service.device = "cuda" if torch.cuda.is_available() else "cpu"
    return service

# Mocks for the external libraries
@pytest.fixture
def mock_gfpganer():
    with patch('app.services.ai.model_enhancement_service.GFPGANer') as mock:
        yield mock

@pytest.fixture
def mock_mediapipe_pose():
    with patch('app.services.ai.model_enhancement_service.mp.solutions.pose.Pose') as mock:
        yield mock

@pytest.fixture
def mock_insightface_app():
    with patch('app.services.ai.model_enhancement_service.insightface.app.FaceAnalysis') as mock:
        mock_instance = MagicMock()
        # Mock the .prepare() method if it's called
        mock_instance.prepare = MagicMock()
        # Mock the .get() method to return an empty list (no faces) by default for simple tests
        mock_instance.get = MagicMock(return_value=[])
        mock.return_value = mock_instance
        yield mock


def test_mes_initialization(mes_service_instance):
    assert not mes_service_instance.model_loaded
    assert mes_service_instance.gfpgan_enhancer is None
    assert mes_service_instance.pose_detector is None
    assert mes_service_instance.face_analyzer is None
    assert mes_service_instance.device in ["cuda", "cpu"]

@patch.object(ModelEnhancementService, '_load_gfpgan', return_value=True)
@patch.object(ModelEnhancementService, '_load_mediapipe', return_value=True)
@patch.object(ModelEnhancementService, '_load_insightface', return_value=True)
def test_load_models_all_available_and_successful(
    mock_load_insightface, mock_load_mediapipe, mock_load_gfpgan,
    mes_service_instance, mock_mes_logger
):
    mes_service_instance.load_models()

    mock_load_gfpgan.assert_called_once()
    mock_load_mediapipe.assert_called_once()
    mock_load_insightface.assert_called_once()
    assert mes_service_instance.model_loaded  # Should be True, indicating service is "ready"
    mock_mes_logger.info.assert_any_call("Model enhancement models loading process completed. Some models may be using fallbacks.")

def test_load_models_gfpgan_unavailable(monkeypatch, mes_service_instance, mock_mes_logger):
    monkeypatch.setattr('app.services.ai.model_enhancement_service.GFPGAN_AVAILABLE', False)
    # Mock other loaders to succeed
    with patch.object(ModelEnhancementService, '_load_mediapipe', return_value=True), \
         patch.object(ModelEnhancementService, '_load_insightface', return_value=True):
        mes_service_instance.load_models()

    mock_mes_logger.warning.assert_any_call("GFPGAN package not available. Skipping GFPGAN loading.")
    assert mes_service_instance.gfpgan_enhancer is None
    assert mes_service_instance.model_loaded # Still true as other models might load / fallbacks active

@patch('app.services.ai.model_enhancement_service.GFPGANer', side_effect=Exception("GFPGAN load error"))
def test_load_gfpgan_failure(mock_gfpgan_init, mes_service_instance, mock_mes_logger):
    # Ensure GFPGAN_AVAILABLE is True for this test
    monkeypatch.setattr('app.services.ai.model_enhancement_service.GFPGAN_AVAILABLE', True)

    result = mes_service_instance._load_gfpgan() # Test private method directly

    assert not result # Should return False on failure
    assert mes_service_instance.gfpgan_enhancer is None
    mock_mes_logger.warning.assert_any_call("Failed to load GFPGAN: GFPGAN load error", exc_info=True)

# Similar tests can be written for _load_mediapipe and _load_insightface failures

def test_is_model_loaded(mes_service_instance):
    assert not mes_service_instance.is_model_loaded() # Initially false, as load_models not called
    mes_service_instance.model_loaded = True # Simulate that load_models was called
    assert mes_service_instance.is_model_loaded()


@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.empty_cache')
def test_unload_models_cuda(mock_empty_cache, mock_is_cuda_available, mes_service_instance, mock_mes_logger):
    # Simulate that some models were loaded
    mes_service_instance.gfpgan_enhancer = MagicMock()
    mes_service_instance.pose_detector = MagicMock()
    mes_service_instance.face_analyzer = MagicMock()
    mes_service_instance.model_loaded = True
    mes_service_instance.device = "cuda"

    mes_service_instance.unload_models()

    assert mes_service_instance.gfpgan_enhancer is None
    assert mes_service_instance.pose_detector is None
    assert mes_service_instance.face_analyzer is None
    assert not mes_service_instance.model_loaded
    mock_empty_cache.assert_called_once()
    mock_mes_logger.info.assert_any_call("Model enhancement models unloaded successfully. Unloaded 3 components.")


# Test fallback usage in enhance_model
@patch.object(ModelEnhancementService, '_fallback_face_enhancement')
@patch.object(ModelEnhancementService, '_fallback_pose_detection') # if optimize_body_proportions calls it
def test_enhance_model_uses_fallbacks(
    mock_fallback_pose, mock_fallback_face,
    mes_service_instance, monkeypatch, mock_mes_logger, mock_insightface_app # Use mock_insightface_app
):
    # Ensure primary models are "unavailable" or "not loaded"
    monkeypatch.setattr('app.services.ai.model_enhancement_service.GFPGAN_AVAILABLE', False)
    monkeypatch.setattr('app.services.ai.model_enhancement_service.MEDIAPIPE_AVAILABLE', False)

    # Re-initialize after setting AVAILABLE to False, or mock internal loaders to fail
    # For simplicity, we ensure the enhancer/detector attributes are None
    mes_service_instance.gfpgan_enhancer = None
    mes_service_instance.pose_detector = None

    # Simulate face detection returning one face so enhance_face is attempted
    mock_face_data = [{"bbox": [0,0,10,10], "confidence": 0.9}]
    with patch.object(mes_service_instance, 'detect_faces', return_value=mock_face_data):
        mock_image = Image.new("RGB", (100,100))
        result = mes_service_instance.enhance_model(mock_image)

    mock_fallback_face.assert_called_once()
    # Depending on optimize_body_proportions logic, _fallback_pose_detection might be called if pose_detector is None
    # The current enhance_model calls detect_pose, which then calls _fallback_pose_detection if self.pose_detector is None.
    # So mock_fallback_pose should be called.
    mock_fallback_pose.assert_called_once()
    assert result["fallback_used"] is True
    assert result["component_fallbacks"]["gfpgan"] is True
    assert result["component_fallbacks"]["mediapipe_pose"] is True


# Restore ACTUAL flags for other test modules
@pytest.fixture(scope="module", autouse=True)
def restore_availability_flags(monkeypatch):
    yield
    monkeypatch.setattr('app.services.ai.model_enhancement_service.GFPGAN_AVAILABLE', ACTUAL_GFPGAN_AVAILABLE)
    monkeypatch.setattr('app.services.ai.model_enhancement_service.MEDIAPIPE_AVAILABLE', ACTUAL_MEDIAPIPE_AVAILABLE)
    monkeypatch.setattr('app.services.ai.model_enhancement_service.INSIGHTFACE_AVAILABLE', ACTUAL_INSIGHTFACE_AVAILABLE)

# TODO: Add tests for detect_faces, enhance_face, detect_pose and their fallbacks individually.
# TODO: Add tests for optimize_body_proportions and its helpers.
# TODO: Add tests for _assess_face_quality.
# These would involve more complex mocking of CV and PIL operations.
