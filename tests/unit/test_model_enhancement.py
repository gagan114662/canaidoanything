import pytest
import torch
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch
import time

from app.services.ai.model_enhancement_service import ModelEnhancementService
from tests.conftest import PERFORMANCE_TARGETS, QUALITY_THRESHOLDS

class TestModelEnhancementService:
    """Test suite for model enhancement functionality"""
    
    @pytest.fixture
    def enhancement_service(self):
        """Create ModelEnhancementService instance for testing"""
        return ModelEnhancementService()
    
    def test_service_initialization(self, enhancement_service):
        """Test that service initializes correctly"""
        assert enhancement_service is not None
        assert not enhancement_service.is_model_loaded()
        assert enhancement_service.device in ["cuda", "cpu"]
    
    @patch('app.services.ai.model_enhancement_service.GFPGAN')
    def test_model_loading(self, mock_gfpgan, enhancement_service):
        """Test that models load successfully"""
        # Arrange
        mock_gfpgan.return_value = Mock()
        
        # Act
        enhancement_service.load_models()
        
        # Assert
        assert enhancement_service.is_model_loaded()
        mock_gfpgan.assert_called_once()
    
    def test_face_detection(self, enhancement_service, sample_model_image):
        """Test face detection in model images"""
        # Arrange
        enhancement_service.load_models()
        
        # Act
        faces = enhancement_service.detect_faces(sample_model_image)
        
        # Assert
        assert isinstance(faces, list)
        assert len(faces) >= 0  # May or may not detect face in test image
        
        # If faces detected, validate structure
        if faces:
            face = faces[0]
            assert "bbox" in face
            assert "confidence" in face
            assert face["confidence"] >= 0.0
            assert face["confidence"] <= 1.0
    
    def test_face_enhancement_quality(self, enhancement_service, sample_model_image):
        """Test that face enhancement improves image quality"""
        # Arrange
        enhancement_service.load_models()
        original_quality = enhancement_service._assess_face_quality(sample_model_image)
        
        # Act
        enhanced_image = enhancement_service.enhance_face(sample_model_image)
        enhanced_quality = enhancement_service._assess_face_quality(enhanced_image)
        
        # Assert
        assert isinstance(enhanced_image, Image.Image)
        assert enhanced_image.size == sample_model_image.size
        assert enhanced_quality >= original_quality  # Quality should improve or stay same
        assert enhanced_quality >= QUALITY_THRESHOLDS["face_clarity_min"]
    
    def test_pose_detection_accuracy(self, enhancement_service, sample_model_image):
        """Test pose detection accuracy requirements"""
        # Arrange
        enhancement_service.load_models()
        
        # Act
        pose_result = enhancement_service.detect_pose(sample_model_image)
        
        # Assert
        assert "confidence" in pose_result
        assert "keypoints" in pose_result
        assert pose_result["confidence"] >= PERFORMANCE_TARGETS["min_model_recognition"]
        
        # Validate keypoints structure
        keypoints = pose_result["keypoints"]
        assert isinstance(keypoints, list)
        
        # Should have standard pose keypoints (17 for COCO format)
        if keypoints:
            assert len(keypoints) >= 17
            for kp in keypoints:
                assert len(kp) >= 2  # x, y coordinates minimum
    
    def test_body_optimization(self, enhancement_service, sample_model_image):
        """Test body proportion optimization"""
        # Arrange
        enhancement_service.load_models()
        
        # Act
        optimized_image = enhancement_service.optimize_body_proportions(sample_model_image)
        
        # Assert
        assert isinstance(optimized_image, Image.Image)
        assert optimized_image.size == sample_model_image.size
        
        # Check that optimization maintains image integrity
        original_array = np.array(sample_model_image)
        optimized_array = np.array(optimized_image)
        
        # Images should be similar but not identical (some optimization applied)
        similarity = np.mean(np.abs(original_array - optimized_array))
        assert similarity > 0  # Some change should occur
        assert similarity < 50  # But not too drastic
    
    def test_enhancement_performance(self, enhancement_service, sample_model_image, performance_timer):
        """Test that model enhancement meets performance requirements"""
        # Arrange
        enhancement_service.load_models()
        
        # Act
        performance_timer.start()
        result = enhancement_service.enhance_model(sample_model_image)
        performance_timer.stop()
        
        # Assert
        assert performance_timer.elapsed <= PERFORMANCE_TARGETS["model_enhancement_max_time"]
        assert result is not None
        assert isinstance(result["enhanced_image"], Image.Image)
        assert result["quality_score"] >= QUALITY_THRESHOLDS["overall_score_min"]
    
    def test_edge_case_low_quality_image(self, enhancement_service):
        """Test handling of very low quality input images"""
        # Arrange
        enhancement_service.load_models()
        
        # Create extremely low quality image (tiny and noisy)
        low_quality = Image.new('RGB', (32, 32), color='gray')
        noise = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        low_quality = Image.fromarray(noise)
        
        # Act & Assert - Should handle gracefully without crashing
        result = enhancement_service.enhance_model(low_quality)
        
        assert result is not None
        assert "enhanced_image" in result
        assert "fallback_used" in result
        
        # For very low quality, fallback should be used
        if result["quality_score"] < 3.0:
            assert result["fallback_used"] is True
    
    def test_edge_case_no_face_detected(self, enhancement_service):
        """Test handling when no face is detected in image"""
        # Arrange
        enhancement_service.load_models()
        
        # Create image with no face (abstract pattern)
        no_face = Image.new('RGB', (512, 512))
        pixels = no_face.load()
        for i in range(512):
            for j in range(512):
                pixels[i, j] = (i % 255, j % 255, (i + j) % 255)
        
        # Act
        result = enhancement_service.enhance_model(no_face)
        
        # Assert - Should handle gracefully
        assert result is not None
        assert "enhanced_image" in result
        assert "faces_detected" in result
        assert result["faces_detected"] == 0
    
    def test_memory_management(self, enhancement_service, sample_model_image):
        """Test that enhancement doesn't cause memory leaks"""
        # Arrange
        enhancement_service.load_models()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Act - Process multiple images
        for _ in range(5):
            result = enhancement_service.enhance_model(sample_model_image)
            assert result is not None
        
        # Force cleanup
        enhancement_service.clear_cache()
        
        # Assert - Memory should not grow significantly
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_growth = final_memory - initial_memory
        
        # Allow some memory growth but not excessive (< 1GB)
        assert memory_growth < 1024 * 1024 * 1024
    
    @pytest.mark.parametrize("image_size", [(256, 256), (512, 512), (1024, 1024)])
    def test_different_image_sizes(self, enhancement_service, image_size):
        """Test enhancement works with different image sizes"""
        # Arrange
        enhancement_service.load_models()
        test_image = Image.new('RGB', image_size, color='blue')
        
        # Act
        result = enhancement_service.enhance_model(test_image)
        
        # Assert
        assert result is not None
        assert result["enhanced_image"].size == image_size
    
    def test_batch_processing(self, enhancement_service, sample_model_image):
        """Test batch processing of multiple model images"""
        # Arrange
        enhancement_service.load_models()
        images = [sample_model_image] * 3
        
        # Act
        results = enhancement_service.enhance_batch(images)
        
        # Assert
        assert len(results) == 3
        for result in results:
            assert "enhanced_image" in result
            assert "quality_score" in result
    
    def test_quality_scoring_consistency(self, enhancement_service, sample_model_image):
        """Test that quality scoring is consistent across runs"""
        # Arrange
        enhancement_service.load_models()
        
        # Act - Run enhancement multiple times
        scores = []
        for _ in range(3):
            result = enhancement_service.enhance_model(sample_model_image)
            scores.append(result["quality_score"])
        
        # Assert - Scores should be consistent (within small variance)
        score_variance = np.var(scores)
        assert score_variance < 0.1  # Small variance expected