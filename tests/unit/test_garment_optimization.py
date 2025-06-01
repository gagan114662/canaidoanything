import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch
import time

from app.services.ai.garment_optimization_service import GarmentOptimizationService
from tests.conftest import PERFORMANCE_TARGETS, QUALITY_THRESHOLDS

class TestGarmentOptimizationService:
    """Test suite for garment optimization functionality"""
    
    @pytest.fixture
    def garment_service(self):
        """Create GarmentOptimizationService instance for testing"""
        return GarmentOptimizationService()
    
    def test_service_initialization(self, garment_service):
        """Test that service initializes correctly"""
        assert garment_service is not None
        assert not garment_service.is_model_loaded()
        assert garment_service.device in ["cuda", "cpu"]
    
    def test_garment_segmentation_accuracy(self, garment_service, sample_model_image):
        """Test garment segmentation accuracy"""
        # Arrange
        garment_service.load_models()
        
        # Act
        segmentation_result = garment_service.segment_garment(sample_model_image)
        
        # Assert
        assert "mask" in segmentation_result
        assert "confidence" in segmentation_result
        assert "garment_type" in segmentation_result
        
        mask = segmentation_result["mask"]
        assert isinstance(mask, Image.Image)
        assert mask.size == sample_model_image.size
        assert mask.mode in ["L", "1"]  # Grayscale or binary mask
        
        # Check segmentation quality
        confidence = segmentation_result["confidence"]
        assert confidence >= 0.8  # High confidence required
    
    def test_garment_type_detection(self, garment_service, sample_garment_image):
        """Test accurate garment type detection"""
        # Arrange
        garment_service.load_models()
        
        # Act
        detection_result = garment_service.detect_garment_type(sample_garment_image)
        
        # Assert
        assert "garment_type" in detection_result
        assert "confidence" in detection_result
        assert "attributes" in detection_result
        
        garment_type = detection_result["garment_type"]
        assert garment_type in ["shirt", "dress", "pants", "jacket", "top", "bottom", "outerwear"]
        
        confidence = detection_result["confidence"]
        assert confidence >= 0.7  # Reasonable confidence threshold
    
    def test_fit_optimization_quality(self, garment_service, sample_model_image):
        """Test that fit optimization improves garment appearance"""
        # Arrange
        garment_service.load_models()
        
        # Get original garment fit score
        original_fit = garment_service._assess_garment_fit(sample_model_image)
        
        # Act
        optimized_result = garment_service.optimize_fit(sample_model_image)
        
        # Assert
        assert "optimized_image" in optimized_result
        assert "fit_score" in optimized_result
        assert "improvements" in optimized_result
        
        optimized_image = optimized_result["optimized_image"]
        optimized_fit = garment_service._assess_garment_fit(optimized_image)
        
        # Fit should improve or stay the same
        assert optimized_fit >= original_fit
        assert optimized_result["fit_score"] >= QUALITY_THRESHOLDS["garment_fit_min"]
    
    def test_style_enhancement(self, garment_service, sample_model_image):
        """Test garment style enhancement capabilities"""
        # Arrange
        garment_service.load_models()
        style_parameters = {
            "enhancement_level": 0.8,
            "color_adjustment": True,
            "wrinkle_reduction": True,
            "fabric_enhancement": True
        }
        
        # Act
        enhanced_result = garment_service.enhance_style(sample_model_image, style_parameters)
        
        # Assert
        assert "enhanced_image" in enhanced_result
        assert "style_score" in enhanced_result
        assert "enhancements_applied" in enhanced_result
        
        enhanced_image = enhanced_result["enhanced_image"]
        assert isinstance(enhanced_image, Image.Image)
        assert enhanced_image.size == sample_model_image.size
        
        # Style score should meet threshold
        assert enhanced_result["style_score"] >= 7.0  # Good style score
    
    def test_color_correction_accuracy(self, garment_service, sample_garment_image):
        """Test color correction and enhancement"""
        # Arrange
        garment_service.load_models()
        
        # Act
        color_result = garment_service.correct_colors(sample_garment_image)
        
        # Assert
        assert "corrected_image" in color_result
        assert "color_analysis" in color_result
        assert "adjustments_made" in color_result
        
        corrected_image = color_result["corrected_image"]
        color_analysis = color_result["color_analysis"]
        
        # Check color consistency
        assert "dominant_colors" in color_analysis
        assert "color_harmony_score" in color_analysis
        assert color_analysis["color_harmony_score"] >= 0.7
    
    def test_wrinkle_removal(self, garment_service, sample_model_image):
        """Test wrinkle detection and removal"""
        # Arrange
        garment_service.load_models()
        
        # Act
        wrinkle_result = garment_service.remove_wrinkles(sample_model_image)
        
        # Assert
        assert "smoothed_image" in wrinkle_result
        assert "wrinkles_detected" in wrinkle_result
        assert "smoothness_score" in wrinkle_result
        
        smoothed_image = wrinkle_result["smoothed_image"]
        smoothness_score = wrinkle_result["smoothness_score"]
        
        # Smoothness should improve
        assert smoothness_score >= 0.8
        assert isinstance(smoothed_image, Image.Image)
    
    def test_performance_requirements(self, garment_service, sample_model_image, performance_timer):
        """Test that garment optimization meets performance requirements"""
        # Arrange
        garment_service.load_models()
        
        # Act
        performance_timer.start()
        result = garment_service.optimize_garment(sample_model_image)
        performance_timer.stop()
        
        # Assert
        assert performance_timer.elapsed <= PERFORMANCE_TARGETS["garment_optimization_max_time"]
        assert result is not None
        assert "optimized_image" in result
        assert result["overall_score"] >= QUALITY_THRESHOLDS["garment_fit_min"]
    
    def test_edge_case_no_garment_detected(self, garment_service):
        """Test handling when no garment is detected"""
        # Arrange
        garment_service.load_models()
        
        # Create image with no clear garment (face only)
        face_only = Image.new('RGB', (512, 512))
        pixels = face_only.load()
        # Simple face pattern
        for i in range(200, 300):
            for j in range(200, 300):
                pixels[i, j] = (255, 220, 177)  # Skin tone
        
        # Act
        result = garment_service.optimize_garment(face_only)
        
        # Assert - Should handle gracefully
        assert result is not None
        assert "optimized_image" in result
        assert "garments_detected" in result
        assert result["garments_detected"] == 0
        assert "fallback_used" in result
        assert result["fallback_used"] is True
    
    def test_multiple_garments_handling(self, garment_service):
        """Test handling of multiple garments in single image"""
        # Arrange
        garment_service.load_models()
        
        # Create image with multiple garment types
        multi_garment = Image.new('RGB', (512, 512))
        pixels = multi_garment.load()
        
        # Top garment
        for i in range(100, 250):
            for j in range(150, 350):
                pixels[i, j] = (255, 0, 0)  # Red top
        
        # Bottom garment
        for i in range(300, 450):
            for j in range(150, 350):
                pixels[i, j] = (0, 0, 255)  # Blue bottom
        
        # Act
        result = garment_service.optimize_garment(multi_garment)
        
        # Assert
        assert result is not None
        assert "garments_detected" in result
        assert result["garments_detected"] >= 2
        assert "garment_types" in result
        assert len(result["garment_types"]) >= 2
    
    def test_fabric_texture_enhancement(self, garment_service, sample_garment_image):
        """Test fabric texture enhancement"""
        # Arrange
        garment_service.load_models()
        
        # Act
        texture_result = garment_service.enhance_fabric_texture(sample_garment_image)
        
        # Assert
        assert "enhanced_image" in texture_result
        assert "texture_score" in texture_result
        assert "fabric_type" in texture_result
        
        enhanced_image = texture_result["enhanced_image"]
        texture_score = texture_result["texture_score"]
        
        assert isinstance(enhanced_image, Image.Image)
        assert texture_score >= 0.7  # Good texture enhancement
    
    def test_garment_lighting_optimization(self, garment_service, sample_model_image):
        """Test garment-specific lighting optimization"""
        # Arrange
        garment_service.load_models()
        
        # Act
        lighting_result = garment_service.optimize_garment_lighting(sample_model_image)
        
        # Assert
        assert "lit_image" in lighting_result
        assert "lighting_score" in lighting_result
        assert "lighting_type" in lighting_result
        
        lit_image = lighting_result["lit_image"]
        lighting_score = lighting_result["lighting_score"]
        
        assert isinstance(lit_image, Image.Image)
        assert lighting_score >= 0.8  # High lighting quality
    
    @pytest.mark.parametrize("garment_type", ["shirt", "dress", "pants", "jacket"])
    def test_type_specific_optimization(self, garment_service, sample_model_image, garment_type):
        """Test optimization works for different garment types"""
        # Arrange
        garment_service.load_models()
        
        # Act
        result = garment_service.optimize_by_type(sample_model_image, garment_type)
        
        # Assert
        assert result is not None
        assert "optimized_image" in result
        assert "type_specific_score" in result
        assert result["garment_type"] == garment_type
    
    def test_memory_efficiency(self, garment_service, sample_model_image):
        """Test memory efficiency during garment optimization"""
        # Arrange
        garment_service.load_models()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Act - Process multiple images
        for _ in range(3):
            result = garment_service.optimize_garment(sample_model_image)
            assert result is not None
        
        # Force cleanup
        garment_service.clear_cache()
        
        # Assert - Memory should not grow excessively
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_growth = final_memory - initial_memory
        
        # Allow some growth but not excessive (< 2GB)
        assert memory_growth < 2 * 1024 * 1024 * 1024
    
    def test_quality_consistency(self, garment_service, sample_model_image):
        """Test that optimization results are consistent across runs"""
        # Arrange
        garment_service.load_models()
        
        # Act - Run optimization multiple times
        scores = []
        for _ in range(3):
            result = garment_service.optimize_garment(sample_model_image)
            scores.append(result["overall_score"])
        
        # Assert - Scores should be consistent
        score_variance = np.var(scores)
        assert score_variance < 0.2  # Low variance expected
    
    def test_batch_optimization(self, garment_service, sample_model_image):
        """Test batch optimization of multiple garment images"""
        # Arrange
        garment_service.load_models()
        images = [sample_model_image] * 4
        
        # Act
        results = garment_service.optimize_batch(images)
        
        # Assert
        assert len(results) == 4
        for result in results:
            assert "optimized_image" in result
            assert "overall_score" in result
            assert result["overall_score"] >= 6.0  # Decent quality threshold