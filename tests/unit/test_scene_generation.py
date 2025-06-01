import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch
import time

from app.services.ai.scene_generation_service import SceneGenerationService
from tests.conftest import PERFORMANCE_TARGETS, QUALITY_THRESHOLDS

class TestSceneGenerationService:
    """Test suite for scene generation functionality"""
    
    @pytest.fixture
    def scene_service(self):
        """Create SceneGenerationService instance for testing"""
        return SceneGenerationService()
    
    def test_service_initialization(self, scene_service):
        """Test that service initializes correctly"""
        assert scene_service is not None
        assert not scene_service.is_model_loaded()
        assert scene_service.device in ["cuda", "cpu"]
    
    @patch('app.services.ai.scene_generation_service.FluxPipeline')
    def test_model_loading(self, mock_flux, scene_service):
        """Test that FLUX Kontext models load successfully"""
        # Arrange
        mock_flux.from_pretrained.return_value = Mock()
        
        # Act
        scene_service.load_models()
        
        # Assert
        assert scene_service.is_model_loaded()
        mock_flux.from_pretrained.assert_called()
    
    def test_background_generation_coherence(self, scene_service, sample_model_image, style_variations):
        """Test that generated backgrounds are coherent with style"""
        # Arrange
        scene_service.load_models()
        style = style_variations["editorial"]
        
        # Act
        bg_result = scene_service.generate_background(
            sample_model_image, 
            style["style_prompt"],
            style["background_type"]
        )
        
        # Assert
        assert "background_image" in bg_result
        assert "coherence_score" in bg_result
        assert "style_match" in bg_result
        
        background = bg_result["background_image"]
        assert isinstance(background, Image.Image)
        assert background.size == sample_model_image.size
        
        # Coherence should meet threshold
        assert bg_result["coherence_score"] >= QUALITY_THRESHOLDS["scene_coherence_min"]
    
    def test_scene_composition_quality(self, scene_service, sample_model_image, sample_garment_image):
        """Test scene composition with model and garment"""
        # Arrange
        scene_service.load_models()
        
        # Act
        composition_result = scene_service.compose_scene(
            model_image=sample_model_image,
            garment_image=sample_garment_image,
            background_type="studio",
            composition_style="commercial"
        )
        
        # Assert
        assert "composed_scene" in composition_result
        assert "composition_score" in composition_result
        assert "elements_integrated" in composition_result
        
        composed_scene = composition_result["composed_scene"]
        assert isinstance(composed_scene, Image.Image)
        assert composed_scene.size == sample_model_image.size
        
        # Composition quality should be high
        assert composition_result["composition_score"] >= 8.0
    
    def test_multiple_style_variations(self, scene_service, sample_model_image, style_variations):
        """Test generation of multiple style variations"""
        # Arrange
        scene_service.load_models()
        
        # Act
        variation_results = []
        for style_name, style_config in style_variations.items():
            result = scene_service.generate_styled_scene(
                sample_model_image,
                style_config["style_prompt"],
                style_name
            )
            variation_results.append((style_name, result))
        
        # Assert
        assert len(variation_results) == 5  # All 5 styles generated
        
        for style_name, result in variation_results:
            assert "styled_scene" in result
            assert "style_consistency" in result
            assert "variation_id" in result
            
            # Each variation should be distinct but high quality
            assert result["style_consistency"] >= 0.8
            assert isinstance(result["styled_scene"], Image.Image)
    
    def test_lighting_simulation_accuracy(self, scene_service, sample_model_image):
        """Test professional lighting simulation"""
        # Arrange
        scene_service.load_models()
        lighting_types = ["studio", "natural", "dramatic", "soft", "rim"]
        
        # Act & Assert
        for lighting_type in lighting_types:
            lighting_result = scene_service.apply_lighting(
                sample_model_image,
                lighting_type=lighting_type,
                intensity=0.8
            )
            
            assert "lit_image" in lighting_result
            assert "lighting_quality" in lighting_result
            assert "lighting_type" in lighting_result
            
            lit_image = lighting_result["lit_image"]
            assert isinstance(lit_image, Image.Image)
            assert lit_image.size == sample_model_image.size
            
            # Lighting quality should meet threshold
            assert lighting_result["lighting_quality"] >= QUALITY_THRESHOLDS["lighting_quality_min"]
    
    def test_depth_and_perspective_control(self, scene_service, sample_model_image):
        """Test depth and perspective manipulation"""
        # Arrange
        scene_service.load_models()
        
        # Act
        depth_result = scene_service.adjust_depth_perspective(
            sample_model_image,
            depth_level=0.7,
            perspective_angle=15,
            focal_point="center"
        )
        
        # Assert
        assert "depth_adjusted_image" in depth_result
        assert "depth_map" in depth_result
        assert "perspective_quality" in depth_result
        
        depth_image = depth_result["depth_adjusted_image"]
        depth_map = depth_result["depth_map"]
        
        assert isinstance(depth_image, Image.Image)
        assert isinstance(depth_map, Image.Image)
        assert depth_result["perspective_quality"] >= 0.8
    
    def test_context_aware_backgrounds(self, scene_service, sample_model_image):
        """Test context-aware background generation"""
        # Arrange
        scene_service.load_models()
        contexts = ["office", "outdoor", "party", "casual", "formal"]
        
        # Act & Assert
        for context in contexts:
            bg_result = scene_service.generate_context_background(
                sample_model_image,
                context=context,
                adaptation_level=0.8
            )
            
            assert "background_image" in bg_result
            assert "context_match" in bg_result
            assert "adaptation_score" in bg_result
            
            # Context should be well-matched
            assert bg_result["context_match"] >= 0.8
            assert bg_result["adaptation_score"] >= 0.75
    
    def test_brand_environment_matching(self, scene_service, sample_model_image):
        """Test brand-specific environment generation"""
        # Arrange
        scene_service.load_models()
        brand_guidelines = {
            "color_palette": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
            "style": "modern minimalist",
            "mood": "sophisticated",
            "environment_type": "studio"
        }
        
        # Act
        brand_result = scene_service.generate_brand_environment(
            sample_model_image,
            brand_guidelines
        )
        
        # Assert
        assert "brand_scene" in brand_result
        assert "brand_consistency" in brand_result
        assert "guideline_adherence" in brand_result
        
        brand_scene = brand_result["brand_scene"]
        assert isinstance(brand_scene, Image.Image)
        assert brand_result["brand_consistency"] >= 0.85
        assert brand_result["guideline_adherence"] >= 0.8
    
    def test_performance_requirements(self, scene_service, sample_model_image, performance_timer):
        """Test that scene generation meets performance requirements"""
        # Arrange
        scene_service.load_models()
        
        # Act
        performance_timer.start()
        result = scene_service.generate_complete_scene(sample_model_image)
        performance_timer.stop()
        
        # Assert
        assert performance_timer.elapsed <= PERFORMANCE_TARGETS["scene_generation_max_time"]
        assert result is not None
        assert "complete_scene" in result
        assert result["scene_quality"] >= QUALITY_THRESHOLDS["scene_coherence_min"]
    
    def test_batch_scene_generation(self, scene_service, sample_model_image):
        """Test batch generation of scenes for multiple styles"""
        # Arrange
        scene_service.load_models()
        images = [sample_model_image] * 3
        styles = ["editorial", "commercial", "lifestyle"]
        
        # Act
        batch_results = scene_service.generate_batch_scenes(images, styles)
        
        # Assert
        assert len(batch_results) == 3
        for i, result in enumerate(batch_results):
            assert "styled_scene" in result
            assert "style_applied" in result
            assert result["style_applied"] == styles[i]
            assert result["scene_quality"] >= 7.5
    
    def test_edge_case_minimal_input(self, scene_service):
        """Test handling of minimal input images"""
        # Arrange
        scene_service.load_models()
        
        # Create minimal input (very small image)
        minimal_image = Image.new('RGB', (64, 64), color='gray')
        
        # Act
        result = scene_service.generate_background(
            minimal_image,
            "professional background",
            "studio"
        )
        
        # Assert - Should handle gracefully
        assert result is not None
        assert "background_image" in result
        assert "fallback_used" in result
        
        # For minimal input, fallback should be used
        if result["coherence_score"] < 5.0:
            assert result["fallback_used"] is True
    
    def test_edge_case_high_resolution(self, scene_service):
        """Test handling of high resolution images"""
        # Arrange
        scene_service.load_models()
        
        # Create high resolution image
        high_res = Image.new('RGB', (2048, 2048), color='blue')
        
        # Act
        result = scene_service.generate_background(
            high_res,
            "professional studio",
            "clean"
        )
        
        # Assert
        assert result is not None
        assert "background_image" in result
        
        # Should handle high resolution appropriately
        bg_image = result["background_image"]
        assert isinstance(bg_image, Image.Image)
        
        # May be resized for processing efficiency
        assert max(bg_image.size) <= 2048
    
    def test_style_consistency_across_variations(self, scene_service, sample_model_image):
        """Test style consistency when generating multiple variations"""
        # Arrange
        scene_service.load_models()
        base_style = "professional fashion photography"
        
        # Act - Generate multiple variations of same style
        variations = []
        for i in range(3):
            result = scene_service.generate_styled_scene(
                sample_model_image,
                base_style,
                f"variation_{i}"
            )
            variations.append(result)
        
        # Assert - All variations should maintain style consistency
        for variation in variations:
            assert variation["style_consistency"] >= 0.85
            assert "styled_scene" in variation
        
        # Compare variations for reasonable consistency
        consistency_scores = [v["style_consistency"] for v in variations]
        score_variance = np.var(consistency_scores)
        assert score_variance < 0.05  # Low variance in consistency
    
    def test_composition_rules_adherence(self, scene_service, sample_model_image):
        """Test adherence to composition rules (rule of thirds, etc.)"""
        # Arrange
        scene_service.load_models()
        
        # Act
        composition_result = scene_service.apply_composition_rules(
            sample_model_image,
            rules=["rule_of_thirds", "golden_ratio", "symmetry"],
            primary_rule="rule_of_thirds"
        )
        
        # Assert
        assert "composed_image" in composition_result
        assert "rule_adherence" in composition_result
        assert "composition_analysis" in composition_result
        
        # Rule adherence should be high
        assert composition_result["rule_adherence"] >= 0.8
        
        # Composition analysis should include applied rules
        analysis = composition_result["composition_analysis"]
        assert "rules_applied" in analysis
        assert "rule_of_thirds" in analysis["rules_applied"]
    
    def test_scene_element_integration(self, scene_service, sample_model_image):
        """Test integration of multiple scene elements"""
        # Arrange
        scene_service.load_models()
        elements = {
            "background": "studio",
            "props": ["chair", "lighting_equipment"],
            "atmosphere": "professional",
            "color_scheme": "monochromatic"
        }
        
        # Act
        integration_result = scene_service.integrate_scene_elements(
            sample_model_image,
            elements
        )
        
        # Assert
        assert "integrated_scene" in integration_result
        assert "element_harmony" in integration_result
        assert "integration_quality" in integration_result
        
        # Integration should be seamless
        assert integration_result["element_harmony"] >= 0.8
        assert integration_result["integration_quality"] >= 8.0
    
    def test_memory_efficiency_scene_generation(self, scene_service, sample_model_image):
        """Test memory efficiency during scene generation"""
        # Arrange
        scene_service.load_models()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Act - Generate multiple scenes
        for i in range(3):
            result = scene_service.generate_complete_scene(sample_model_image)
            assert result is not None
        
        # Force cleanup
        scene_service.clear_cache()
        
        # Assert - Memory should not grow excessively
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_growth = final_memory - initial_memory
        
        # Allow some growth but not excessive (< 3GB)
        assert memory_growth < 3 * 1024 * 1024 * 1024
    
    def test_concurrent_generation_stability(self, scene_service, sample_model_image):
        """Test stability when generating scenes concurrently"""
        # Arrange
        scene_service.load_models()
        
        # Act - Simulate concurrent requests
        results = []
        styles = ["editorial", "commercial", "lifestyle"]
        
        for style in styles:
            result = scene_service.generate_styled_scene(
                sample_model_image,
                f"{style} photography",
                style
            )
            results.append(result)
        
        # Assert - All results should be valid
        assert len(results) == 3
        for result in results:
            assert "styled_scene" in result
            assert result["style_consistency"] >= 0.8
    
    @pytest.mark.parametrize("background_type", ["studio", "outdoor", "urban", "nature", "abstract"])
    def test_background_type_specificity(self, scene_service, sample_model_image, background_type):
        """Test generation of specific background types"""
        # Arrange
        scene_service.load_models()
        
        # Act
        result = scene_service.generate_background(
            sample_model_image,
            f"professional {background_type} background",
            background_type
        )
        
        # Assert
        assert result is not None
        assert "background_image" in result
        assert "background_type_match" in result
        
        # Type-specific score should be high
        assert result["background_type_match"] >= 0.8
    
    def test_quality_metrics_calculation(self, scene_service, sample_model_image):
        """Test accurate calculation of scene quality metrics"""
        # Arrange
        scene_service.load_models()
        
        # Act
        result = scene_service.generate_complete_scene(sample_model_image)
        metrics = scene_service.calculate_scene_metrics(result["complete_scene"])
        
        # Assert
        assert "coherence" in metrics
        assert "composition" in metrics
        assert "lighting_quality" in metrics
        assert "style_consistency" in metrics
        assert "overall_score" in metrics
        
        # All metrics should be in valid range
        for metric_name, value in metrics.items():
            if metric_name != "overall_score":
                assert 0.0 <= value <= 1.0
            else:
                assert 0.0 <= value <= 10.0