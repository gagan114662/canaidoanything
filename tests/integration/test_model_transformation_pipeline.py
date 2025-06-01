import pytest
import asyncio
import time
from PIL import Image
import numpy as np

from app.services.tasks.model_transformation import ModelTransformationPipeline
from app.services.ai.model_enhancement_service import ModelEnhancementService
from app.services.ai.garment_optimization_service import GarmentOptimizationService
from app.services.ai.scene_generation_service import SceneGenerationService
from tests.conftest import PERFORMANCE_TARGETS, QUALITY_THRESHOLDS

class TestModelTransformationPipeline:
    """Integration tests for the complete model transformation pipeline"""
    
    @pytest.fixture
    def transformation_pipeline(self):
        """Create complete transformation pipeline"""
        return ModelTransformationPipeline()
    
    @pytest.fixture
    def all_services_loaded(self, transformation_pipeline):
        """Ensure all services are loaded"""
        transformation_pipeline.load_all_models()
        return transformation_pipeline
    
    def test_end_to_end_transformation(
        self, 
        all_services_loaded, 
        sample_model_image, 
        style_variations,
        performance_timer
    ):
        """Test complete end-to-end transformation pipeline"""
        # Arrange
        pipeline = all_services_loaded
        style = style_variations["commercial"]
        
        # Act
        performance_timer.start()
        result = pipeline.transform_model(
            model_image=sample_model_image,
            style_prompt=style["style_prompt"],
            negative_prompt=style["negative_prompt"],
            num_variations=5,
            enhance_model=True,
            optimize_garment=True,
            generate_scene=True
        )
        performance_timer.stop()
        
        # Assert
        # Performance requirement
        assert performance_timer.elapsed <= PERFORMANCE_TARGETS["total_processing_max_time"]
        
        # Result structure
        assert "variations" in result
        assert "metadata" in result
        assert "quality_scores" in result
        
        # Number of variations
        assert len(result["variations"]) == 5
        
        # Quality requirements
        assert result["quality_scores"]["overall_average"] >= QUALITY_THRESHOLDS["overall_score_min"]
        
        # Each variation should be valid
        for i, variation in enumerate(result["variations"]):
            assert "transformed_image" in variation
            assert "style_type" in variation
            assert "quality_score" in variation
            
            transformed_image = variation["transformed_image"]
            assert isinstance(transformed_image, Image.Image)
            assert transformed_image.size == sample_model_image.size
            assert variation["quality_score"] >= QUALITY_THRESHOLDS["overall_score_min"]
    
    def test_model_enhancement_integration(
        self,
        all_services_loaded,
        sample_model_image,
        performance_timer
    ):
        """Test model enhancement integration in pipeline"""
        # Arrange
        pipeline = all_services_loaded
        
        # Act
        performance_timer.start()
        result = pipeline.enhance_model_step(sample_model_image)
        performance_timer.stop()
        
        # Assert
        assert performance_timer.elapsed <= PERFORMANCE_TARGETS["model_enhancement_max_time"]
        
        assert "enhanced_image" in result
        assert "enhancement_metadata" in result
        assert "quality_improvement" in result
        
        enhanced_image = result["enhanced_image"]
        assert isinstance(enhanced_image, Image.Image)
        assert enhanced_image.size == sample_model_image.size
        
        # Quality should improve
        assert result["quality_improvement"] >= 0  # At least no degradation
    
    def test_garment_optimization_integration(
        self,
        all_services_loaded,
        sample_model_image,
        performance_timer
    ):
        """Test garment optimization integration in pipeline"""
        # Arrange
        pipeline = all_services_loaded
        
        # Act
        performance_timer.start()
        result = pipeline.optimize_garment_step(sample_model_image)
        performance_timer.stop()
        
        # Assert
        assert performance_timer.elapsed <= PERFORMANCE_TARGETS["garment_optimization_max_time"]
        
        assert "optimized_image" in result
        assert "optimization_metadata" in result
        assert "garment_quality_score" in result
        
        optimized_image = result["optimized_image"]
        assert isinstance(optimized_image, Image.Image)
        assert result["garment_quality_score"] >= QUALITY_THRESHOLDS["garment_fit_min"]
    
    def test_scene_generation_integration(
        self,
        all_services_loaded,
        sample_model_image,
        style_variations,
        performance_timer
    ):
        """Test scene generation integration in pipeline"""
        # Arrange
        pipeline = all_services_loaded
        style = style_variations["editorial"]
        
        # Act
        performance_timer.start()
        result = pipeline.generate_scene_step(
            sample_model_image,
            style["style_prompt"],
            style["background_type"]
        )
        performance_timer.stop()
        
        # Assert
        assert performance_timer.elapsed <= PERFORMANCE_TARGETS["scene_generation_max_time"]
        
        assert "scene_image" in result
        assert "scene_metadata" in result
        assert "scene_quality_score" in result
        
        scene_image = result["scene_image"]
        assert isinstance(scene_image, Image.Image)
        assert result["scene_quality_score"] >= QUALITY_THRESHOLDS["scene_coherence_min"]
    
    def test_multiple_style_variations_quality(
        self,
        all_services_loaded,
        sample_model_image,
        style_variations
    ):
        """Test quality consistency across multiple style variations"""
        # Arrange
        pipeline = all_services_loaded
        
        # Act
        variation_results = []
        for style_name, style_config in style_variations.items():
            result = pipeline.generate_style_variation(
                sample_model_image,
                style_config,
                style_name
            )
            variation_results.append((style_name, result))
        
        # Assert
        assert len(variation_results) == 5  # All styles processed
        
        quality_scores = []
        for style_name, result in variation_results:
            assert "variation_image" in result
            assert "style_consistency" in result
            assert "quality_score" in result
            
            # Individual quality requirements
            assert result["style_consistency"] >= 0.8
            assert result["quality_score"] >= QUALITY_THRESHOLDS["overall_score_min"]
            
            quality_scores.append(result["quality_score"])
        
        # Quality consistency across variations
        avg_quality = np.mean(quality_scores)
        quality_variance = np.var(quality_scores)
        
        assert avg_quality >= QUALITY_THRESHOLDS["overall_score_min"]
        assert quality_variance < 1.0  # Reasonable consistency
    
    def test_pipeline_error_handling(
        self,
        transformation_pipeline,
        quality_metrics
    ):
        """Test pipeline error handling with invalid inputs"""
        # Arrange
        pipeline = transformation_pipeline
        
        # Test with corrupted image
        corrupted_image = Image.new('RGB', (10, 10))  # Too small
        
        # Act
        result = pipeline.transform_model(
            model_image=corrupted_image,
            style_prompt="test style",
            num_variations=1
        )
        
        # Assert - Should handle gracefully
        assert "variations" in result
        assert "error_handling" in result
        assert result["error_handling"]["fallbacks_used"] > 0
        
        # Should still produce output, even if low quality
        assert len(result["variations"]) == 1
        variation = result["variations"][0]
        assert "transformed_image" in variation
    
    def test_memory_management_across_pipeline(
        self,
        all_services_loaded,
        sample_model_image
    ):
        """Test memory management across the entire pipeline"""
        # Arrange
        pipeline = all_services_loaded
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Act - Process multiple images
        for i in range(3):
            result = pipeline.transform_model(
                model_image=sample_model_image,
                style_prompt="test transformation",
                num_variations=2
            )
            assert len(result["variations"]) == 2
        
        # Force cleanup
        pipeline.clear_all_caches()
        
        # Assert
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_growth = final_memory - initial_memory
        
        # Memory should not grow excessively (< 4GB)
        assert memory_growth < 4 * 1024 * 1024 * 1024
    
    def test_concurrent_processing_stability(
        self,
        all_services_loaded,
        sample_model_image
    ):
        """Test stability under concurrent processing"""
        # Arrange
        pipeline = all_services_loaded
        
        # Act - Simulate concurrent requests
        results = []
        for i in range(3):
            result = pipeline.transform_model(
                model_image=sample_model_image,
                style_prompt=f"concurrent test {i}",
                num_variations=2
            )
            results.append(result)
        
        # Assert - All results should be valid
        assert len(results) == 3
        for result in results:
            assert "variations" in result
            assert len(result["variations"]) == 2
            assert result["quality_scores"]["overall_average"] >= 6.0
    
    def test_quality_vs_speed_tradeoff(
        self,
        all_services_loaded,
        sample_model_image,
        performance_timer
    ):
        """Test quality vs speed tradeoff options"""
        # Arrange
        pipeline = all_services_loaded
        
        # Act - Fast mode
        performance_timer.start()
        fast_result = pipeline.transform_model(
            model_image=sample_model_image,
            style_prompt="fast mode test",
            num_variations=3,
            quality_mode="fast"
        )
        fast_time = performance_timer.elapsed
        performance_timer.stop()
        
        # Act - High quality mode
        performance_timer.start()
        quality_result = pipeline.transform_model(
            model_image=sample_model_image,
            style_prompt="quality mode test",
            num_variations=3,
            quality_mode="high"
        )
        quality_time = performance_timer.elapsed
        performance_timer.stop()
        
        # Assert
        # Fast mode should be faster
        assert fast_time <= PERFORMANCE_TARGETS["total_processing_max_time"] * 0.7
        
        # Quality mode should produce higher quality
        fast_avg_quality = fast_result["quality_scores"]["overall_average"]
        quality_avg_quality = quality_result["quality_scores"]["overall_average"]
        
        # Allow for some variance, but quality mode should generally be better
        assert quality_avg_quality >= fast_avg_quality - 0.5
    
    def test_pipeline_scalability(
        self,
        all_services_loaded,
        sample_model_image
    ):
        """Test pipeline scalability with varying numbers of variations"""
        # Arrange
        pipeline = all_services_loaded
        variation_counts = [1, 3, 5]
        
        # Act & Assert
        for count in variation_counts:
            start_time = time.time()
            
            result = pipeline.transform_model(
                model_image=sample_model_image,
                style_prompt="scalability test",
                num_variations=count
            )
            
            processing_time = time.time() - start_time
            
            # Assertions
            assert len(result["variations"]) == count
            
            # Processing time should scale reasonably
            max_expected_time = PERFORMANCE_TARGETS["total_processing_max_time"] * (count / 5)
            assert processing_time <= max_expected_time
            
            # Quality should remain consistent regardless of batch size
            assert result["quality_scores"]["overall_average"] >= QUALITY_THRESHOLDS["overall_score_min"]
    
    def test_pipeline_determinism(
        self,
        all_services_loaded,
        sample_model_image
    ):
        """Test pipeline determinism with same inputs"""
        # Arrange
        pipeline = all_services_loaded
        style_prompt = "determinism test"
        
        # Act - Run same transformation twice
        result1 = pipeline.transform_model(
            model_image=sample_model_image,
            style_prompt=style_prompt,
            num_variations=2,
            seed=42  # Fixed seed
        )
        
        result2 = pipeline.transform_model(
            model_image=sample_model_image,
            style_prompt=style_prompt,
            num_variations=2,
            seed=42  # Same seed
        )
        
        # Assert - Results should be similar (allowing for minor variations)
        assert len(result1["variations"]) == len(result2["variations"])
        
        quality_diff = abs(
            result1["quality_scores"]["overall_average"] - 
            result2["quality_scores"]["overall_average"]
        )
        assert quality_diff < 0.5  # Small variance allowed
    
    def test_progressive_enhancement_pipeline(
        self,
        all_services_loaded,
        sample_model_image,
        performance_timer
    ):
        """Test progressive enhancement through pipeline stages"""
        # Arrange
        pipeline = all_services_loaded
        
        # Act - Track quality at each stage
        performance_timer.start()
        
        # Stage 1: Model enhancement
        enhanced_result = pipeline.enhance_model_step(sample_model_image)
        stage1_quality = pipeline.assess_image_quality(enhanced_result["enhanced_image"])
        
        # Stage 2: Garment optimization
        garment_result = pipeline.optimize_garment_step(enhanced_result["enhanced_image"])
        stage2_quality = pipeline.assess_image_quality(garment_result["optimized_image"])
        
        # Stage 3: Scene generation
        scene_result = pipeline.generate_scene_step(
            garment_result["optimized_image"],
            "professional photography",
            "studio"
        )
        stage3_quality = pipeline.assess_image_quality(scene_result["scene_image"])
        
        performance_timer.stop()
        
        # Assert
        assert performance_timer.elapsed <= PERFORMANCE_TARGETS["total_processing_max_time"]
        
        # Quality should improve or remain stable at each stage
        original_quality = pipeline.assess_image_quality(sample_model_image)
        
        assert stage1_quality >= original_quality  # Model enhancement improves
        assert stage2_quality >= stage1_quality * 0.95  # Allow slight variance
        assert stage3_quality >= stage2_quality * 0.95  # Allow slight variance
        
        # Final quality should meet requirements
        assert stage3_quality >= QUALITY_THRESHOLDS["overall_score_min"]
    
    def test_batch_processing_efficiency(
        self,
        all_services_loaded,
        sample_model_image
    ):
        """Test batch processing efficiency"""
        # Arrange
        pipeline = all_services_loaded
        batch_size = 3
        images = [sample_model_image] * batch_size
        
        # Act
        start_time = time.time()
        
        batch_result = pipeline.transform_batch(
            images=images,
            style_prompts=["batch test"] * batch_size,
            num_variations_per_image=2
        )
        
        batch_time = time.time() - start_time
        
        # Compare with individual processing
        individual_start = time.time()
        
        individual_results = []
        for image in images:
            result = pipeline.transform_model(
                model_image=image,
                style_prompt="individual test",
                num_variations=2
            )
            individual_results.append(result)
        
        individual_time = time.time() - individual_start
        
        # Assert
        assert len(batch_result) == batch_size
        
        # Batch processing should be more efficient
        # Allow some overhead, but should be faster than sequential
        assert batch_time <= individual_time * 0.8
        
        # Quality should be maintained
        for result in batch_result:
            assert result["quality_scores"]["overall_average"] >= QUALITY_THRESHOLDS["overall_score_min"]
    
    @pytest.mark.parametrize("image_size", [(256, 256), (512, 512), (1024, 1024)])
    def test_different_image_sizes(
        self,
        all_services_loaded,
        image_size
    ):
        """Test pipeline with different image sizes"""
        # Arrange
        pipeline = all_services_loaded
        test_image = Image.new('RGB', image_size, color='blue')
        
        # Act
        result = pipeline.transform_model(
            model_image=test_image,
            style_prompt="size test",
            num_variations=2
        )
        
        # Assert
        assert len(result["variations"]) == 2
        
        for variation in result["variations"]:
            transformed_image = variation["transformed_image"]
            assert transformed_image.size == image_size
            assert variation["quality_score"] >= 5.0  # Lower threshold for test images
    
    def test_edge_case_handling_comprehensive(
        self,
        all_services_loaded
    ):
        """Test comprehensive edge case handling"""
        # Arrange
        pipeline = all_services_loaded
        
        edge_cases = [
            # Very small image
            Image.new('RGB', (32, 32), color='red'),
            # Very large image (will be resized)
            Image.new('RGB', (4096, 4096), color='green'),
            # Unusual aspect ratio
            Image.new('RGB', (100, 500), color='blue'),
        ]
        
        # Act & Assert
        for i, edge_image in enumerate(edge_cases):
            result = pipeline.transform_model(
                model_image=edge_image,
                style_prompt=f"edge case {i}",
                num_variations=1
            )
            
            # Should handle gracefully
            assert "variations" in result
            assert len(result["variations"]) == 1
            
            variation = result["variations"][0]
            assert "transformed_image" in variation
            
            # May use fallbacks for extreme cases
            if "fallbacks_used" in result.get("error_handling", {}):
                assert result["error_handling"]["fallbacks_used"] >= 0