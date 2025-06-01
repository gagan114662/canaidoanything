import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

# Attempt to import services and pipeline
try:
    from app.services.tasks.model_transformation import ModelTransformationPipeline
    from app.services.ai.model_enhancement_service import ModelEnhancementService
    from app.services.ai.garment_optimization_service import GarmentOptimizationService
    from app.services.ai.scene_generation_service import SceneGenerationService
    SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import services for ModelTransformationPipeline tests: {e}")
    ModelTransformationPipeline = None
    ModelEnhancementService = None
    GarmentOptimizationService = None
    SceneGenerationService = None
    SERVICES_AVAILABLE = False

class TestModelTransformationPipelineNewStyles(unittest.TestCase):

    @patch('app.services.tasks.model_transformation.SceneGenerationService')
    @patch('app.services.tasks.model_transformation.GarmentOptimizationService')
    @patch('app.services.tasks.model_transformation.ModelEnhancementService')
    def setUp(self, MockModelEnhancement, MockGarmentOptimization, MockSceneGeneration):
        if not SERVICES_AVAILABLE:
            self.skipTest("Required services for ModelTransformationPipeline not available.")

        # Store mocks
        self.mock_model_enhancement_service = MockModelEnhancement.return_value
        self.mock_garment_optimization_service = MockGarmentOptimization.return_value
        self.mock_scene_generation_service = MockSceneGeneration.return_value

        # Instantiate the pipeline
        self.pipeline = ModelTransformationPipeline()
        self.pipeline.services_loaded = True # Assume models are loaded within services

        # Replace actual service instances with mocks
        self.pipeline.model_enhancement_service = self.mock_model_enhancement_service
        self.pipeline.garment_optimization_service = self.mock_garment_optimization_service
        self.pipeline.scene_generation_service = self.mock_scene_generation_service

        # Configure mock return values for service methods
        # These are simplified and assume the image object is passed through
        self.dummy_image = Image.new('RGB', (100, 100), 'red')

        self.mock_model_enhancement_service.enhance_model.return_value = {
            'enhanced_image': self.dummy_image,
            'quality_improvement': 0.1
        }
        self.mock_garment_optimization_service.optimize_garment.return_value = {
            'optimized_image': self.dummy_image,
            'overall_score': 8.0
        }
        # generate_scene_step calls compose_scene and apply_lighting.
        # So, mock what generate_scene_step would return, or mock deeper.
        # For these tests, we care about what ModelTransformationPipeline passes to SceneGenerationService methods.
        self.mock_scene_generation_service.compose_scene.return_value = {
            'composed_scene': self.dummy_image,
            'composition_score': 8.0,
            'background_coherence': 0.8,
            # Add other keys that might be accessed by the pipeline
            'coherence_score': 0.8, # if generate_background output is directly used
            'style_match': 0.8,
            'background_type_match': 0.9,
            'processing_time': 0.1,
            'generation_method': "FLUX"
        }
        self.mock_scene_generation_service.apply_lighting.return_value = {
            'lit_image': self.dummy_image,
            'lighting_quality': 0.8
        }
        # Mock methods used by assess_image_quality if it's complex
        # For now, assuming assess_image_quality is simple or we don't care about its exact output
        # self.pipeline.assess_image_quality = MagicMock(return_value=8.0)
        # Alternatively, mock the cv2 import if assess_image_quality uses it and cv2 is not available
        # For now, let's assume a default quality will be fine or mock assess_image_quality directly if needed

    def test_pipeline_initialization_with_mocks(self):
        """Test that the pipeline initializes correctly with mocked services."""
        self.assertIsNotNone(self.pipeline)
        self.assertIsInstance(self.pipeline.model_enhancement_service, MagicMock)
        self.assertIsInstance(self.pipeline.garment_optimization_service, MagicMock)
        self.assertIsInstance(self.pipeline.scene_generation_service, MagicMock)

    def test_new_artistic_style_configs_run_successfully(self):
        """
        Test that the pipeline can run with the new artistic style configurations
        and that specific parameters like background_type are passed correctly.
        """
        new_styles_to_test = ["surreal_fashion", "painterly_portrait", "avant_garde_design"]
        dummy_input_image = Image.new('RGB', (200, 300), 'blue') # Different size for clarity

        for style_name in new_styles_to_test:
            # Reset mock call counts for each style if checking per-style calls specifically
            self.mock_scene_generation_service.compose_scene.reset_mock()

            # The 'style_prompt' passed to transform_model is used to select the style_config,
            # and can also append additional user prompts. Here, we use it primarily to select the config.
            result = self.pipeline.transform_model(
                model_image=dummy_input_image,
                style_prompt=style_name, # This will be used to pick the style config by key
                num_variations=1, # Test with 1 variation for simplicity
                quality_mode="fast" # Use fast mode to simplify some processing paths
            )

            self.assertIsNotNone(result, f"Result should not be None for style {style_name}")
            self.assertIn("variations", result, f"Result should have 'variations' for style {style_name}")
            self.assertEqual(len(result["variations"]), 1, f"Should generate 1 variation for style {style_name}")

            variation = result["variations"][0]
            self.assertEqual(variation["style_type"], style_name, f"Variation style_type should match for {style_name}")
            self.assertIsInstance(variation["variation_image"], Image.Image, f"Variation image should be a PIL Image for {style_name}")
            self.assertGreater(variation["quality_score"], 0, f"Quality score should be > 0 for {style_name}")

            # Verify that scene generation was called
            self.mock_scene_generation_service.compose_scene.assert_called()

            # Specific checks for "surreal_fashion"
            if style_name == "surreal_fashion":
                # compose_scene(model_image, background_type, composition_style, creativity_level, control_image, scale, rule_name)
                called_kwargs = self.mock_scene_generation_service.compose_scene.call_args.kwargs
                self.assertEqual(called_kwargs.get("background_type"), "surreal_dreamscape")

            # Could add similar checks for other styles if they have unique, testable params passed to mocks

    def test_creativity_level_passed_to_scene_generation(self):
        """
        Test that the creativity_level from the style_config is correctly passed
        to the scene_generation_service.compose_scene method.
        """
        dummy_input_image = Image.new('RGB', (200, 300), 'green')
        style_name_to_test = "surreal_fashion" # This style has creativity_level: 0.95

        # Expected creativity level from the style_configs dictionary for "surreal_fashion"
        expected_creativity = self.pipeline.style_configs[style_name_to_test]["creativity_level"] # Should be 0.95

        self.mock_scene_generation_service.compose_scene.reset_mock()

        _ = self.pipeline.transform_model(
            model_image=dummy_input_image,
            style_prompt=style_name_to_test,
            num_variations=1,
            quality_mode="fast"
        )

        self.mock_scene_generation_service.compose_scene.assert_called_once()

        # Check the keyword arguments passed to compose_scene
        called_kwargs = self.mock_scene_generation_service.compose_scene.call_args.kwargs

        self.assertIn('creativity_level', called_kwargs, "creativity_level should be a keyword argument to compose_scene")
        self.assertEqual(
            called_kwargs['creativity_level'],
            expected_creativity,
            f"creativity_level passed to compose_scene ({called_kwargs['creativity_level']}) "
            f"did not match expected ({expected_creativity}) for style {style_name_to_test}"
        )

if __name__ == '__main__':
    unittest.main()
