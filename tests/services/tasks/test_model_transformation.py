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

    @patch('app.services.tasks.model_transformation.BrandConsistencyService') # Added
    @patch('app.services.tasks.model_transformation.SceneGenerationService')
    @patch('app.services.tasks.model_transformation.GarmentOptimizationService')
    @patch('app.services.tasks.model_transformation.ModelEnhancementService')
    def setUp(self, MockModelEnhancement, MockGarmentOptimization, MockSceneGeneration, MockBrandConsistency): # Added
        if not SERVICES_AVAILABLE:
            self.skipTest("Required services for ModelTransformationPipeline not available.")

        # Store mocks
        self.mock_model_enhancement_service = MockModelEnhancement.return_value
        self.mock_garment_optimization_service = MockGarmentOptimization.return_value
        self.mock_scene_generation_service = MockSceneGeneration.return_value
        self.mock_brand_consistency_service = MockBrandConsistency.return_value # Added

        # Instantiate the pipeline
        self.pipeline = ModelTransformationPipeline()
        self.pipeline.services_loaded = True # Assume models are loaded within services

        # Replace actual service instances with mocks
        self.pipeline.model_enhancement_service = self.mock_model_enhancement_service
        self.pipeline.garment_optimization_service = self.mock_garment_optimization_service
        self.pipeline.scene_generation_service = self.mock_scene_generation_service
        self.pipeline.brand_consistency_service = self.mock_brand_consistency_service # Added

        # Configure mock return values for service methods
        self.dummy_image = Image.new('RGB', (100, 100), 'red')
        self.processed_image_mock = Image.new('RGB', (100, 100), 'blue') # For simulating change

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
        self.mock_scene_generation_service.suggest_crop.return_value = {
            'x': 0, 'y': 0, 'width': 100, 'height': 100, 'rule_applied': 'saliency_basic'
        }

        # New post-processing mocks
        self.mock_brand_consistency_service.apply_advanced_color_grade.return_value = {
            'graded_image': self.processed_image_mock, 'lut_applied': 'brand_lut.cube', 'success': True
            # Assuming 'success' key for brand consistency, will align with actual implementation
        }
        self.mock_garment_optimization_service.apply_cinematic_color_grade.return_value = {
            'graded_image': self.processed_image_mock, 'grade_applied': 'vintage'
        }
        self.mock_model_enhancement_service.apply_ai_denoising.return_value = {
            'processed_image': self.processed_image_mock, 'success': True
        }
        self.mock_garment_optimization_service.apply_ai_garment_denoising.return_value = {
            'processed_image': self.processed_image_mock, 'success': True
        }
        self.mock_model_enhancement_service.apply_ai_sharpening.return_value = {
            'processed_image': self.processed_image_mock, 'success': True
        }
        self.mock_garment_optimization_service.apply_ai_fabric_sharpening.return_value = {
            'processed_image': self.processed_image_mock, 'success': True
        }
        self.mock_model_enhancement_service.apply_ai_model_detail_enhancement.return_value = {
            'processed_image': self.processed_image_mock, 'face_enhanced': True, 'accessory_enhanced': True
        }
        self.mock_garment_optimization_service.apply_ai_fabric_detail_enhancement.return_value = {
            'processed_image': self.processed_image_mock, 'success': True
        }
        # self.pipeline.assess_image_quality = MagicMock(return_value=8.0) # Keep if needed

    def test_pipeline_initialization_with_mocks(self):
        """Test that the pipeline initializes correctly with mocked services."""
        self.assertIsNotNone(self.pipeline)
        self.assertIsInstance(self.pipeline.model_enhancement_service, MagicMock)
        self.assertIsInstance(self.pipeline.garment_optimization_service, MagicMock)
        self.assertIsInstance(self.pipeline.scene_generation_service, MagicMock)
        self.assertIsInstance(self.pipeline.brand_consistency_service, MagicMock) # Added

    def test_pipeline_applies_brand_color_grade(self):
        self.mock_brand_consistency_service.apply_advanced_color_grade.return_value = {
            'graded_image': self.processed_image_mock,
            'lut_applied': "brand_lut.cube", # Simulating successful application
            'error': None
        }
        result = self.pipeline.transform_model(
            model_image=self.dummy_image,
            style_prompt="editorial",
            num_variations=1,
            apply_brand_color_grade_lut="brand_lut.cube"
        )
        self.mock_brand_consistency_service.apply_advanced_color_grade.assert_called_once_with(
            unittest.mock.ANY, # The image passed can be a copy
            "brand_lut.cube"
        )
        self.assertIn("brand_lut:brand_lut.cube", result["variations"][0]["post_processing_applied"])
        self.assertIs(result["variations"][0]["variation_image"], self.processed_image_mock)

    def test_pipeline_applies_cinematic_color_grade(self):
        self.mock_garment_optimization_service.apply_cinematic_color_grade.return_value = {
            'graded_image': self.processed_image_mock,
            'grade_applied': 'vintage',
            'lut_path': 'luts/cinematic/vintage_look.cube',
            'error': None
        }
        result = self.pipeline.transform_model(
            model_image=self.dummy_image,
            style_prompt="editorial",
            num_variations=1,
            apply_cinematic_color_grade_style="vintage"
        )
        self.mock_garment_optimization_service.apply_cinematic_color_grade.assert_called_once_with(
            unittest.mock.ANY,
            "vintage"
        )
        self.assertIn("cinematic_lut:vintage", result["variations"][0]["post_processing_applied"])
        self.assertIs(result["variations"][0]["variation_image"], self.processed_image_mock)

    def test_pipeline_uses_ai_smart_denoising(self):
        # Ensure mocks return different images to simulate change
        denoised_model_img = Image.new('RGB', (100,100), 'gray')
        denoised_garment_img = Image.new('RGB', (100,100), 'darkgray')
        self.mock_model_enhancement_service.apply_ai_denoising.return_value = {'processed_image': denoised_model_img, 'success': True}
        self.mock_garment_optimization_service.apply_ai_garment_denoising.return_value = {'processed_image': denoised_garment_img, 'success': True}

        result = self.pipeline.transform_model(
            model_image=self.dummy_image,
            style_prompt="editorial",
            num_variations=1,
            use_ai_smart_denoising=True
        )
        self.mock_model_enhancement_service.apply_ai_denoising.assert_called_once_with(unittest.mock.ANY, strength=0.5)
        # The garment denoising is called with the result of model denoising
        self.mock_garment_optimization_service.apply_ai_garment_denoising.assert_called_once_with(denoised_model_img, strength=0.5)
        self.assertIn("ai_denoise", result["variations"][0]["post_processing_applied"])
        self.assertIs(result["variations"][0]["variation_image"], denoised_garment_img)

    def test_pipeline_uses_ai_smart_sharpening(self):
        sharpened_model_img = Image.new('RGB', (100,100), 'lightgray')
        sharpened_fabric_img = Image.new('RGB', (100,100), 'silver')
        self.mock_model_enhancement_service.apply_ai_sharpening.return_value = {'processed_image': sharpened_model_img, 'success': True}
        self.mock_garment_optimization_service.apply_ai_fabric_sharpening.return_value = {'processed_image': sharpened_fabric_img, 'success': True}

        result = self.pipeline.transform_model(
            model_image=self.dummy_image,
            style_prompt="editorial",
            num_variations=1,
            use_ai_smart_sharpening=True
        )
        self.mock_model_enhancement_service.apply_ai_sharpening.assert_called_once_with(unittest.mock.ANY, intensity=0.5)
        self.mock_garment_optimization_service.apply_ai_fabric_sharpening.assert_called_once_with(sharpened_model_img, intensity=0.5)
        self.assertIn("ai_sharpen", result["variations"][0]["post_processing_applied"])
        self.assertIs(result["variations"][0]["variation_image"], sharpened_fabric_img)

    def test_pipeline_uses_ai_detail_enhancement(self):
        enhanced_model_img = Image.new('RGB', (100,100), 'pink')
        enhanced_fabric_img = Image.new('RGB', (100,100), 'purple')
        self.mock_model_enhancement_service.apply_ai_model_detail_enhancement.return_value = {
            'processed_image': enhanced_model_img, 'face_enhanced': True, 'accessory_enhanced': True
        }
        self.mock_garment_optimization_service.apply_ai_fabric_detail_enhancement.return_value = {
            'processed_image': enhanced_fabric_img, 'success': True
        }

        result = self.pipeline.transform_model(
            model_image=self.dummy_image,
            style_prompt="editorial",
            num_variations=1,
            use_ai_detail_enhancement=True
        )
        self.mock_model_enhancement_service.apply_ai_model_detail_enhancement.assert_called_once_with(
            unittest.mock.ANY, face_intensity=0.5, accessory_intensity=0.5
        )
        self.mock_garment_optimization_service.apply_ai_fabric_detail_enhancement.assert_called_once_with(
            enhanced_model_img, intensity=0.5
        )
        self.assertIn("ai_detail_enhance", result["variations"][0]["post_processing_applied"])
        self.assertIs(result["variations"][0]["variation_image"], enhanced_fabric_img)

    def test_pipeline_requests_composition_suggestion(self):
        mock_suggestion = {'x':10, 'y':10, 'width':80, 'height':80, 'rule_applied': 'saliency_basic'}
        self.mock_scene_generation_service.suggest_crop.return_value = mock_suggestion

        result = self.pipeline.transform_model(
            model_image=self.dummy_image,
            style_prompt="editorial",
            num_variations=1,
            request_composition_suggestion=True,
            composition_rule="rule_of_thirds" # Also pass a rule to suggest_crop
        )
        self.mock_scene_generation_service.suggest_crop.assert_called_once_with(
            unittest.mock.ANY, # The image passed can be a copy or processed version
            "rule_of_thirds"
        )
        self.assertIn("suggested_crop_coordinates", result["variations"][0])
        self.assertEqual(result["variations"][0]["suggested_crop_coordinates"], mock_suggestion)

    def test_pipeline_applies_multiple_post_processing_steps(self):
        # Mock return values for all services that will be called
        self.mock_garment_optimization_service.apply_cinematic_color_grade.return_value = {
            'graded_image': self.processed_image_mock, 'grade_applied': 'vintage'
        }
        # Simulate denoising returns a new image instance for chaining
        denoised_model_img = Image.new('RGB', (100,100), 'gray')
        denoised_garment_img = Image.new('RGB', (100,100), 'darkgray')
        self.mock_model_enhancement_service.apply_ai_denoising.return_value = {'processed_image': denoised_model_img, 'success': True}
        self.mock_garment_optimization_service.apply_ai_garment_denoising.return_value = {'processed_image': denoised_garment_img, 'success': True}

        self.mock_scene_generation_service.suggest_crop.return_value = {'x':0, 'y':0, 'width':10, 'height':10}


        result = self.pipeline.transform_model(
            model_image=self.dummy_image,
            style_prompt="editorial",
            num_variations=1,
            apply_cinematic_color_grade_style="vintage",
            use_ai_smart_denoising=True,
            request_composition_suggestion=True,
            composition_rule="golden_ratio"
        )

        self.mock_garment_optimization_service.apply_cinematic_color_grade.assert_called_once()
        self.mock_model_enhancement_service.apply_ai_denoising.assert_called_once()
        self.mock_garment_optimization_service.apply_ai_garment_denoising.assert_called_once()
        self.mock_scene_generation_service.suggest_crop.assert_called_once_with(unittest.mock.ANY, "golden_ratio")

        applied_steps = result["variations"][0]["post_processing_applied"]
        self.assertIn("cinematic_lut:vintage", applied_steps)
        self.assertIn("ai_denoise", applied_steps)
        self.assertIn("suggested_crop_coordinates", result["variations"][0])
        # Check final image is the one from the last step in the chain (denoising in this case)
        self.assertIs(result["variations"][0]["variation_image"], denoised_garment_img)

    def test_pipeline_no_post_processing_by_default(self):
        result = self.pipeline.transform_model(
            model_image=self.dummy_image,
            style_prompt="editorial",
            num_variations=1,
            # All post-processing flags are default (None or False)
        )

        self.mock_brand_consistency_service.apply_advanced_color_grade.assert_not_called()
        self.mock_garment_optimization_service.apply_cinematic_color_grade.assert_not_called()
        self.mock_model_enhancement_service.apply_ai_denoising.assert_not_called()
        self.mock_garment_optimization_service.apply_ai_garment_denoising.assert_not_called()
        self.mock_model_enhancement_service.apply_ai_sharpening.assert_not_called()
        self.mock_garment_optimization_service.apply_ai_fabric_sharpening.assert_not_called()
        self.mock_model_enhancement_service.apply_ai_model_detail_enhancement.assert_not_called()
        self.mock_garment_optimization_service.apply_ai_fabric_detail_enhancement.assert_not_called()
        self.mock_scene_generation_service.suggest_crop.assert_not_called()

        self.assertEqual(len(result["variations"][0].get("post_processing_applied", [])), 0)
        self.assertNotIn("suggested_crop_coordinates", result["variations"][0])


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
