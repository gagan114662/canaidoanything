import unittest
from PIL import Image, ImageDraw

# Attempt to import SceneGenerationService, handling potential import issues for testing
try:
    from app.services.ai.scene_generation_service import SceneGenerationService
except ImportError as e:
    # This might happen if PYTHONPATH isn't correctly set for tests in this environment
    # or if there are complex dependencies not easily mocked.
    # For now, we'll print a warning and proceed. Tests might fail if the class can't be imported.
    print(f"Warning: Could not import SceneGenerationService: {e}")
    SceneGenerationService = None

class TestSceneGenerationServiceCompositionRules(unittest.TestCase):

    def setUp(self):
        """Set up for each test method."""
        if SceneGenerationService is None:
            self.skipTest("SceneGenerationService could not be imported.")
        self.service = SceneGenerationService()
        # Potentially mock self.service.logger if it's heavily used and not configured
        # For these specific methods, it's mostly used for warnings on small images.

    def _create_dummy_image(self, width: int, height: int, color_left=(255, 0, 0), color_right=None) -> Image.Image:
        """Creates a simple PIL Image for testing.
        If color_right is specified, the image will have two distinct halves.
        """
        image = Image.new('RGB', (width, height), color_left)
        if color_right:
            draw = ImageDraw.Draw(image)
            draw.rectangle([(width // 2, 0), (width, height)], fill=color_right)
        return image

    def test_initialization(self):
        """Test that the service can be initialized."""
        self.assertIsNotNone(self.service, "SceneGenerationService should initialize.")
        self.assertTrue(hasattr(self.service, '_apply_rule_of_thirds'))
        self.assertTrue(hasattr(self.service, '_apply_golden_ratio'))
        self.assertTrue(hasattr(self.service, '_apply_symmetry'))
        self.assertTrue(hasattr(self.service, '_apply_leading_lines'))

    def test_apply_rule_of_thirds(self):
        """Test the rule of thirds application."""
        original_width, original_height = 300, 200
        test_image = self._create_dummy_image(original_width, original_height)

        processed_image = self.service._apply_rule_of_thirds(test_image)

        self.assertIsNotNone(processed_image)
        self.assertIsInstance(processed_image, Image.Image)

        expected_width = int(original_width * 2 / 3) # 200
        expected_height = int(original_height * 2 / 3) # 133

        self.assertEqual(processed_image.width, expected_width)
        self.assertEqual(processed_image.height, expected_height)

        # Check if it's a different image (cropped)
        self.assertNotEqual(processed_image.size, test_image.size)

        # Test with a very small image (should return original or a very small crop)
        small_image = self._create_dummy_image(10, 6)
        processed_small_image = self.service._apply_rule_of_thirds(small_image)
        self.assertIsNotNone(processed_small_image)
        # Expected: 2/3 of 10 is 6, 2/3 of 6 is 4. So, 6x4.
        self.assertEqual(processed_small_image.width, 6) # int(10*2/3)
        self.assertEqual(processed_small_image.height, 4) # int(6*2/3)

        # Test with an image that might cause crop box to go out of bounds if not handled
        # Top-left power point is (30/3, 30/3) = (10,10)
        # New size is (20,20). Crop center (10,10).
        # Crop box: (10-10, 10-10, 10+10, 10+10) -> (0,0,20,20)
        square_image = self._create_dummy_image(30,30)
        processed_square = self.service._apply_rule_of_thirds(square_image)
        self.assertEqual(processed_square.size, (20,20))

    def test_apply_golden_ratio(self):
        """Test the golden ratio application."""
        golden_ratio_val = 1.618

        # Test landscape image (300x200)
        # Expected: new_width = 200 * 1.618 = 323.6 -> too wide.
        # So, new_width = 300, new_height = 300 / 1.618 = 185.4 -> (300, 185)
        landscape_image = self._create_dummy_image(300, 200)
        processed_landscape = self.service._apply_golden_ratio(landscape_image)
        self.assertIsNotNone(processed_landscape)
        self.assertEqual(processed_landscape.width, 300)
        self.assertEqual(processed_landscape.height, int(300 / golden_ratio_val)) # 185

        # Test portrait image (200x300)
        # Expected: new_height = 200 * 1.618 = 323.6 -> too tall.
        # So, new_height = 300, new_width = 300 / 1.618 = 185.4 -> (185, 300)
        portrait_image = self._create_dummy_image(200, 300)
        processed_portrait = self.service._apply_golden_ratio(portrait_image)
        self.assertIsNotNone(processed_portrait)
        self.assertEqual(processed_portrait.width, int(300 / golden_ratio_val)) # 185
        self.assertEqual(processed_portrait.height, 300)

        # Test image that is already golden ratio (approx)
        # e.g. width = 324, height = 200 (324/200 = 1.62)
        already_golden_img = self._create_dummy_image(324, 200)
        processed_already_golden = self.service._apply_golden_ratio(already_golden_img)
        self.assertIsNotNone(processed_already_golden)
        # new_width = 200 * 1.618 = 323.6 -> int(323.6) = 323. This is <= 324.
        self.assertEqual(processed_already_golden.width, int(200 * golden_ratio_val))
        self.assertEqual(processed_already_golden.height, 200)

        # Test with a very small image
        small_image = self._create_dummy_image(10, 6) # landscape
        # Expected: new_width = 6 * 1.618 = 9.7 -> 9. This is <= 10.
        # So, (9,6)
        processed_small = self.service._apply_golden_ratio(small_image)
        self.assertIsNotNone(processed_small)
        self.assertEqual(processed_small.width, int(6 * golden_ratio_val)) #9
        self.assertEqual(processed_small.height, 6)

    def test_apply_symmetry(self):
        """Test the symmetry application."""
        original_width, original_height = 300, 200
        # Create an image with distinct left (red) and right (blue) halves
        test_image = self._create_dummy_image(original_width, original_height, color_left=(255,0,0), color_right=(0,0,255))

        processed_image = self.service._apply_symmetry(test_image)

        self.assertIsNotNone(processed_image)
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, test_image.size) # Dimensions should remain the same

        # The entire image should now look like the left half reflected.
        # So, the left half should be red, and the right half should also be red.
        # Let's sample some pixels.
        # Left half, e.g., at (width/4, height/2) should be red.
        left_sample = processed_image.getpixel((original_width // 4, original_height // 2))
        self.assertEqual(left_sample, (255,0,0), "Pixel in left half should be from original left (red)")

        # Right half, e.g., at (width*3/4, height/2) should also be red (reflection of left).
        right_sample = processed_image.getpixel((original_width * 3 // 4, original_height // 2))
        self.assertEqual(right_sample, (255,0,0), "Pixel in right half should be reflection of left (red)")

        # Test with an odd width to ensure it handles it gracefully
        odd_width_image = self._create_dummy_image(301, 200, color_left=(0,255,0), color_right=(0,0,255))
        processed_odd_width = self.service._apply_symmetry(odd_width_image)
        self.assertIsNotNone(processed_odd_width)
        self.assertEqual(processed_odd_width.size, odd_width_image.size)
        left_sample_odd = processed_odd_width.getpixel((301 // 4, 100))
        self.assertEqual(left_sample_odd, (0,255,0))
        right_sample_odd = processed_odd_width.getpixel((301 * 3 // 4, 100))
        self.assertEqual(right_sample_odd, (0,255,0)) # Should be green

        # Test with a very narrow image (width < 2), should return original
        narrow_image = self._create_dummy_image(1, 100)
        processed_narrow = self.service._apply_symmetry(narrow_image)
        self.assertIs(processed_narrow, narrow_image) # Expecting original image instance back

        narrow_image_2 = self._create_dummy_image(0, 100)
        processed_narrow_2 = self.service._apply_symmetry(narrow_image_2)
        self.assertIs(processed_narrow_2, narrow_image_2)

    def test_apply_leading_lines(self):
        """Test the leading lines application (placeholder)."""
        test_image = self._create_dummy_image(300, 200)

        # As _apply_leading_lines is a placeholder, it should return the image as is.
        processed_image = self.service._apply_leading_lines(test_image)

        self.assertIsNotNone(processed_image)
        self.assertIsInstance(processed_image, Image.Image)
        self.assertIs(processed_image, test_image, "Leading lines should return the original image instance.")


if __name__ == '__main__':
    unittest.main()


# New Test Class for Backgrounds
from unittest.mock import patch, MagicMock

class TestSceneGenerationServiceBackgrounds(unittest.TestCase):

    @patch('app.services.ai.scene_generation_service.SceneGenerationService.load_models', new_callable=MagicMock)
    def setUp(self, mock_load_models):
        if SceneGenerationService is None:
            self.skipTest("SceneGenerationService could not be imported.")
        self.service = SceneGenerationService()
        self.service.model_loaded = True # Assume models are loaded
        self.service.flux_pipeline = MagicMock() # Simulate FLUX pipeline is available

    @patch('app.services.ai.scene_generation_service.SceneGenerationService._generate_fallback_background', return_value=Image.new('RGB', (100, 100), 'green'))
    @patch('app.services.ai.scene_generation_service.SceneGenerationService._generate_flux_background', return_value=Image.new('RGB', (100, 100), 'blue'))
    def test_new_background_templates_generate_successfully(self, mock_flux_gen, mock_fallback_gen):
        """Test that new background templates call _generate_flux_background."""
        new_templates = [
            "surreal_dreamscape",
            "baroque_opulence",
            "cyberpunk_alley",
            "ethereal_forest",
            "cosmic_expanse"
        ]
        dummy_ref_image = Image.new('RGB', (100, 100), 'red')

        for template_name in new_templates:
            result = self.service.generate_background(
                reference_image=dummy_ref_image,
                style_prompt="test style",
                background_type=template_name
            )
            self.assertIsNotNone(result)
            self.assertIn("background_image", result)
            self.assertIsInstance(result["background_image"], Image.Image)
            # Check that the mock FLUX output was used
            self.assertEqual(result["background_image"].getpixel((0,0)), (0,0,255)) # Blue from mock_flux_gen

        self.assertTrue(mock_flux_gen.called)
        self.assertEqual(mock_flux_gen.call_count, len(new_templates))
        mock_fallback_gen.assert_not_called()

    @patch('app.services.ai.scene_generation_service.SceneGenerationService._generate_fallback_background', return_value=Image.new('RGB', (100, 100), 'green'))
    @patch('app.services.ai.scene_generation_service.SceneGenerationService._generate_flux_background', return_value=Image.new('RGB', (100, 100), 'blue'))
    def test_invalid_background_template_uses_default(self, mock_flux_gen, mock_fallback_gen):
        """Test that an invalid background template uses the default 'studio' prompt via FLUX."""
        dummy_ref_image = Image.new('RGB', (100, 100), 'red')

        result = self.service.generate_background(
            reference_image=dummy_ref_image,
            style_prompt="test style",
            background_type="non_existent_template_type_blah_blah"
        )
        self.assertIsNotNone(result)
        self.assertIn("background_image", result)
        self.assertEqual(result["background_image"].getpixel((0,0)), (0,0,255)) # Blue from mock_flux_gen

        self.assertTrue(mock_flux_gen.called)
        self.assertEqual(mock_flux_gen.call_count, 1)

        # Check that the prompt passed to _generate_flux_background was the default studio prompt
        # The actual call is _generate_flux_background(self, reference_image, prompt, negative_prompt, ...)
        # So, args[0] is self, args[1] is reference_image, args[2] is prompt
        # Or using keyword args if available: call_args.kwargs['prompt']

        # Check if call_args has kwargs (it should if called with keyword arguments)
        if mock_flux_gen.call_args.kwargs:
            called_prompt = mock_flux_gen.call_args.kwargs.get('prompt', '')
        else: # else access by position
            called_prompt = mock_flux_gen.call_args.args[2]

        self.assertIn("professional photography studio", called_prompt)
        self.assertIn("clean white background", called_prompt)
        mock_fallback_gen.assert_not_called()