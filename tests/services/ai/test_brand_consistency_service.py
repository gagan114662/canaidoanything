import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

try:
    from app.services.ai.brand_consistency_service import BrandConsistencyService
    SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import BrandConsistencyService: {e}")
    BrandConsistencyService = None
    SERVICE_AVAILABLE = False

class TestBrandConsistencyServiceAdvancedColorGrading(unittest.TestCase):

    @patch('app.services.ai.brand_consistency_service.BrandConsistencyService.load_models', new_callable=MagicMock)
    def setUp(self, mock_load_models):
        if not SERVICE_AVAILABLE:
            self.skipTest("BrandConsistencyService not available for testing.")
        self.service = BrandConsistencyService()
        self.service.model_loaded = True # Assume models are loaded

        self.dummy_image = Image.new('RGB', (10, 10), 'red')
        self.processed_image_mock = Image.new('RGB', (10, 10), 'blue') # Different content

    @patch('app.utils.image_utils.apply_lut')
    def test_apply_advanced_color_grade_with_valid_lut(self, mock_apply_lut):
        mock_apply_lut.return_value = self.processed_image_mock

        lut_path = "mock_lut_path.cube"
        result = self.service.apply_advanced_color_grade(self.dummy_image, lut_path)

        mock_apply_lut.assert_called_once_with(unittest.mock.ANY, lut_path) # ANY for the image copy
        self.assertIsNotNone(result["lut_applied"], "lut_applied should be the path if effect occurred.")
        self.assertEqual(result["lut_applied"], lut_path)
        self.assertIs(result["graded_image"], self.processed_image_mock)
        self.assertIsNone(result["error"])

    @patch('app.utils.image_utils.apply_lut')
    def test_apply_advanced_color_grade_no_lut_path(self, mock_apply_lut):
        result = self.service.apply_advanced_color_grade(self.dummy_image, None)

        mock_apply_lut.assert_not_called()
        self.assertIsNone(result["lut_applied"])
        self.assertIs(result["graded_image"], self.dummy_image)
        self.assertIsNone(result["error"])

    @patch('app.utils.image_utils.apply_lut')
    def test_apply_advanced_color_grade_lut_application_fails(self, mock_apply_lut):
        mock_apply_lut.side_effect = Exception("LUT error")
        lut_path = "mock_lut_path.cube"

        result = self.service.apply_advanced_color_grade(self.dummy_image, lut_path)

        mock_apply_lut.assert_called_once()
        self.assertIsNone(result["lut_applied"]) # Should be None as application failed
        self.assertIs(result["graded_image"], self.dummy_image) # Original image returned
        self.assertIsNotNone(result["error"])
        self.assertIn("LUT error", result["error"])

    # Test for the success flag when image doesn't change (mock returns original)
    @patch('app.utils.image_utils.apply_lut')
    def test_apply_advanced_color_grade_lut_no_change(self, mock_apply_lut):
        # Simulate apply_lut returning the same image (no change)
        mock_apply_lut.return_value = self.dummy_image

        lut_path = "another_mock_lut.cube"
        result = self.service.apply_advanced_color_grade(self.dummy_image, lut_path)

        mock_apply_lut.assert_called_once()
        self.assertIsNone(result["lut_applied"], "lut_applied should be None if image is unchanged.")
        self.assertIs(result["graded_image"], self.dummy_image)
        self.assertIsNone(result["error"])


if __name__ == '__main__':
    unittest.main()
