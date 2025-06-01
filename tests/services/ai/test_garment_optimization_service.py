import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

try:
    from app.services.ai.garment_optimization_service import GarmentOptimizationService
    SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import GarmentOptimizationService: {e}")
    GarmentOptimizationService = None
    SERVICE_AVAILABLE = False

class TestGarmentOptimizationServicePostProcessing(unittest.TestCase):

    @patch('app.services.ai.garment_optimization_service.GarmentOptimizationService.load_models', new_callable=MagicMock)
    def setUp(self, mock_load_models):
        if not SERVICE_AVAILABLE:
            self.skipTest("GarmentOptimizationService not available for testing.")

        self.service = GarmentOptimizationService()
        self.service.model_loaded = True # Assume models are loaded

        self.dummy_image = Image.new('RGB', (20, 20), 'green')
        self.processed_image_mock = Image.new('RGB', (20, 20), 'yellow') # Different content

    def test_service_initialization(self):
        """Test that the service can be initialized."""
        self.assertIsNotNone(self.service)

    @patch('app.utils.image_utils.apply_lut')
    def test_apply_cinematic_color_grade_valid_style(self, mock_apply_lut):
        mock_apply_lut.return_value = self.processed_image_mock
        style_name = "vintage" # Assumes "vintage" is in self.service.cinematic_luts
        expected_lut_path = self.service.cinematic_luts[style_name]

        result = self.service.apply_cinematic_color_grade(self.dummy_image, style_name)

        mock_apply_lut.assert_called_once_with(unittest.mock.ANY, expected_lut_path)
        self.assertEqual(result["grade_applied"], style_name)
        self.assertEqual(result["lut_path"], expected_lut_path)
        self.assertIs(result["graded_image"], self.processed_image_mock)
        self.assertIsNone(result["error"])

    @patch('app.utils.image_utils.apply_lut')
    def test_apply_cinematic_color_grade_invalid_style(self, mock_apply_lut):
        style_name = "non_existent_style"
        result = self.service.apply_cinematic_color_grade(self.dummy_image, style_name)

        mock_apply_lut.assert_not_called()
        self.assertIsNone(result["grade_applied"])
        self.assertIs(result["graded_image"], self.dummy_image)
        self.assertIsNotNone(result["error"])
        self.assertIn(f"Grade style '{style_name}' not found", result["error"])

    @patch('app.utils.image_utils.ai_smart_sharpen')
    def test_apply_ai_fabric_sharpening(self, mock_ai_smart_sharpen):
        mock_ai_smart_sharpen.return_value = self.processed_image_mock
        intensity = 0.7

        result = self.service.apply_ai_fabric_sharpening(self.dummy_image, intensity=intensity)

        mock_ai_smart_sharpen.assert_called_once_with(unittest.mock.ANY, intensity=intensity)
        self.assertTrue(result["success"])
        self.assertEqual(result["operation"], "ai_fabric_sharpen")
        self.assertEqual(result["intensity"], intensity)
        self.assertIs(result["processed_image"], self.processed_image_mock)

    @patch('app.utils.image_utils.ai_smart_denoise')
    def test_apply_ai_garment_denoising(self, mock_ai_smart_denoise):
        mock_ai_smart_denoise.return_value = self.processed_image_mock
        strength = 0.6

        result = self.service.apply_ai_garment_denoising(self.dummy_image, strength=strength)

        mock_ai_smart_denoise.assert_called_once_with(unittest.mock.ANY, strength=strength)
        self.assertTrue(result["success"])
        self.assertEqual(result["operation"], "ai_garment_denoise")
        self.assertEqual(result["strength"], strength)
        self.assertIs(result["processed_image"], self.processed_image_mock)

    @patch('app.utils.image_utils.ai_enhance_fabric_texture')
    def test_apply_ai_fabric_detail_enhancement(self, mock_ai_enhance_fabric_texture):
        mock_ai_enhance_fabric_texture.return_value = self.processed_image_mock
        intensity = 0.8

        result = self.service.apply_ai_fabric_detail_enhancement(self.dummy_image, intensity=intensity)

        mock_ai_enhance_fabric_texture.assert_called_once_with(unittest.mock.ANY, intensity=intensity)
        self.assertTrue(result["success"])
        self.assertEqual(result["operation"], "ai_fabric_detail_enhance")
        self.assertEqual(result["intensity"], intensity)
        self.assertIs(result["processed_image"], self.processed_image_mock)

if __name__ == '__main__':
    unittest.main()
