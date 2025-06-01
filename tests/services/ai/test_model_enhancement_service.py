import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

try:
    from app.services.ai.model_enhancement_service import ModelEnhancementService
    SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ModelEnhancementService: {e}")
    ModelEnhancementService = None
    SERVICE_AVAILABLE = False

class TestModelEnhancementServicePostProcessing(unittest.TestCase):

    @patch('app.services.ai.model_enhancement_service.ModelEnhancementService.load_models', new_callable=MagicMock)
    def setUp(self, mock_load_models):
        if not SERVICE_AVAILABLE:
            self.skipTest("ModelEnhancementService not available for testing.")

        self.service = ModelEnhancementService()
        self.service.model_loaded = True # Assume models are loaded

        self.dummy_image = Image.new('RGB', (30, 30), 'blue')
        self.processed_image_mock = Image.new('RGB', (30, 30), 'cyan') # Different content

    def test_service_initialization(self):
        """Test that the service can be initialized."""
        self.assertIsNotNone(self.service)

    @patch('app.utils.image_utils.ai_smart_sharpen')
    def test_apply_ai_sharpening(self, mock_ai_smart_sharpen):
        mock_ai_smart_sharpen.return_value = self.processed_image_mock
        intensity = 0.5

        result = self.service.apply_ai_sharpening(self.dummy_image, intensity=intensity)

        mock_ai_smart_sharpen.assert_called_once_with(unittest.mock.ANY, intensity=intensity)
        self.assertTrue(result["success"])
        self.assertEqual(result["operation"], "ai_sharpen")
        self.assertEqual(result["intensity"], intensity)
        self.assertIs(result["processed_image"], self.processed_image_mock)

    @patch('app.utils.image_utils.ai_smart_denoise')
    def test_apply_ai_denoising(self, mock_ai_smart_denoise):
        mock_ai_smart_denoise.return_value = self.processed_image_mock
        strength = 0.5

        result = self.service.apply_ai_denoising(self.dummy_image, strength=strength)

        mock_ai_smart_denoise.assert_called_once_with(unittest.mock.ANY, strength=strength)
        self.assertTrue(result["success"])
        self.assertEqual(result["operation"], "ai_denoise")
        self.assertEqual(result["strength"], strength)
        self.assertIs(result["processed_image"], self.processed_image_mock)

    @patch('app.utils.image_utils.ai_enhance_accessory_details')
    @patch('app.utils.image_utils.ai_enhance_facial_details')
    def test_apply_ai_model_detail_enhancement(self, mock_facial_enhance, mock_accessory_enhance):
        # Mock returns for facial and accessory enhancements
        # Simulate facial enhancement changes the image, accessory enhancement also changes it further
        intermediate_image_mock = Image.new('RGB', (30,30), 'magenta')
        final_image_mock = Image.new('RGB', (30,30), 'yellow')

        mock_facial_enhance.return_value = intermediate_image_mock
        mock_accessory_enhance.return_value = final_image_mock

        face_intensity = 0.6
        accessory_intensity = 0.7

        result = self.service.apply_ai_model_detail_enhancement(
            self.dummy_image,
            face_intensity=face_intensity,
            accessory_intensity=accessory_intensity
        )

        mock_facial_enhance.assert_called_once_with(unittest.mock.ANY, intensity=face_intensity)
        # The second mock should be called with the result of the first one
        mock_accessory_enhance.assert_called_once_with(intermediate_image_mock, intensity=accessory_intensity)

        self.assertEqual(result["operation"], "ai_model_detail_enhance")
        self.assertEqual(result["face_intensity"], face_intensity)
        self.assertEqual(result["accessory_intensity"], accessory_intensity)
        self.assertTrue(result["face_enhanced"])
        self.assertTrue(result["accessory_enhanced"])
        self.assertIs(result["processed_image"], final_image_mock)

    @patch('app.utils.image_utils.ai_enhance_accessory_details')
    @patch('app.utils.image_utils.ai_enhance_facial_details')
    def test_apply_ai_model_detail_enhancement_no_change(self, mock_facial_enhance, mock_accessory_enhance):
        # Test case where enhancement utilities don't change the image
        mock_facial_enhance.return_value = self.dummy_image # Returns original
        mock_accessory_enhance.return_value = self.dummy_image # Returns original (which was passed to it)

        face_intensity = 0.6
        accessory_intensity = 0.7

        result = self.service.apply_ai_model_detail_enhancement(
            self.dummy_image,
            face_intensity=face_intensity,
            accessory_intensity=accessory_intensity
        )

        self.assertFalse(result["face_enhanced"])
        self.assertFalse(result["accessory_enhanced"])
        self.assertIs(result["processed_image"], self.dummy_image)

if __name__ == '__main__':
    unittest.main()
