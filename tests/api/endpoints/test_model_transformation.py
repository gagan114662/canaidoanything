import unittest
from unittest.mock import patch, MagicMock
import io
from PIL import Image

from fastapi import UploadFile
from fastapi.testclient import TestClient

# Attempt to import the FastAPI app instance
try:
    from app.main import app # Assuming app.main hosts the FastAPI instance
    APP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import FastAPI app from app.main: {e}")
    app = None
    APP_AVAILABLE = False

# For type hinting, though direct modification is problematic
try:
    from app.models.schemas import ModelTransformationRequest
except ImportError:
    ModelTransformationRequest = None # Fallback if schema cannot be imported

class TestModelTransformationApiParams(unittest.TestCase):

    def setUp(self):
        if not APP_AVAILABLE:
            self.skipTest("FastAPI app instance not available for testing.")
        self.client = TestClient(app)

    def test_client_exists(self):
        """Sanity check that the TestClient was created."""
        self.assertIsNotNone(self.client)

    @patch('app.api.endpoints.model_transformation.process_transformation_task', new_callable=MagicMock)
    @patch('app.api.endpoints.model_transformation.transformation_pipeline', new_callable=MagicMock)
    def test_transform_model_endpoint_with_new_params(self, mock_pipeline_global, mock_process_task):
        """Test the endpoint with composition_rule and controlnet_condition_image."""

        # Configure the global pipeline mock if its methods are called directly by endpoint
        # (Currently, it's mostly for loading, actual work is in process_transformation_task)
        # mock_pipeline_global.some_method.return_value = ...

        # Prepare dummy main image file
        main_image_bytes = io.BytesIO()
        Image.new('RGB', (60, 30), color='red').save(main_image_bytes, format='JPEG')
        main_image_bytes.seek(0)

        # Prepare dummy ControlNet condition image file
        control_image_bytes = io.BytesIO()
        Image.new('RGB', (60, 30), color='blue').save(control_image_bytes, format='JPEG')
        control_image_bytes.seek(0)

        composition_rule = "rule_of_thirds"

        response = self.client.post(
            "/transform-model",
            data={
                "style_prompt": "test_style_prompt",
                "composition_rule": composition_rule,
                "num_variations": 1,
                "enhance_model": True, # FastAPI converts to bool
                "optimize_garment": True,
                "generate_scene": True,
                "quality_mode": "fast",
                # Add other required form fields if any are missing from default
            },
            files={
                "file": ("main.jpg", main_image_bytes, "image/jpeg"),
                "controlnet_condition_image": ("control.jpg", control_image_bytes, "image/jpeg")
            }
        )

        self.assertEqual(response.status_code, 200, response.text)
        mock_process_task.assert_called_once()

        # Inspect arguments passed to process_transformation_task
        # process_transformation_task(transformation_id, input_image, request_data)
        call_args = mock_process_task.call_args
        self.assertIsNotNone(call_args)
        request_data_passed = call_args.args[2] # request_data is the 3rd positional arg

        self.assertEqual(getattr(request_data_passed, 'composition_rule', None), composition_rule)

        control_path = getattr(request_data_passed, 'controlnet_condition_image_path', None)
        self.assertIsNotNone(control_path)
        self.assertIsInstance(control_path, str)
        # The exact name depends on save_upload_file logic (uses transformation_id + suffix)
        # Here, we check if it contains the suffix.
        self.assertTrue(control_path.endswith('_controlnet.jpg') or control_path.endswith('_controlnet.png'))

    @patch('app.api.endpoints.model_transformation.process_transformation_task', new_callable=MagicMock)
    @patch('app.api.endpoints.model_transformation.transformation_pipeline', new_callable=MagicMock)
    def test_transform_model_endpoint_without_optional_params(self, mock_pipeline_global, mock_process_task):
        """Test the endpoint without optional composition_rule and controlnet_condition_image."""

        # Prepare dummy main image file
        main_image_bytes = io.BytesIO()
        Image.new('RGB', (60, 30), color='green').save(main_image_bytes, format='JPEG')
        main_image_bytes.seek(0)

        response = self.client.post(
            "/transform-model",
            data={
                "style_prompt": "test_style_another",
                "num_variations": 1,
                "enhance_model": True,
                "optimize_garment": True,
                "generate_scene": True,
                "quality_mode": "balanced",
            },
            files={
                "file": ("main2.jpg", main_image_bytes, "image/jpeg"),
                # No controlnet_condition_image here
            }
        )

        self.assertEqual(response.status_code, 200, response.text)
        mock_process_task.assert_called_once()

        call_args = mock_process_task.call_args
        self.assertIsNotNone(call_args)
        request_data_passed = call_args.args[2]

        self.assertIsNone(getattr(request_data_passed, 'composition_rule', None))
        self.assertIsNone(getattr(request_data_passed, 'controlnet_condition_image_path', None))


# This allows running the tests directly if the file is executed
if __name__ == '__main__':
    unittest.main()
