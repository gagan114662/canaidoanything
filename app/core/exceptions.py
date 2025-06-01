class ModelLoadError(Exception):
    """Custom exception for errors during AI model loading."""
    def __init__(self, model_name: str, original_exception: Exception, path: str = "N/A"):
        self.model_name = model_name
        self.original_exception = original_exception
        self.path = path
        message = f"Failed to load model '{model_name}'"
        if path != "N/A":
            message += f" from path/ID '{path}'"
        message += f". Original error: {original_exception}"
        super().__init__(message)

    def __str__(self):
        return f"ModelLoadError: {super().__str__()} (Caused by: {type(self.original_exception).__name__})"

class PipelineLoadError(Exception):
    """Custom exception for errors when a critical part of a pipeline fails to load."""
    def __init__(self, pipeline_name: str, message: str, service_errors: dict = None):
        self.pipeline_name = pipeline_name
        self.service_errors = service_errors or {}
        full_message = f"Failed to load pipeline '{pipeline_name}': {message}"
        if self.service_errors:
            full_message += "\nService-specific errors:\n"
            for service, error in self.service_errors.items():
                full_message += f"  - {service}: {error}\n"
        super().__init__(full_message)

classServiceInitializationError(Exception):
    """Custom exception for errors during AI service initialization (not model loading)."""
    def __init__(self, service_name: str, original_exception: Exception):
        self.service_name = service_name
        self.original_exception = original_exception
        super().__init__(f"Failed to initialize service '{service_name}': {original_exception}")
