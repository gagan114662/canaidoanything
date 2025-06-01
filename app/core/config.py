import os
from typing import List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Garment Creative AI"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    ALLOWED_HOSTS: List[str] = os.getenv("ALLOWED_HOSTS", "*").split(",")
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./garment_ai.db")
    
    # Redis settings for Celery
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
    
    # File storage settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "app/static/uploads")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "app/static/outputs")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]
    
    # AI Model settings
    FLUX_MODEL_PATH: str = os.getenv("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev")
    SAM2_MODEL_PATH: str = os.getenv("SAM2_MODEL_PATH", "facebook/sam2-hiera-large")
    ESRGAN_MODEL_PATH: str = os.getenv("ESRGAN_MODEL_PATH", "ai-forever/Real-ESRGAN")
    
    # HuggingFace settings
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # Processing settings
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
    UPSCALE_FACTOR: int = int(os.getenv("UPSCALE_FACTOR", "4"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()