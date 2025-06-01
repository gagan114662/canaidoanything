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
    UPSCALE_FACTOR: int = int(os.getenv("UPSCALE_FACTOR", "4")) # General upscale factor

    # ESRGAN Specific Settings
    ESRGAN_RRDB_NUM_IN_CH: int = int(os.getenv("ESRGAN_RRDB_NUM_IN_CH", "3"))
    ESRGAN_RRDB_NUM_OUT_CH: int = int(os.getenv("ESRGAN_RRDB_NUM_OUT_CH", "3"))
    ESRGAN_RRDB_NUM_FEAT: int = int(os.getenv("ESRGAN_RRDB_NUM_FEAT", "64"))
    ESRGAN_RRDB_NUM_BLOCK: int = int(os.getenv("ESRGAN_RRDB_NUM_BLOCK", "23"))
    ESRGAN_RRDB_NUM_GROW_CH: int = int(os.getenv("ESRGAN_RRDB_NUM_GROW_CH", "32"))
    ESRGAN_RRDB_SCALE: int = int(os.getenv("ESRGAN_RRDB_SCALE", "4")) # Model's inherent scale
    
    ESRGAN_DEFAULT_SCALE_PARAM: int = int(os.getenv("ESRGAN_DEFAULT_SCALE_PARAM", "4")) # Default for RealESRGANer instance
    ESRGAN_TILE_SIZE: int = int(os.getenv("ESRGAN_TILE_SIZE", "0")) # 0 for no tiling
    ESRGAN_TILE_PAD: int = int(os.getenv("ESRGAN_TILE_PAD", "10"))
    ESRGAN_PRE_PAD: int = int(os.getenv("ESRGAN_PRE_PAD", "0"))

    # Fallback Upscale settings
    FALLBACK_UPSCALE_BLUR_SIGMA: float = float(os.getenv("FALLBACK_UPSCALE_BLUR_SIGMA", "2.0"))
    FALLBACK_UPSCALE_AW_ALPHA: float = float(os.getenv("FALLBACK_UPSCALE_AW_ALPHA", "1.5")) # src1 weight
    FALLBACK_UPSCALE_AW_BETA: float = float(os.getenv("FALLBACK_UPSCALE_AW_BETA", "-0.5")) # src2 weight (blurred)
    FALLBACK_UPSCALE_AW_GAMMA: float = float(os.getenv("FALLBACK_UPSCALE_AW_GAMMA", "0.0"))

    # Enhance Quality settings (CV based)
    ENHANCE_QUALITY_BILATERAL_D: int = int(os.getenv("ENHANCE_QUALITY_BILATERAL_D", "9"))
    ENHANCE_QUALITY_BILATERAL_SIGMA_COLOR: int = int(os.getenv("ENHANCE_QUALITY_BILATERAL_SIGMA_COLOR", "75"))
    ENHANCE_QUALITY_BILATERAL_SIGMA_SPACE: int = int(os.getenv("ENHANCE_QUALITY_BILATERAL_SIGMA_SPACE", "75"))
    # Sharpening kernel is usually fixed, but strength could be a factor if needed
    CLAHE_CLIP_LIMIT: float = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0")) # Default was 3.0, making it configurable
    CLAHE_TILE_GRID_SIZE: int = int(os.getenv("CLAHE_TILE_GRID_SIZE", "8"))

    # ModelEnhancementService settings
    GFPGAN_MODEL_PATH: str = os.getenv("GFPGAN_MODEL_PATH", "GFPGANv1.4.pth") # Example, should be actual path or downloadable
    GFPGAN_UPSCALE_FACTOR: int = int(os.getenv("GFPGAN_UPSCALE_FACTOR", "2"))
    GFPGAN_ARCH: str = os.getenv("GFPGAN_ARCH", "clean")
    GFPGAN_CHANNEL_MULTIPLIER: int = int(os.getenv("GFPGAN_CHANNEL_MULTIPLIER", "2"))

    MEDIAPIPE_STATIC_MODE: bool = os.getenv("MEDIAPIPE_STATIC_MODE", "True").lower() == "true"
    MEDIAPIPE_MODEL_COMPLEXITY: int = int(os.getenv("MEDIAPIPE_MODEL_COMPLEXITY", "2"))
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = float(os.getenv("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", "0.7"))

    INSIGHTFACE_MODEL_NAME: str = os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l")

    FALLBACK_ENHANCE_SHARPNESS: float = float(os.getenv("FALLBACK_ENHANCE_SHARPNESS", "1.2"))
    FALLBACK_ENHANCE_CONTRAST: float = float(os.getenv("FALLBACK_ENHANCE_CONTRAST", "1.1"))
    FALLBACK_ENHANCE_COLOR: float = float(os.getenv("FALLBACK_ENHANCE_COLOR", "1.05"))

    BODY_PROPORTION_STRETCH_FACTOR: float = float(os.getenv("BODY_PROPORTION_STRETCH_FACTOR", "1.02"))

    # GarmentOptimizationService settings
    SKIN_TONE_HSV_LOWER: List[int] = [int(x) for x in os.getenv("SKIN_TONE_HSV_LOWER", "0,20,70").split(',')]
    SKIN_TONE_HSV_UPPER: List[int] = [int(x) for x in os.getenv("SKIN_TONE_HSV_UPPER", "20,255,255").split(',')]
    FALLBACK_SEG_KERNEL_SIZE: int = int(os.getenv("FALLBACK_SEG_KERNEL_SIZE", "5"))
    PATTERN_EDGE_DENSITY_THRESHOLD: float = float(os.getenv("PATTERN_EDGE_DENSITY_THRESHOLD", "0.1"))

    FORMALITY_SATURATION_THRESH_FORMAL: float = float(os.getenv("FORMALITY_SATURATION_THRESH_FORMAL", "30.0"))
    FORMALITY_VALUE_THRESH_FORMAL: float = float(os.getenv("FORMALITY_VALUE_THRESH_FORMAL", "50.0"))
    FORMALITY_SATURATION_THRESH_BUSINESS: float = float(os.getenv("FORMALITY_SATURATION_THRESH_BUSINESS", "50.0"))
    FORMALITY_VALUE_THRESH_BUSINESS: float = float(os.getenv("FORMALITY_VALUE_THRESH_BUSINESS", "70.0"))

    FIT_COMPACTNESS_OVERSIZED: float = float(os.getenv("FIT_COMPACTNESS_OVERSIZED", "50.0"))
    FIT_COMPACTNESS_LOOSE: float = float(os.getenv("FIT_COMPACTNESS_LOOSE", "30.0"))
    FIT_COMPACTNESS_TIGHT: float = float(os.getenv("FIT_COMPACTNESS_TIGHT", "20.0")) # Lower means tighter

    SHAPE_ENHANCE_BLUR_KERNEL: int = int(os.getenv("SHAPE_ENHANCE_BLUR_KERNEL", "5")) # Must be odd
    SHAPE_ENHANCE_ALPHA: float = float(os.getenv("SHAPE_ENHANCE_ALPHA", "1.5")) # For addWeighted, sharpen details

    DRAPE_CLAHE_CLIP_LIMIT: float = float(os.getenv("DRAPE_CLAHE_CLIP_LIMIT", "2.0"))
    DRAPE_CLAHE_TILE_GRID: int = int(os.getenv("DRAPE_CLAHE_TILE_GRID", "8"))

    COLOR_ADJUST_SATURATION_FACTOR: float = float(os.getenv("COLOR_ADJUST_SATURATION_FACTOR", "1.1"))
    FABRIC_BLUR_KERNEL: int = int(os.getenv("FABRIC_BLUR_KERNEL", "3")) # Must be odd
    FABRIC_BLUR_SIGMA: float = float(os.getenv("FABRIC_BLUR_SIGMA", "0.5"))
    FABRIC_SHARPEN_ALPHA: float = float(os.getenv("FABRIC_SHARPEN_ALPHA", "1.5"))

    COLOR_HARMONY_TARGET_DISTANCE: float = float(os.getenv("COLOR_HARMONY_TARGET_DISTANCE", "100.0"))
    COLOR_HARMONY_RANGE_DIVISOR: float = float(os.getenv("COLOR_HARMONY_RANGE_DIVISOR", "100.0"))

    COLOR_CORRECT_CLIP_LIMIT: float = float(os.getenv("COLOR_CORRECT_CLIP_LIMIT", "2.0"))
    COLOR_CORRECT_TILE_GRID: int = int(os.getenv("COLOR_CORRECT_TILE_GRID", "8"))

    MIN_GARMENT_CONFIDENCE_THRESHOLD: float = float(os.getenv("MIN_GARMENT_CONFIDENCE_THRESHOLD", "0.3"))
    GARMENT_OPT_WEIGHT_FIT: float = float(os.getenv("GARMENT_OPT_WEIGHT_FIT", "0.4"))
    GARMENT_OPT_WEIGHT_STYLE: float = float(os.getenv("GARMENT_OPT_WEIGHT_STYLE", "0.4"))
    GARMENT_OPT_WEIGHT_LIGHTING: float = float(os.getenv("GARMENT_OPT_WEIGHT_LIGHTING", "0.2"))

    # General AI settings
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    GPU_DEVICE_ID: Optional[int] = os.getenv("GPU_DEVICE_ID", None) # Optional: Specific GPU ID to use e.g. 0, 1

    # SceneGenerationService settings (example, if it had configurable parts)
    # SCENE_GEN_DEFAULT_MODEL: str = os.getenv("SCENE_GEN_DEFAULT_MODEL", "some-scene-model")

    # BrandConsistencyService settings (example)
    # BRAND_GUIDELINE_STORAGE_PATH: str = os.getenv("BRAND_GUIDELINE_STORAGE_PATH", "brand_guidelines/")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()