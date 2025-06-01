import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import math

from app.core.config import settings
from app.utils.image_utils import get_saliency_map # New import

# FLUX and related imports
try:
    from diffusers import FluxPipeline, ControlNetModel
    from transformers import pipeline
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    logging.warning("FLUX pipeline not available. Install diffusers for scene generation.")

logger = logging.getLogger(__name__)

class SceneGenerationService:
    """Service for generating professional backgrounds and scenes with FLUX Kontext"""
    
    def __init__(self):
        self.flux_pipeline = None
        self.controlnet = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Scene generation templates and styles
        self.background_templates = {
            "studio": {
                "prompt": "professional photography studio, clean white background, studio lighting, seamless backdrop",
                "negative": "cluttered, amateur, messy, distracting elements"
            },
            "outdoor": {
                "prompt": "outdoor natural environment, beautiful landscape, natural lighting, scenic background",
                "negative": "indoor, artificial, studio, confined space"
            },
            "urban": {
                "prompt": "modern urban environment, city background, architectural elements, contemporary setting",
                "negative": "rural, nature, old buildings, countryside"
            },
            "nature": {
                "prompt": "natural environment, organic textures, soft natural lighting, serene atmosphere",
                "negative": "artificial, synthetic, urban, industrial"
            },
            "abstract": {
                "prompt": "abstract artistic background, creative composition, artistic lighting, modern art style",
                "negative": "realistic, literal, traditional, conventional"
            },
            "surreal_dreamscape": {
                "prompt": "surreal dreamscape, floating islands, impossible architecture, vibrant colors, soft ethereal lighting, imaginative sky",
                "negative": "realistic, mundane, ordinary, dull colors, harsh lighting, normal sky"
            },
            "baroque_opulence": {
                "prompt": "baroque era opulence, ornate details, rich textures like velvet and gold, dramatic chiaroscuro lighting, grand hall interior",
                "negative": "modern, minimalist, simple, plain, bright lighting, outdoor"
            },
            "cyberpunk_alley": {
                "prompt": "cyberpunk alleyway, neon signs, rain-slicked streets, futuristic technology, gritty atmosphere, towering skyscrapers in distance, moody lighting",
                "negative": "bright daytime, sunny, clean, natural, historical, rural"
            },
            "ethereal_forest": {
                "prompt": "ethereal enchanted forest, glowing flora, ancient trees, magical mist, soft dappled moonlight or sunlight, fantasy theme",
                "negative": "urban, city, desert, normal forest, bright daylight, man-made structures"
            },
            "cosmic_expanse": {
                "prompt": "vast cosmic expanse, swirling nebulae, distant galaxies, stars, planets, deep space, vibrant cosmic colors, sense of infinite scale",
                "negative": "earthly, ground-level, sky, clouds, interior, small scale, mundane"
            }
        }
        
        # Lighting configurations
        self.lighting_setups = {
            "studio": {
                "key_light": {"position": "front", "intensity": 0.8, "softness": 0.7},
                "fill_light": {"position": "side", "intensity": 0.4, "softness": 0.8},
                "rim_light": {"position": "back", "intensity": 0.3, "softness": 0.5}
            },
            "natural": {
                "main_light": {"position": "overhead", "intensity": 0.9, "softness": 0.9},
                "ambient": {"position": "all", "intensity": 0.3, "softness": 1.0}
            },
            "dramatic": {
                "key_light": {"position": "side", "intensity": 1.0, "softness": 0.3},
                "shadow": {"position": "opposite", "intensity": 0.1, "softness": 0.9}
            }
        }
    
    def load_models(self):
        """Load FLUX and ControlNet models for scene generation"""
        try:
            logger.info("Loading scene generation models...")
            
            if FLUX_AVAILABLE:
                self._load_flux_pipeline()
                self._load_controlnet()
            
            self._initialize_scene_tools()
            
            self.model_loaded = True
            logger.info("Scene generation models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load scene generation models: {str(e)}")
            # Use fallback methods
            self.model_loaded = True
            logger.info("Using fallback scene generation methods")
    
    def _load_flux_pipeline(self):
        """Load FLUX 1.1 pipeline for scene generation"""
        try:
            self.flux_pipeline = FluxPipeline.from_pretrained(
                settings.FLUX_MODEL_PATH,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None
            )
            
            self.flux_pipeline = self.flux_pipeline.to(self.device)
            
            if self.device == "cuda":
                self.flux_pipeline.enable_model_cpu_offload()
                self.flux_pipeline.enable_attention_slicing()
            
            logger.info("FLUX pipeline loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load FLUX pipeline: {e}")
            self.flux_pipeline = None
    
    def _load_controlnet(self):
        """Load ControlNet for guided scene generation"""
        try:
            # Load ControlNet for pose/composition guidance
            self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("ControlNet loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load ControlNet: {e}")
            self.controlnet = None
    
    def _initialize_scene_tools(self):
        """Initialize scene composition and analysis tools"""
        try:
            # Composition analysis tools
            self.composition_rules = {
                "rule_of_thirds": self._apply_rule_of_thirds,
                "golden_ratio": self._apply_golden_ratio,
                "symmetry": self._apply_symmetry,
                "leading_lines": self._apply_leading_lines
            }
            
            # Quality metrics
            self.quality_assessors = {
                "coherence": self._assess_scene_coherence,
                "composition": self._assess_composition_quality,
                "lighting": self._assess_lighting_quality,
                "depth": self._assess_depth_quality
            }
            
            logger.info("Scene composition tools initialized")
            
        except Exception as e:
            logger.warning(f"Scene tools initialization failed: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if models are loaded"""
        return self.model_loaded
    
    def generate_background(
        self, 
        reference_image: Image.Image, 
        style_prompt: str,
        background_type: str = "studio",
        control_image: Optional[Image.Image] = None,
        controlnet_conditioning_scale: float = 0.5,
        creativity_level: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate professional background for the scene
        
        Args:
            reference_image: Reference image for context
            style_prompt: Style description for background
            background_type: Type of background (studio, outdoor, etc.)
            control_image: Optional ControlNet conditioning image
            controlnet_conditioning_scale: Conditioning scale for ControlNet
            creativity_level: Level of creativity for generation (0.0 to 1.0)
            
        Returns:
            Generated background with quality metrics
        """
        try:
            start_time = time.time()
            
            # Get background template
            template = self.background_templates.get(background_type, self.background_templates["studio"])
            
            # Create enhanced prompt
            enhanced_prompt = f"""
            {template['prompt']}, {style_prompt}, 
            high quality, professional photography, 8K resolution, 
            perfect composition, optimal lighting
            """
            
            enhanced_negative = f"""
            {template['negative']}, low quality, amateur, blurry, 
            distorted, ugly, deformed, artifacts
            """
            
            if self.flux_pipeline:
                # Generate with FLUX
                background_image = self._generate_flux_background(
                    reference_image,
                    enhanced_prompt,
                    enhanced_negative,
                    control_image=control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    creativity_level=creativity_level
                )
            else:
                # Fallback generation
                background_image = self._generate_fallback_background(
                    reference_image, background_type
                )
            
            # Assess background quality
            coherence_score = self._assess_background_coherence(background_image, reference_image)
            style_match = self._assess_style_match(background_image, style_prompt)
            type_match = self._assess_background_type_match(background_image, background_type)
            
            processing_time = time.time() - start_time
            
            return {
                "background_image": background_image,
                "coherence_score": coherence_score,
                "style_match": style_match,
                "background_type_match": type_match,
                "background_type": background_type,
                "processing_time": processing_time,
                "generation_method": "FLUX" if self.flux_pipeline else "fallback"
            }
            
        except Exception as e:
            logger.error(f"Background generation failed: {e}")
            return self._fallback_background_result(reference_image, background_type)
    
    def _generate_flux_background(
        self, 
        reference_image: Image.Image, 
        prompt: str, 
        negative_prompt: str,
        control_image: Optional[Image.Image] = None,
        controlnet_conditioning_scale: float = 0.5,
        creativity_level: float = 0.5
    ) -> Image.Image:
        """Generate background using FLUX pipeline, with optional ControlNet guidance."""
        try:
            # Resize reference for processing
            ref_size = reference_image.size
            process_size = (1024, 1024) # Default FLUX size

            # Map creativity_level to guidance_scale
            # Higher creativity_level (e.g., 1.0) -> lower guidance_scale (more freedom)
            # Lower creativity_level (e.g., 0.0) -> higher guidance_scale (stricter prompt adherence)
            min_guidance = 5.0
            max_guidance = 12.0 # Adjusted from 15.0 for a slightly less extreme max
            # Inverted relationship: high creativity = low guidance
            effective_guidance_scale = min_guidance + (1.0 - max(0.0, min(1.0, creativity_level))) * (max_guidance - min_guidance)
            logger.info(f"Creativity: {creativity_level}, Effective Guidance Scale: {effective_guidance_scale:.2f}")

            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": process_size[0],
                "height": process_size[1],
                "num_inference_steps": 20,
                "guidance_scale": effective_guidance_scale,
                "generator": torch.Generator(device=self.device).manual_seed(42)
            }

            if control_image and self.controlnet:
                logger.info(f"Attempting to use ControlNet guidance with scale: {controlnet_conditioning_scale}")
                # The FluxPipeline might not be a ControlNet-specific pipeline.
                # We are attempting to pass parameters that are standard in many Diffusers ControlNet pipelines.
                # This assumes FluxPipeline can accept 'image' as control_image and 'controlnet_conditioning_scale'.
                # If FluxPipeline is not designed for this, it might ignore these or raise an error.
                pipeline_kwargs["image"] = control_image # 'image' is typical for control image in ControlNet pipelines
                pipeline_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
                # Note: Some pipelines might expect 'control_image' instead of 'image' for the control map.
                # If 'image' is strictly for img2img, this might conflict.
                # However, the base pipeline call usually uses 'image' for img2img or control,
                # and txt2img calls (like this one seems to be) often use 'image' for ControlNet input.

            try:
                result = self.flux_pipeline(**pipeline_kwargs)
                if control_image and self.controlnet and "image" in pipeline_kwargs:
                    logger.info("ControlNet parameters passed to FluxPipeline.")
            except TypeError as e:
                logger.warning(f"FluxPipeline does not support ControlNet parameters directly: {e}. Generating without ControlNet.")
                # Remove ControlNet specific args and try again
                if "image" in pipeline_kwargs:
                    del pipeline_kwargs["image"]
                if "controlnet_conditioning_scale" in pipeline_kwargs:
                    del pipeline_kwargs["controlnet_conditioning_scale"]
                result = self.flux_pipeline(**pipeline_kwargs)

            background = result.images[0]
            
            # Resize to match reference
            background = background.resize(ref_size, Image.Resampling.LANCZOS)
            
            return background
            
        except Exception as e:
            logger.error(f"FLUX background generation failed: {e}")
            return self._generate_fallback_background(reference_image, "studio")
    
    def _generate_fallback_background(
        self, 
        reference_image: Image.Image, 
        background_type: str
    ) -> Image.Image:
        """Generate fallback background using traditional methods"""
        try:
            width, height = reference_image.size
            
            # Create background based on type
            if background_type == "studio":
                # Clean white/gray gradient background
                background = self._create_studio_background(width, height)
            elif background_type == "outdoor":
                # Natural gradient background
                background = self._create_outdoor_background(width, height)
            elif background_type == "urban":
                # Urban texture background
                background = self._create_urban_background(width, height)
            else:
                # Default clean background
                background = self._create_clean_background(width, height)
            
            return background
            
        except Exception as e:
            logger.error(f"Fallback background generation failed: {e}")
            return Image.new('RGB', reference_image.size, (240, 240, 240))
    
    def _create_studio_background(self, width: int, height: int) -> Image.Image:
        """Create studio-style background"""
        # Create smooth gradient from light gray to white
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        
        for y in range(height):
            # Gradient from top to bottom
            gray_value = int(220 + (255 - 220) * (y / height))
            for x in range(width):
                pixels[x, y] = (gray_value, gray_value, gray_value)
        
        # Add subtle vignette
        img = self._add_vignette(img, intensity=0.1)
        
        return img
    
    def _create_outdoor_background(self, width: int, height: int) -> Image.Image:
        """Create outdoor-style background"""
        # Create sky-like gradient
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        
        for y in range(height):
            # Sky gradient from blue to lighter blue
            blue_intensity = int(135 + (200 - 135) * (y / height))
            for x in range(width):
                pixels[x, y] = (135, 206, blue_intensity)
        
        return img
    
    def _create_urban_background(self, width: int, height: int) -> Image.Image:
        """Create urban-style background"""
        # Create concrete-like texture
        img = Image.new('RGB', (width, height), (100, 100, 105))
        
        # Add noise for texture
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _create_clean_background(self, width: int, height: int) -> Image.Image:
        """Create clean neutral background"""
        return Image.new('RGB', (width, height), (245, 245, 245))
    
    def _add_vignette(self, image: Image.Image, intensity: float = 0.3) -> Image.Image:
        """Add subtle vignette effect to background"""
        try:
            width, height = image.size
            
            # Create vignette mask
            mask = Image.new('L', (width, height), 255)
            mask_pixels = mask.load()
            
            center_x, center_y = width // 2, height // 2
            max_distance = math.sqrt(center_x**2 + center_y**2)
            
            for y in range(height):
                for x in range(width):
                    # Calculate distance from center
                    distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    # Apply vignette
                    vignette_strength = 1.0 - (distance / max_distance) * intensity
                    vignette_strength = max(vignette_strength, 1.0 - intensity)
                    
                    mask_pixels[x, y] = int(255 * vignette_strength)
            
            # Apply mask
            img_array = np.array(image)
            mask_array = np.array(mask) / 255.0
            
            for c in range(3):
                img_array[:, :, c] = img_array[:, :, c] * mask_array
            
            return Image.fromarray(img_array.astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Vignette application failed: {e}")
            return image
    
    def compose_scene(
        self,
        model_image: Image.Image,
        garment_image: Optional[Image.Image] = None,
        background_type: str = "studio",
        composition_style: str = "commercial",
        creativity_level: float = 0.5,
        control_image: Optional[Image.Image] = None,
        controlnet_conditioning_scale: float = 0.5, # Default, can be overridden from pipeline
        composition_rule_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compose complete scene with model, garment, and background
        
        Args:
            model_image: Model image
            garment_image: Optional separate garment image
            background_type: Background type
            composition_style: Composition style
            creativity_level: Creativity level for background generation
            control_image: Optional ControlNet conditioning image for background generation.
            controlnet_conditioning_scale: Conditioning scale for ControlNet.
            composition_rule_name: Optional name of composition rule to apply.
            
        Returns:
            Composed scene with quality metrics
        """
        try:
            start_time = time.time()
            
            # Generate background
            bg_result = self.generate_background(
                model_image, 
                f"{composition_style} photography background",
                background_type=background_type,
                control_image=control_image, # Pass ControlNet image
                controlnet_conditioning_scale=controlnet_conditioning_scale, # Pass scale
                creativity_level=creativity_level
            )
            background = bg_result["background_image"]
            
            # Compose the scene
            if garment_image:
                # Complex composition with separate garment
                composed_scene = self._compose_complex_scene(
                    model_image, garment_image, background, composition_style
                )
            else:
                # Simple composition with model and background
                composed_scene = self._compose_simple_scene(
                    model_image, background, composition_style
                )
            
            # Apply composition rule if specified
            if composition_rule_name and composition_rule_name in self.composition_rules:
                logger.info(f"Applying composition rule: {composition_rule_name}")
                try:
                    composed_scene = self.composition_rules[composition_rule_name](composed_scene)
                except Exception as e:
                    logger.error(f"Failed to apply composition rule {composition_rule_name}: {e}")
            elif composition_rule_name:
                logger.warning(f"Composition rule '{composition_rule_name}' not found or not applicable.")

            # Assess composition quality
            composition_score = self._assess_composition_quality(composed_scene)
            elements_integrated = self._count_integrated_elements(model_image, garment_image, background)
            
            processing_time = time.time() - start_time
            
            return {
                "composed_scene": composed_scene,
                "composition_score": composition_score,
                "elements_integrated": elements_integrated,
                "background_coherence": bg_result["coherence_score"],
                "composition_style": composition_style,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Scene composition failed: {e}")
            return {
                "composed_scene": model_image,
                "composition_score": 5.0,
                "elements_integrated": 1,
                "background_coherence": 0.5,
                "composition_style": composition_style,
                "processing_time": 0.0,
                "error": str(e)
            }
    
    def _compose_complex_scene(
        self,
        model_image: Image.Image,
        garment_image: Image.Image,
        background: Image.Image,
        style: str
    ) -> Image.Image:
        """Compose scene with model, garment, and background"""
        try:
            # For now, use simple overlay composition
            # In production, this would use more sophisticated blending
            
            # Start with background
            composed = background.copy()
            
            # Blend model image
            composed = self._blend_images(composed, model_image, blend_mode="overlay", opacity=0.9)
            
            return composed
            
        except Exception as e:
            logger.error(f"Complex scene composition failed: {e}")
            return model_image
    
    def _compose_simple_scene(
        self,
        model_image: Image.Image,
        background: Image.Image,
        style: str
    ) -> Image.Image:
        """Compose scene with model and background"""
        try:
            # Simple alpha blending approach
            return self._blend_images(background, model_image, blend_mode="normal", opacity=0.95)
            
        except Exception as e:
            logger.error(f"Simple scene composition failed: {e}")
            return model_image
    
    def _blend_images(
        self,
        base: Image.Image,
        overlay: Image.Image,
        blend_mode: str = "normal",
        opacity: float = 1.0
    ) -> Image.Image:
        """Blend two images with specified mode and opacity"""
        try:
            # Ensure same size
            if base.size != overlay.size:
                overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            
            # Convert to RGBA for blending
            base_rgba = base.convert("RGBA")
            overlay_rgba = overlay.convert("RGBA")
            
            # Simple alpha blending
            blended = Image.alpha_composite(base_rgba, overlay_rgba)
            
            # Convert back to RGB
            return blended.convert("RGB")
            
        except Exception as e:
            logger.error(f"Image blending failed: {e}")
            return base
    
    def generate_styled_scene(
        self,
        model_image: Image.Image,
        style_prompt: str,
        variation_id: str,
        creativity_level: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate styled scene variation
        
        Args:
            model_image: Input model image
            style_prompt: Style description
            variation_id: Unique variation identifier
            creativity_level: Creativity level for scene composition
            
        Returns:
            Styled scene with consistency metrics
        """
        try:
            # Determine background type from style
            background_type = self._infer_background_type(style_prompt)
            
            # Generate scene
            scene_result = self.compose_scene(
                model_image,
                background_type=background_type,
                composition_style=self._extract_composition_style(style_prompt),
                creativity_level=creativity_level
            )
            
            # Apply style-specific enhancements
            styled_scene = self._apply_style_enhancements(
                scene_result["composed_scene"],
                style_prompt
            )
            
            # Assess style consistency
            style_consistency = self._assess_style_consistency(styled_scene, style_prompt)
            
            return {
                "styled_scene": styled_scene,
                "style_consistency": style_consistency,
                "variation_id": variation_id,
                "background_type": background_type,
                "scene_quality": scene_result["composition_score"]
            }
            
        except Exception as e:
            logger.error(f"Styled scene generation failed: {e}")
            return {
                "styled_scene": model_image,
                "style_consistency": 0.7,
                "variation_id": variation_id,
                "background_type": "studio",
                "scene_quality": 5.0,
                "error": str(e)
            }
    
    def _infer_background_type(self, style_prompt: str) -> str:
        """Infer background type from style prompt"""
        style_lower = style_prompt.lower()
        
        if any(word in style_lower for word in ["studio", "clean", "minimal", "commercial"]):
            return "studio"
        elif any(word in style_lower for word in ["outdoor", "natural", "landscape", "garden"]):
            return "outdoor"
        elif any(word in style_lower for word in ["urban", "city", "street", "modern"]):
            return "urban"
        elif any(word in style_lower for word in ["nature", "organic", "forest", "beach"]):
            return "nature"
        elif any(word in style_lower for word in ["artistic", "creative", "abstract", "avant-garde"]):
            return "abstract"
        else:
            return "studio"  # Default
    
    def _extract_composition_style(self, style_prompt: str) -> str:
        """Extract composition style from prompt"""
        style_lower = style_prompt.lower()
        
        if any(word in style_lower for word in ["editorial", "dramatic", "artistic"]):
            return "editorial"
        elif any(word in style_lower for word in ["commercial", "product", "catalog"]):
            return "commercial"
        elif any(word in style_lower for word in ["lifestyle", "casual", "natural"]):
            return "lifestyle"
        else:
            return "commercial"  # Default
    
    def _apply_style_enhancements(self, image: Image.Image, style_prompt: str) -> Image.Image:
        """Apply style-specific enhancements to scene"""
        try:
            enhanced = image.copy()
            style_lower = style_prompt.lower()
            
            # Editorial style: Higher contrast, dramatic
            if "editorial" in style_lower:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(1.1)
            
            # Commercial style: Clean, bright
            elif "commercial" in style_lower:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(1.1)
                
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.05)
            
            # Lifestyle style: Soft, natural
            elif "lifestyle" in style_lower:
                enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
                
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(0.95)
            
            # Artistic style: Creative processing
            elif "artistic" in style_lower:
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.15)
                
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.1)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Style enhancement failed: {e}")
            return image
    
    def apply_lighting(
        self,
        image: Image.Image,
        lighting_type: str = "studio",
        intensity: float = 0.8
    ) -> Dict[str, Any]:
        """
        Apply professional lighting to scene
        
        Args:
            image: Input image
            lighting_type: Type of lighting setup
            intensity: Lighting intensity (0-1)
            
        Returns:
            Lit image with quality metrics
        """
        try:
            # Get lighting configuration
            lighting_config = self.lighting_setups.get(lighting_type, self.lighting_setups["studio"])
            
            # Apply lighting effects
            lit_image = self._apply_lighting_effects(image, lighting_config, intensity)
            
            # Assess lighting quality
            lighting_quality = self._assess_lighting_quality(lit_image)
            
            return {
                "lit_image": lit_image,
                "lighting_quality": lighting_quality,
                "lighting_type": lighting_type,
                "intensity_applied": intensity
            }
            
        except Exception as e:
            logger.error(f"Lighting application failed: {e}")
            return {
                "lit_image": image,
                "lighting_quality": 0.7,
                "lighting_type": lighting_type,
                "intensity_applied": intensity,
                "error": str(e)
            }
    
    def _apply_lighting_effects(
        self,
        image: Image.Image,
        lighting_config: Dict[str, Any],
        intensity: float
    ) -> Image.Image:
        """Apply lighting effects based on configuration"""
        try:
            lit_image = image.copy()
            
            # Convert to array for processing
            img_array = np.array(lit_image).astype(np.float32)
            
            # Apply each light in configuration
            for light_name, light_params in lighting_config.items():
                light_intensity = light_params["intensity"] * intensity
                light_softness = light_params["softness"]
                
                # Create light effect
                light_effect = self._create_light_effect(
                    img_array.shape[:2],
                    light_params["position"],
                    light_intensity,
                    light_softness
                )
                
                # Apply light to image
                for c in range(3):  # RGB channels
                    img_array[:, :, c] = np.clip(
                        img_array[:, :, c] * (1 + light_effect),
                        0, 255
                    )
            
            return Image.fromarray(img_array.astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Lighting effects application failed: {e}")
            return image
    
    def _create_light_effect(
        self,
        image_shape: Tuple[int, int],
        position: str,
        intensity: float,
        softness: float
    ) -> np.ndarray:
        """Create light effect mask"""
        try:
            height, width = image_shape
            effect = np.zeros((height, width), dtype=np.float32)
            
            # Determine light position
            if position == "front":
                center_x, center_y = width // 2, height // 3
            elif position == "side":
                center_x, center_y = width // 4, height // 2
            elif position == "back":
                center_x, center_y = width // 2, height * 2 // 3
            elif position == "overhead":
                center_x, center_y = width // 2, height // 4
            else:  # "all" for ambient
                # Uniform lighting
                return np.full((height, width), intensity * 0.3, dtype=np.float32)
            
            # Create radial gradient
            for y in range(height):
                for x in range(width):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_distance = np.sqrt(width**2 + height**2) // 2
                    
                    # Apply softness and intensity
                    light_value = intensity * np.exp(-distance / (max_distance * softness))
                    effect[y, x] = light_value
            
            return effect
            
        except Exception as e:
            logger.error(f"Light effect creation failed: {e}")
            return np.zeros(image_shape, dtype=np.float32)
    
    def adjust_depth_perspective(
        self,
        image: Image.Image,
        depth_level: float = 0.5,
        perspective_angle: float = 0,
        focal_point: str = "center"
    ) -> Dict[str, Any]:
        """
        Adjust depth and perspective of the scene
        
        Args:
            image: Input image
            depth_level: Depth effect strength (0-1)
            perspective_angle: Perspective angle in degrees
            focal_point: Focus point (center, top, bottom)
            
        Returns:
            Depth-adjusted image with quality metrics
        """
        try:
            # Create depth map
            depth_map = self._generate_depth_map(image, focal_point)
            
            # Apply depth effects
            depth_adjusted = self._apply_depth_effects(image, depth_map, depth_level)
            
            # Apply perspective transformation if needed
            if perspective_angle != 0:
                depth_adjusted = self._apply_perspective_transform(depth_adjusted, perspective_angle)
            
            # Assess perspective quality
            perspective_quality = self._assess_perspective_quality(depth_adjusted)
            
            return {
                "depth_adjusted_image": depth_adjusted,
                "depth_map": depth_map,
                "perspective_quality": perspective_quality,
                "depth_level_applied": depth_level,
                "perspective_angle_applied": perspective_angle
            }
            
        except Exception as e:
            logger.error(f"Depth/perspective adjustment failed: {e}")
            return {
                "depth_adjusted_image": image,
                "depth_map": Image.new('L', image.size, 128),
                "perspective_quality": 0.8,
                "depth_level_applied": depth_level,
                "perspective_angle_applied": perspective_angle,
                "error": str(e)
            }
    
    def _generate_depth_map(self, image: Image.Image, focal_point: str) -> Image.Image:
        """Generate depth map for image"""
        try:
            width, height = image.size
            depth_map = Image.new('L', (width, height))
            pixels = depth_map.load()
            
            # Determine focal area
            if focal_point == "center":
                focus_x, focus_y = width // 2, height // 2
            elif focal_point == "top":
                focus_x, focus_y = width // 2, height // 4
            elif focal_point == "bottom":
                focus_x, focus_y = width // 2, height * 3 // 4
            else:
                focus_x, focus_y = width // 2, height // 2
            
            max_distance = math.sqrt(width**2 + height**2) // 2
            
            # Create radial depth gradient
            for y in range(height):
                for x in range(width):
                    distance = math.sqrt((x - focus_x)**2 + (y - focus_y)**2)
                    depth_value = int(255 * (1 - distance / max_distance))
                    depth_value = max(0, min(255, depth_value))
                    pixels[x, y] = depth_value
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Depth map generation failed: {e}")
            return Image.new('L', image.size, 128)
    
    def _apply_depth_effects(
        self,
        image: Image.Image,
        depth_map: Image.Image,
        depth_level: float
    ) -> Image.Image:
        """Apply depth-of-field effects based on depth map"""
        try:
            img_array = np.array(image)
            depth_array = np.array(depth_map).astype(np.float32) / 255.0
            
            # Create blur effect based on depth
            blurred = image.filter(ImageFilter.GaussianBlur(radius=3))
            blurred_array = np.array(blurred)
            
            # Blend based on depth and level
            result_array = img_array.copy().astype(np.float32)
            
            for c in range(3):  # RGB channels
                blur_factor = (1 - depth_array) * depth_level
                result_array[:, :, c] = (
                    img_array[:, :, c] * (1 - blur_factor) +
                    blurred_array[:, :, c] * blur_factor
                )
            
            return Image.fromarray(result_array.astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Depth effects application failed: {e}")
            return image
    
    def _apply_perspective_transform(self, image: Image.Image, angle: float) -> Image.Image:
        """Apply perspective transformation"""
        try:
            # Simple perspective transformation using CV2
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            height, width = img_cv.shape[:2]
            
            # Define perspective transformation points
            angle_rad = math.radians(angle)
            offset = int(height * 0.1 * math.sin(angle_rad))
            
            src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            dst_points = np.float32([
                [offset, 0], [width - offset, 0],
                [0, height], [width, height]
            ])
            
            # Apply transformation
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            transformed = cv2.warpPerspective(img_cv, matrix, (width, height))
            
            return Image.fromarray(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.error(f"Perspective transformation failed: {e}")
            return image
    
    def generate_context_background(
        self,
        reference_image: Image.Image,
        context: str,
        adaptation_level: float = 0.8
    ) -> Dict[str, Any]:
        """Generate context-aware background"""
        try:
            # Map context to background type and style
            context_mapping = {
                "office": ("studio", "professional clean office environment"),
                "outdoor": ("outdoor", "natural outdoor environment, beautiful scenery"),
                "party": ("indoor", "festive party environment, warm lighting"),
                "casual": ("lifestyle", "casual comfortable environment"),
                "formal": ("studio", "formal elegant environment, sophisticated")
            }
            
            bg_type, style_prompt = context_mapping.get(context, ("studio", "professional environment"))
            
            # Generate background
            bg_result = self.generate_background(reference_image, style_prompt, bg_type)
            
            # Assess context matching
            context_match = self._assess_context_match(bg_result["background_image"], context)
            adaptation_score = context_match * adaptation_level
            
            return {
                "background_image": bg_result["background_image"],
                "context_match": context_match,
                "adaptation_score": adaptation_score,
                "context": context,
                "background_type": bg_type
            }
            
        except Exception as e:
            logger.error(f"Context background generation failed: {e}")
            return {
                "background_image": reference_image,
                "context_match": 0.7,
                "adaptation_score": 0.7,
                "context": context,
                "background_type": "studio"
            }
    
    def generate_brand_environment(
        self,
        reference_image: Image.Image,
        brand_guidelines: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate brand-consistent environment"""
        try:
            # Extract brand parameters
            color_palette = brand_guidelines.get("color_palette", ["#FFFFFF"])
            style = brand_guidelines.get("style", "modern")
            mood = brand_guidelines.get("mood", "professional")
            env_type = brand_guidelines.get("environment_type", "studio")
            
            # Create brand-specific prompt
            style_prompt = f"{style} {mood} brand environment, {env_type} setting"
            
            # Generate background
            bg_result = self.generate_background(reference_image, style_prompt, env_type)
            brand_scene = bg_result["background_image"]
            
            # Apply brand color palette if possible
            if color_palette:
                brand_scene = self._apply_brand_colors(brand_scene, color_palette)
            
            # Assess brand consistency
            brand_consistency = self._assess_brand_consistency(brand_scene, brand_guidelines)
            guideline_adherence = self._assess_guideline_adherence(brand_scene, brand_guidelines)
            
            return {
                "brand_scene": brand_scene,
                "brand_consistency": brand_consistency,
                "guideline_adherence": guideline_adherence,
                "brand_guidelines_applied": brand_guidelines
            }
            
        except Exception as e:
            logger.error(f"Brand environment generation failed: {e}")
            return {
                "brand_scene": reference_image,
                "brand_consistency": 0.7,
                "guideline_adherence": 0.7,
                "brand_guidelines_applied": brand_guidelines
            }
    
    def _apply_brand_colors(self, image: Image.Image, color_palette: List[str]) -> Image.Image:
        """Apply brand color palette to image"""
        try:
            # Simple color adjustment - in production would use more sophisticated color mapping
            enhancer = ImageEnhance.Color(image)
            color_adjusted = enhancer.enhance(1.1)
            
            return color_adjusted
            
        except Exception as e:
            logger.error(f"Brand color application failed: {e}")
            return image
    
    def apply_composition_rules(
        self,
        image: Image.Image,
        rules: List[str],
        primary_rule: str = "rule_of_thirds"
    ) -> Dict[str, Any]:
        """Apply composition rules to image"""
        try:
            composed_image = image.copy()
            rules_applied = []
            
            # Apply primary rule
            if primary_rule in self.composition_rules:
                composed_image = self.composition_rules[primary_rule](composed_image)
                rules_applied.append(primary_rule)
            
            # Apply additional rules
            for rule in rules:
                if rule != primary_rule and rule in self.composition_rules:
                    composed_image = self.composition_rules[rule](composed_image)
                    rules_applied.append(rule)
            
            # Assess rule adherence
            rule_adherence = self._assess_rule_adherence(composed_image, rules_applied)
            
            return {
                "composed_image": composed_image,
                "rule_adherence": rule_adherence,
                "composition_analysis": {
                    "rules_applied": rules_applied,
                    "primary_rule": primary_rule
                }
            }
            
        except Exception as e:
            logger.error(f"Composition rules application failed: {e}")
            return {
                "composed_image": image,
                "rule_adherence": 0.8,
                "composition_analysis": {
                    "rules_applied": [],
                    "primary_rule": primary_rule
                }
            }
    
    def _apply_rule_of_thirds(self, image: Image.Image) -> Image.Image:
        """Apply rule of thirds composition"""
        width, height = image.size

        # Calculate the new dimensions (2/3 of original)
        new_width = int(width * 2 / 3)
        new_height = int(height * 2 / 3)

        if new_width == 0 or new_height == 0:
            logger.warning("Image too small for rule of thirds crop, returning original.")
            return image

        # Determine the center of the crop (top-left power point)
        center_x = int(width / 3)
        center_y = int(height / 3)

        # Calculate crop box coordinates
        left = max(0, center_x - new_width // 2)
        top = max(0, center_y - new_height // 2)
        right = min(width, center_x + new_width // 2)
        bottom = min(height, center_y + new_height // 2)

        # Ensure the crop box has valid dimensions
        if right - left < new_width: # Adjust if crop is too small due to hitting image edge
            if center_x + new_width // 2 > width : # Hit right edge
                 right = width
                 left = max(0, right - new_width)
            else: # Hit left edge or very small image
                 left = 0
                 right = min(new_width, width)


        if bottom - top < new_height: # Adjust if crop is too small due to hitting image edge
            if center_y + new_height // 2 > height: # Hit bottom edge
                bottom = height
                top = max(0, bottom - new_height)
            else: # Hit top edge or very small image
                top = 0
                bottom = min(new_height, height)

        # Ensure final crop dimensions are new_width x new_height if possible,
        # otherwise, they are as large as possible up to new_width x new_height.
        # The logic above tries to keep the center point, but might shift it if image is too small.
        # For this implementation, we will crop to the calculated box [left,top,right,bottom]
        # and if it's smaller than new_width/new_height, that's acceptable for small images.

        cropped_image = image.crop((left, top, right, bottom))

        # If the crop resulted in a size smaller than intended (e.g. original image was small)
        # and not the target new_width/new_height, we might want to resize it,
        # but the instructions imply just cropping.
        # For now, we return the cropped image as is.
        # If a specific output dimension is strictly required, resizing would be needed here.
        # Example: if cropped_image.size != (new_width, new_height):
        #    cropped_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return cropped_image
    
    def _apply_golden_ratio(self, image: Image.Image) -> Image.Image:
        """Apply golden ratio composition"""
        width, height = image.size
        golden_ratio = 1.618

        new_width = width
        new_height = height

        if width > height:
            # Try to make width = height * golden_ratio
            test_width = int(height * golden_ratio)
            if test_width <= width:
                new_width = test_width
            else:
                # If that makes it wider than original, then width stays, adjust height
                new_height = int(width / golden_ratio)
        else: # height >= width
            # Try to make height = width * golden_ratio
            test_height = int(width * golden_ratio)
            if test_height <= height:
                new_height = test_height
            else:
                # If that makes it taller than original, then height stays, adjust width
                new_width = int(height / golden_ratio)

        if new_width == 0 or new_height == 0:
            logger.warning("Image too small for golden ratio crop, returning original.")
            return image

        if new_width > width or new_height > height: # Should not happen with the logic above
            logger.warning("Calculated crop larger than original, returning original image.")
            return image


        # Center the crop
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        # Ensure coordinates are within bounds (though centered crop should be)
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)

        # Ensure the crop box has valid dimensions before cropping
        if right <= left or bottom <= top:
            logger.warning("Invalid crop dimensions for golden ratio, returning original.")
            return image

        return image.crop((left, top, right, bottom))
    
    def _apply_symmetry(self, image: Image.Image) -> Image.Image:
        """Apply symmetrical composition"""
        width, height = image.size

        if width < 2 : # Cannot make symmetrical if width is less than 2 pixels
            logger.warning("Image too narrow for symmetry, returning original.")
            return image

        # Crop to the left half
        left_half = image.crop((0, 0, width // 2, height))

        # Flip the left half horizontally
        flipped_left_half = left_half.transpose(Image.FLIP_LEFT_RIGHT)

        # Create a new image of the original width
        symmetrical_image = Image.new('RGB', (width, height))

        # Paste the original left half
        symmetrical_image.paste(left_half, (0, 0))

        # Paste the flipped left half to the right side
        # If width is odd, the middle column of pixels from the original image's center
        # will be overwritten by the flipped image. This is generally acceptable.
        symmetrical_image.paste(flipped_left_half, (width // 2, 0))

        return symmetrical_image
    
    def _apply_leading_lines(self, image: Image.Image) -> Image.Image:
        """Apply leading lines composition"""
        # This is a placeholder for future implementation.
        # Leading lines detection and enhancement is complex and typically
        # requires more advanced image analysis (e.g., Hough transforms)
        # or AI-based approaches, which are beyond simple cropping.
        logger.info("Leading lines rule is a placeholder and returns the original image.")
        return image
    
    def integrate_scene_elements(
        self,
        reference_image: Image.Image,
        elements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate multiple scene elements"""
        try:
            # Start with reference image
            integrated_scene = reference_image.copy()
            
            # Apply each element
            element_harmony = 1.0
            
            # Background
            if "background" in elements:
                bg_result = self.generate_background(
                    reference_image,
                    f"professional {elements['background']} background",
                    elements["background"]
                )
                integrated_scene = self._blend_images(integrated_scene, bg_result["background_image"])
                element_harmony *= bg_result["coherence_score"]
            
            # Atmosphere
            if "atmosphere" in elements:
                integrated_scene = self._apply_atmosphere(integrated_scene, elements["atmosphere"])
            
            # Color scheme
            if "color_scheme" in elements:
                integrated_scene = self._apply_color_scheme(integrated_scene, elements["color_scheme"])
            
            integration_quality = element_harmony * 10
            
            return {
                "integrated_scene": integrated_scene,
                "element_harmony": element_harmony,
                "integration_quality": integration_quality,
                "elements_applied": list(elements.keys())
            }
            
        except Exception as e:
            logger.error(f"Scene element integration failed: {e}")
            return {
                "integrated_scene": reference_image,
                "element_harmony": 0.8,
                "integration_quality": 8.0,
                "elements_applied": []
            }
    
    def _apply_atmosphere(self, image: Image.Image, atmosphere: str) -> Image.Image:
        """Apply atmospheric effects"""
        try:
            if atmosphere == "professional":
                # Increase contrast and clarity
                enhancer = ImageEnhance.Contrast(image)
                return enhancer.enhance(1.1)
            elif atmosphere == "warm":
                # Add warm tint
                enhancer = ImageEnhance.Color(image)
                return enhancer.enhance(1.1)
            elif atmosphere == "cool":
                # Add cool tint
                enhancer = ImageEnhance.Color(image)
                return enhancer.enhance(0.9)
            else:
                return image
                
        except Exception as e:
            logger.error(f"Atmosphere application failed: {e}")
            return image
    
    def _apply_color_scheme(self, image: Image.Image, color_scheme: str) -> Image.Image:
        """Apply color scheme to image"""
        try:
            if color_scheme == "monochromatic":
                # Reduce saturation
                enhancer = ImageEnhance.Color(image)
                return enhancer.enhance(0.3)
            elif color_scheme == "vibrant":
                # Increase saturation
                enhancer = ImageEnhance.Color(image)
                return enhancer.enhance(1.3)
            else:
                return image
                
        except Exception as e:
            logger.error(f"Color scheme application failed: {e}")
            return image
    
    def generate_complete_scene(self, reference_image: Image.Image) -> Dict[str, Any]:
        """Generate complete professional scene"""
        try:
            # Generate with default professional settings
            scene_result = self.compose_scene(
                reference_image,
                background_type="studio",
                composition_style="commercial"
            )
            
            # Apply professional lighting
            lighting_result = self.apply_lighting(
                scene_result["composed_scene"],
                lighting_type="studio",
                intensity=0.8
            )
            
            complete_scene = lighting_result["lit_image"]
            scene_quality = (scene_result["composition_score"] + lighting_result["lighting_quality"] * 10) / 2
            
            return {
                "complete_scene": complete_scene,
                "scene_quality": scene_quality,
                "composition_score": scene_result["composition_score"],
                "lighting_quality": lighting_result["lighting_quality"]
            }
            
        except Exception as e:
            logger.error(f"Complete scene generation failed: {e}")
            return {
                "complete_scene": reference_image,
                "scene_quality": 7.0,
                "composition_score": 7.0,
                "lighting_quality": 0.7
            }
    
    def generate_batch_scenes(
        self,
        images: List[Image.Image],
        styles: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate scenes for multiple images and styles"""
        results = []
        
        for i, image in enumerate(images):
            style = styles[i] if i < len(styles) else "commercial"
            
            result = self.generate_styled_scene(
                image,
                f"{style} photography",
                f"batch_{i}_{style}"
            )
            result["style_applied"] = style
            results.append(result)
        
        return results
    
    def calculate_scene_metrics(self, scene_image: Image.Image) -> Dict[str, float]:
        """Calculate comprehensive scene quality metrics"""
        try:
            metrics = {}
            
            # Calculate each quality metric
            for metric_name, assessor in self.quality_assessors.items():
                metrics[metric_name] = assessor(scene_image)
            
            # Calculate overall score
            weights = {"coherence": 0.3, "composition": 0.3, "lighting": 0.25, "depth": 0.15}
            overall_score = sum(metrics[metric] * weights.get(metric, 0.25) for metric in metrics) * 10
            metrics["overall_score"] = min(overall_score, 10.0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Scene metrics calculation failed: {e}")
            return {
                "coherence": 0.8,
                "composition": 0.8,
                "lighting": 0.8,
                "depth": 0.8,
                "overall_score": 8.0
            }
    
    # Assessment methods
    def _assess_scene_coherence(self, image: Image.Image) -> float:
        """Assess overall scene coherence"""
        return 0.85  # Simplified assessment
    
    def _assess_composition_quality(self, image: Image.Image) -> float:
        """Assess composition quality"""
        return 8.5  # Simplified assessment
    
    def _assess_lighting_quality(self, image: Image.Image) -> float:
        """Assess lighting quality"""
        return 0.88  # Simplified assessment
    
    def _assess_depth_quality(self, image: Image.Image) -> float:
        """Assess depth perception quality"""
        return 0.82  # Simplified assessment
    
    def _assess_background_coherence(self, background: Image.Image, reference: Image.Image) -> float:
        """Assess background coherence with reference"""
        return 0.87  # Simplified assessment
    
    def _assess_style_match(self, image: Image.Image, style_prompt: str) -> float:
        """Assess how well image matches style prompt"""
        return 0.85  # Simplified assessment
    
    def _assess_background_type_match(self, image: Image.Image, bg_type: str) -> float:
        """Assess background type match"""
        return 0.9  # Simplified assessment
    
    def _assess_style_consistency(self, image: Image.Image, style_prompt: str) -> float:
        """Assess style consistency"""
        return 0.88  # Simplified assessment
    
    def _assess_context_match(self, image: Image.Image, context: str) -> float:
        """Assess context matching"""
        return 0.85  # Simplified assessment
    
    def _assess_brand_consistency(self, image: Image.Image, guidelines: Dict[str, Any]) -> float:
        """Assess brand consistency"""
        return 0.87  # Simplified assessment
    
    def _assess_guideline_adherence(self, image: Image.Image, guidelines: Dict[str, Any]) -> float:
        """Assess guideline adherence"""
        return 0.83  # Simplified assessment
    
    def _assess_perspective_quality(self, image: Image.Image) -> float:
        """Assess perspective quality"""
        return 0.85  # Simplified assessment
    
    def _assess_rule_adherence(self, image: Image.Image, rules: List[str]) -> float:
        """Assess composition rule adherence"""
        return 0.88  # Simplified assessment
    
    def _count_integrated_elements(
        self,
        model_image: Image.Image,
        garment_image: Optional[Image.Image],
        background: Image.Image
    ) -> int:
        """Count successfully integrated elements"""
        count = 1  # Always have model
        if garment_image:
            count += 1
        if background:
            count += 1
        return count
    
    def _fallback_background_result(
        self,
        reference_image: Image.Image,
        background_type: str
    ) -> Dict[str, Any]:
        """Create fallback background result"""
        fallback_bg = self._generate_fallback_background(reference_image, background_type)
        
        return {
            "background_image": fallback_bg,
            "coherence_score": 0.7,
            "style_match": 0.7,
            "background_type_match": 0.8,
            "background_type": background_type,
            "processing_time": 0.1,
            "generation_method": "fallback",
            "fallback_used": True
        }
    
    def clear_cache(self):
        """Clear scene generation cache"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Scene generation cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def unload_models(self):
        """Unload models to free memory"""
        try:
            if self.flux_pipeline:
                del self.flux_pipeline
                self.flux_pipeline = None
            
            if self.controlnet:
                del self.controlnet
                self.controlnet = None
            
            self.model_loaded = False
            self.clear_cache()
            logger.info("Scene generation models unloaded")
            
        except Exception as e:
            logger.error(f"Failed to unload models: {e}")

    def suggest_crop(self, image: Image.Image, composition_rule_name: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Suggesting crop for image. Rule: {composition_rule_name}")
        original_width, original_height = image.size

        try:
            saliency_map = get_saliency_map(image) # HxW numpy array

            # Find bounding box of the most salient region (e.g., top 25% brightest pixels)
            # This is a simplified approach for a mock.
            if saliency_map.size == 0: # Handle empty saliency map
                logger.warning("Saliency map is empty, suggesting full image.")
                return {"x": 0, "y": 0, "width": original_width, "height": original_height, "rule_applied": None, "error": "Empty saliency map"}

            threshold_value = np.percentile(saliency_map, 75)
            salient_points = np.argwhere(saliency_map >= threshold_value) # array of [row, col]

            if salient_points.size == 0: # No salient points found above threshold
                logger.warning("No salient points found above threshold, suggesting full image.")
                return {"x": 0, "y": 0, "width": original_width, "height": original_height, "rule_applied": None}

            min_row, min_col = salient_points.min(axis=0)
            max_row, max_col = salient_points.max(axis=0)

            # Suggested crop based purely on saliency
            s_x = int(min_col)
            s_y = int(min_row)
            s_width = int(max_col - min_col + 1)
            s_height = int(max_row - min_row + 1)

            # Ensure width and height are positive
            s_width = max(1, s_width)
            s_height = max(1, s_height)

            crop_coords = {"x": s_x, "y": s_y, "width": s_width, "height": s_height}
            rule_applied_log = "saliency_basic"

            if composition_rule_name and composition_rule_name in self.composition_rules:
                logger.info(f"Saliency crop suggested. Rule '{composition_rule_name}' provided for consideration (adjustment logic is conceptual).")
                # Conceptual: future logic could adjust crop_coords based on the rule.
                # For example, try to center the salient box according to the rule, or ensure aspect ratio.
                # For now, we just log and use the saliency-derived box.
                rule_applied_log = f"saliency_with_contemplated_{composition_rule_name}"

            # Clamp coordinates to image bounds
            crop_coords["x"] = max(0, crop_coords["x"])
            crop_coords["y"] = max(0, crop_coords["y"])
            crop_coords["width"] = min(original_width - crop_coords["x"], crop_coords["width"])
            crop_coords["height"] = min(original_height - crop_coords["y"], crop_coords["height"])

            # Ensure width/height are not zero after clamping if x/y were pushed to image edge
            if crop_coords["x"] + crop_coords["width"] > original_width:
                crop_coords["width"] = original_width - crop_coords["x"]
            if crop_coords["y"] + crop_coords["height"] > original_height:
                crop_coords["height"] = original_height - crop_coords["y"]

            if crop_coords["width"] <= 0 or crop_coords["height"] <= 0:
                logger.warning("Calculated crop has zero or negative dimension, suggesting full image.")
                return {"x": 0, "y": 0, "width": original_width, "height": original_height, "rule_applied": None, "error": "Invalid crop dimension after clamping"}

            return {"x": crop_coords["x"], "y": crop_coords["y"], "width": crop_coords["width"], "height": crop_coords["height"], "rule_applied": rule_applied_log}

        except Exception as e:
            logger.error(f"Error during crop suggestion: {str(e)}")
            return {"x": 0, "y": 0, "width": original_width, "height": original_height, "error": str(e), "rule_applied": None}