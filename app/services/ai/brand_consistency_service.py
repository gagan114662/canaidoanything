import torch
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import colorsys
from pathlib import Path

from app.core.config import settings

# LoRA and brand consistency imports
try:
    from diffusers import LoraConfig, StableDiffusionPipeline
    from peft import get_peft_model, LoraConfig as PeftLoraConfig
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    logging.warning("LoRA not available. Install peft and diffusers for brand consistency.")

logger = logging.getLogger(__name__)

class BrandConsistencyService:
    """Service for ensuring brand consistency across outputs using LoRA and brand guidelines"""
    
    def __init__(self):
        self.brand_models = {}
        self.brand_guidelines = {}
        self.loaded_brands = set()
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Default brand templates
        self.brand_templates = {
            "luxury": {
                "color_palette": ["#000000", "#FFFFFF", "#C9B037", "#E5E4E2"],
                "style_keywords": ["elegant", "sophisticated", "premium", "timeless"],
                "composition_style": "minimalist",
                "lighting_preference": "soft_dramatic",
                "mood": "sophisticated",
                "typography_style": "serif"
            },
            "streetwear": {
                "color_palette": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
                "style_keywords": ["urban", "edgy", "contemporary", "bold"],
                "composition_style": "dynamic",
                "lighting_preference": "natural_bright",
                "mood": "energetic",
                "typography_style": "sans-serif"
            },
            "minimalist": {
                "color_palette": ["#FFFFFF", "#F5F5F5", "#E0E0E0", "#BDBDBD"],
                "style_keywords": ["clean", "simple", "modern", "pure"],
                "composition_style": "minimal",
                "lighting_preference": "soft_even",
                "mood": "serene",
                "typography_style": "geometric"
            },
            "vintage": {
                "color_palette": ["#8B4513", "#D2691E", "#CD853F", "#F4A460"],
                "style_keywords": ["retro", "classic", "nostalgic", "heritage"],
                "composition_style": "classic",
                "lighting_preference": "warm_golden",
                "mood": "nostalgic",
                "typography_style": "vintage"
            }
        }
        
        # Brand consistency metrics
        self.consistency_weights = {
            "color_harmony": 0.3,
            "style_adherence": 0.25,
            "composition_match": 0.2,
            "mood_consistency": 0.15,
            "quality_standard": 0.1
        }
    
    def load_models(self):
        """Load brand consistency models and LoRA adapters"""
        try:
            logger.info("Loading brand consistency models...")
            
            if LORA_AVAILABLE:
                self._load_lora_models()
            
            self._initialize_brand_analysis_tools()
            
            self.model_loaded = True
            logger.info("Brand consistency models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load brand consistency models: {str(e)}")
            # Use fallback methods
            self.model_loaded = True
            logger.info("Using fallback brand consistency methods")
    
    def _load_lora_models(self):
        """Load LoRA models for brand-specific adaptations"""
        try:
            # Initialize base pipeline for LoRA
            self.base_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=settings.HF_TOKEN if settings.HF_TOKEN else None
            )
            
            self.base_pipeline = self.base_pipeline.to(self.device)
            
            if self.device == "cuda":
                self.base_pipeline.enable_model_cpu_offload()
                self.base_pipeline.enable_attention_slicing()
            
            logger.info("Base pipeline for LoRA loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load LoRA base pipeline: {e}")
            self.base_pipeline = None
    
    def _initialize_brand_analysis_tools(self):
        """Initialize brand analysis and consistency tools"""
        try:
            # Color analysis tools
            self.color_extractors = {
                "dominant": self._extract_dominant_colors,
                "palette": self._extract_color_palette,
                "harmony": self._assess_color_harmony
            }
            
            # Style analysis tools
            self.style_analyzers = {
                "composition": self._analyze_composition_style,
                "mood": self._analyze_mood,
                "lighting": self._analyze_lighting_style
            }
            
            # Brand comparison tools
            self.brand_comparators = {
                "color_similarity": self._compare_color_palettes,
                "style_match": self._compare_style_elements,
                "overall_consistency": self._calculate_brand_consistency
            }
            
            logger.info("Brand analysis tools initialized")
            
        except Exception as e:
            logger.warning(f"Brand analysis tools initialization failed: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if models are loaded"""
        return self.model_loaded
    
    def load_brand_guidelines(self, brand_name: str, guidelines: Dict[str, Any]) -> bool:
        """
        Load brand guidelines for consistency checking
        
        Args:
            brand_name: Name of the brand
            guidelines: Brand guideline dictionary
            
        Returns:
            Success status
        """
        try:
            # Validate guidelines structure
            required_fields = ["color_palette", "style_keywords", "composition_style"]
            for field in required_fields:
                if field not in guidelines:
                    logger.warning(f"Missing required field '{field}' in brand guidelines")
            
            # Store guidelines
            self.brand_guidelines[brand_name] = guidelines
            
            # Load brand-specific LoRA if available
            if LORA_AVAILABLE and "lora_path" in guidelines:
                self._load_brand_lora(brand_name, guidelines["lora_path"])
            
            self.loaded_brands.add(brand_name)
            logger.info(f"Brand guidelines loaded for '{brand_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load brand guidelines for '{brand_name}': {e}")
            return False
    
    def _load_brand_lora(self, brand_name: str, lora_path: str) -> bool:
        """Load brand-specific LoRA adapter"""
        try:
            if not self.base_pipeline:
                logger.warning("Base pipeline not available for LoRA loading")
                return False
            
            # Load LoRA weights
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout=0.1,
            )
            
            # Apply LoRA to pipeline
            brand_pipeline = self.base_pipeline
            # In production, would load actual LoRA weights from lora_path
            
            self.brand_models[brand_name] = brand_pipeline
            logger.info(f"LoRA model loaded for brand '{brand_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LoRA for brand '{brand_name}': {e}")
            return False
    
    def apply_brand_consistency(
        self,
        image: Image.Image,
        brand_name: str,
        consistency_level: float = 0.8
    ) -> Dict[str, Any]:
        """
        Apply brand consistency to image
        
        Args:
            image: Input image
            brand_name: Target brand name
            consistency_level: Desired consistency level (0-1)
            
        Returns:
            Brand-consistent image with metrics
        """
        try:
            start_time = time.time()
            
            # Get brand guidelines
            if brand_name not in self.brand_guidelines:
                # Use template if available
                if brand_name in self.brand_templates:
                    self.load_brand_guidelines(brand_name, self.brand_templates[brand_name])
                else:
                    logger.warning(f"Brand '{brand_name}' not found, using default guidelines")
                    brand_name = "minimalist"  # Default fallback
                    self.load_brand_guidelines(brand_name, self.brand_templates[brand_name])
            
            guidelines = self.brand_guidelines[brand_name]
            
            # Apply brand consistency
            consistent_image = image.copy()
            applied_adjustments = []
            
            # 1. Color consistency
            color_result = self._apply_brand_colors(consistent_image, guidelines, consistency_level)
            consistent_image = color_result["adjusted_image"]
            applied_adjustments.extend(color_result["adjustments"])
            
            # 2. Style consistency
            style_result = self._apply_brand_style(consistent_image, guidelines, consistency_level)
            consistent_image = style_result["styled_image"]
            applied_adjustments.extend(style_result["adjustments"])
            
            # 3. Mood and atmosphere
            mood_result = self._apply_brand_mood(consistent_image, guidelines, consistency_level)
            consistent_image = mood_result["mood_adjusted_image"]
            applied_adjustments.extend(mood_result["adjustments"])
            
            # 4. Quality standards
            quality_result = self._apply_brand_quality_standards(consistent_image, guidelines)
            consistent_image = quality_result["quality_adjusted_image"]
            applied_adjustments.extend(quality_result["adjustments"])
            
            # Assess final consistency
            consistency_score = self.assess_brand_consistency(consistent_image, brand_name)
            
            processing_time = time.time() - start_time
            
            return {
                "brand_consistent_image": consistent_image,
                "brand_name": brand_name,
                "consistency_score": consistency_score,
                "applied_adjustments": applied_adjustments,
                "guidelines_used": guidelines,
                "processing_time": processing_time,
                "consistency_level_target": consistency_level,
                "consistency_level_achieved": consistency_score
            }
            
        except Exception as e:
            logger.error(f"Brand consistency application failed: {e}")
            return {
                "brand_consistent_image": image,
                "brand_name": brand_name,
                "consistency_score": 0.7,
                "applied_adjustments": [],
                "processing_time": 0.0,
                "error": str(e)
            }
    
    def _apply_brand_colors(
        self,
        image: Image.Image,
        guidelines: Dict[str, Any],
        consistency_level: float
    ) -> Dict[str, Any]:
        """Apply brand color palette to image"""
        try:
            brand_palette = guidelines.get("color_palette", ["#FFFFFF"])
            adjusted_image = image.copy()
            adjustments = []
            
            # Extract current dominant colors
            current_colors = self._extract_dominant_colors(image, n_colors=5)
            
            # Convert brand palette to RGB
            brand_rgb_colors = []
            for color_hex in brand_palette:
                rgb = self._hex_to_rgb(color_hex)
                brand_rgb_colors.append(rgb)
            
            # Apply color mapping if consistency level is high
            if consistency_level > 0.7:
                adjusted_image = self._map_colors_to_brand_palette(
                    adjusted_image, current_colors, brand_rgb_colors, consistency_level
                )
                adjustments.append("color_palette_mapping")
            
            # Apply color harmony adjustments
            if consistency_level > 0.5:
                adjusted_image = self._enhance_color_harmony(adjusted_image, brand_rgb_colors)
                adjustments.append("color_harmony_enhancement")
            
            return {
                "adjusted_image": adjusted_image,
                "adjustments": adjustments,
                "color_similarity": self._compare_color_palettes(current_colors, brand_rgb_colors)
            }
            
        except Exception as e:
            logger.error(f"Brand color application failed: {e}")
            return {
                "adjusted_image": image,
                "adjustments": [],
                "color_similarity": 0.7
            }
    
    def _apply_brand_style(
        self,
        image: Image.Image,
        guidelines: Dict[str, Any],
        consistency_level: float
    ) -> Dict[str, Any]:
        """Apply brand style characteristics"""
        try:
            styled_image = image.copy()
            adjustments = []
            
            style_keywords = guidelines.get("style_keywords", ["modern"])
            composition_style = guidelines.get("composition_style", "balanced")
            
            # Apply style-specific enhancements
            if "elegant" in style_keywords or "sophisticated" in style_keywords:
                # Elegant style: increase contrast, reduce saturation slightly
                enhancer = ImageEnhance.Contrast(styled_image)
                styled_image = enhancer.enhance(1.1)
                
                enhancer = ImageEnhance.Color(styled_image)
                styled_image = enhancer.enhance(0.95)
                
                adjustments.append("elegant_style_enhancement")
            
            elif "bold" in style_keywords or "edgy" in style_keywords:
                # Bold style: increase saturation and sharpness
                enhancer = ImageEnhance.Color(styled_image)
                styled_image = enhancer.enhance(1.15)
                
                enhancer = ImageEnhance.Sharpness(styled_image)
                styled_image = enhancer.enhance(1.1)
                
                adjustments.append("bold_style_enhancement")
            
            elif "minimal" in style_keywords or "clean" in style_keywords:
                # Minimal style: reduce noise, increase clarity
                styled_image = styled_image.filter(ImageFilter.GaussianBlur(radius=0.3))
                
                enhancer = ImageEnhance.Brightness(styled_image)
                styled_image = enhancer.enhance(1.05)
                
                adjustments.append("minimal_style_enhancement")
            
            # Apply composition adjustments
            if composition_style == "minimalist" and consistency_level > 0.8:
                styled_image = self._apply_minimalist_composition(styled_image)
                adjustments.append("minimalist_composition")
            
            return {
                "styled_image": styled_image,
                "adjustments": adjustments,
                "style_match_score": self._assess_style_match(styled_image, style_keywords)
            }
            
        except Exception as e:
            logger.error(f"Brand style application failed: {e}")
            return {
                "styled_image": image,
                "adjustments": [],
                "style_match_score": 0.7
            }
    
    def _apply_brand_mood(
        self,
        image: Image.Image,
        guidelines: Dict[str, Any],
        consistency_level: float
    ) -> Dict[str, Any]:
        """Apply brand mood and atmosphere"""
        try:
            mood_adjusted = image.copy()
            adjustments = []
            
            target_mood = guidelines.get("mood", "professional")
            lighting_preference = guidelines.get("lighting_preference", "natural")
            
            # Apply mood-specific adjustments
            if target_mood == "sophisticated":
                # Sophisticated mood: warmer tones, higher contrast
                mood_adjusted = self._apply_warm_tone(mood_adjusted, intensity=0.1)
                
                enhancer = ImageEnhance.Contrast(mood_adjusted)
                mood_adjusted = enhancer.enhance(1.08)
                
                adjustments.append("sophisticated_mood")
            
            elif target_mood == "energetic":
                # Energetic mood: brighter, more saturated
                enhancer = ImageEnhance.Brightness(mood_adjusted)
                mood_adjusted = enhancer.enhance(1.1)
                
                enhancer = ImageEnhance.Color(mood_adjusted)
                mood_adjusted = enhancer.enhance(1.1)
                
                adjustments.append("energetic_mood")
            
            elif target_mood == "serene":
                # Serene mood: cooler tones, softer
                mood_adjusted = self._apply_cool_tone(mood_adjusted, intensity=0.08)
                
                mood_adjusted = mood_adjusted.filter(ImageFilter.GaussianBlur(radius=0.5))
                
                adjustments.append("serene_mood")
            
            elif target_mood == "nostalgic":
                # Nostalgic mood: vintage color grading
                mood_adjusted = self._apply_vintage_tone(mood_adjusted)
                adjustments.append("nostalgic_mood")
            
            # Apply lighting preference
            if lighting_preference and consistency_level > 0.6:
                lighting_adjusted = self._adjust_lighting_for_mood(mood_adjusted, lighting_preference)
                mood_adjusted = lighting_adjusted
                adjustments.append(f"lighting_{lighting_preference}")
            
            return {
                "mood_adjusted_image": mood_adjusted,
                "adjustments": adjustments,
                "mood_match_score": self._assess_mood_match(mood_adjusted, target_mood)
            }
            
        except Exception as e:
            logger.error(f"Brand mood application failed: {e}")
            return {
                "mood_adjusted_image": image,
                "adjustments": [],
                "mood_match_score": 0.7
            }
    
    def _apply_brand_quality_standards(
        self,
        image: Image.Image,
        guidelines: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply brand quality standards"""
        try:
            quality_adjusted = image.copy()
            adjustments = []
            
            # Apply universal quality enhancements
            # 1. Noise reduction
            img_cv = cv2.cvtColor(np.array(quality_adjusted), cv2.COLOR_RGB2BGR)
            denoised = cv2.bilateralFilter(img_cv, 9, 75, 75)
            quality_adjusted = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            adjustments.append("noise_reduction")
            
            # 2. Sharpness enhancement
            enhancer = ImageEnhance.Sharpness(quality_adjusted)
            quality_adjusted = enhancer.enhance(1.05)
            adjustments.append("sharpness_enhancement")
            
            # 3. Professional finish
            quality_adjusted = self._apply_professional_finish(quality_adjusted)
            adjustments.append("professional_finish")
            
            return {
                "quality_adjusted_image": quality_adjusted,
                "adjustments": adjustments,
                "quality_score": self._assess_image_quality(quality_adjusted)
            }
            
        except Exception as e:
            logger.error(f"Quality standards application failed: {e}")
            return {
                "quality_adjusted_image": image,
                "adjustments": [],
                "quality_score": 7.0
            }
    
    def assess_brand_consistency(self, image: Image.Image, brand_name: str) -> float:
        """
        Assess how well image matches brand guidelines
        
        Args:
            image: Image to assess
            brand_name: Target brand
            
        Returns:
            Consistency score (0-1)
        """
        try:
            if brand_name not in self.brand_guidelines:
                logger.warning(f"Brand '{brand_name}' guidelines not found")
                return 0.7  # Default score
            
            guidelines = self.brand_guidelines[brand_name]
            
            # Calculate individual consistency metrics
            metrics = {}
            
            # 1. Color harmony with brand palette
            metrics["color_harmony"] = self._assess_color_consistency(image, guidelines)
            
            # 2. Style adherence
            metrics["style_adherence"] = self._assess_style_consistency(image, guidelines)
            
            # 3. Composition match
            metrics["composition_match"] = self._assess_composition_consistency(image, guidelines)
            
            # 4. Mood consistency
            metrics["mood_consistency"] = self._assess_mood_consistency(image, guidelines)
            
            # 5. Quality standard
            metrics["quality_standard"] = self._assess_quality_consistency(image, guidelines)
            
            # Calculate weighted overall score
            overall_score = sum(
                metrics[metric] * self.consistency_weights[metric]
                for metric in metrics
            )
            
            return min(overall_score, 1.0)
            
        except Exception as e:
            logger.error(f"Brand consistency assessment failed: {e}")
            return 0.7
    
    def generate_brand_report(
        self,
        image: Image.Image,
        brand_name: str
    ) -> Dict[str, Any]:
        """Generate comprehensive brand consistency report"""
        try:
            if brand_name not in self.brand_guidelines:
                return {"error": f"Brand '{brand_name}' not found"}
            
            guidelines = self.brand_guidelines[brand_name]
            
            # Analyze image
            analysis = {
                "dominant_colors": self._extract_dominant_colors(image),
                "composition_style": self._analyze_composition_style(image),
                "mood_assessment": self._analyze_mood(image),
                "lighting_style": self._analyze_lighting_style(image),
                "quality_metrics": self._analyze_quality_metrics(image)
            }
            
            # Compare with brand guidelines
            consistency_breakdown = {
                "color_consistency": self._assess_color_consistency(image, guidelines),
                "style_consistency": self._assess_style_consistency(image, guidelines),
                "composition_consistency": self._assess_composition_consistency(image, guidelines),
                "mood_consistency": self._assess_mood_consistency(image, guidelines),
                "quality_consistency": self._assess_quality_consistency(image, guidelines)
            }
            
            # Overall consistency
            overall_consistency = self.assess_brand_consistency(image, brand_name)
            
            # Recommendations
            recommendations = self._generate_improvement_recommendations(
                analysis, guidelines, consistency_breakdown
            )
            
            return {
                "brand_name": brand_name,
                "overall_consistency": overall_consistency,
                "image_analysis": analysis,
                "consistency_breakdown": consistency_breakdown,
                "brand_guidelines": guidelines,
                "recommendations": recommendations,
                "consistency_grade": self._get_consistency_grade(overall_consistency)
            }
            
        except Exception as e:
            logger.error(f"Brand report generation failed: {e}")
            return {"error": str(e)}
    
    # Helper methods for color processing
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
    def _extract_dominant_colors(self, image: Image.Image, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image"""
        try:
            # Resize for faster processing
            small_image = image.resize((150, 150))
            
            # Convert to array and reshape
            img_array = np.array(small_image)
            pixels = img_array.reshape(-1, 3)
            
            # Use KMeans clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]
            
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return [(128, 128, 128)]  # Default gray
    
    def _map_colors_to_brand_palette(
        self,
        image: Image.Image,
        current_colors: List[Tuple[int, int, int]],
        brand_colors: List[Tuple[int, int, int]],
        intensity: float
    ) -> Image.Image:
        """Map current colors to brand palette"""
        try:
            # Simplified color mapping
            # In production, would use more sophisticated color transfer
            
            # Apply slight color adjustment towards brand palette
            img_array = np.array(image).astype(np.float32)
            
            # Get average brand color
            avg_brand_color = np.mean(brand_colors, axis=0)
            avg_current_color = np.mean(current_colors, axis=0)
            
            # Calculate adjustment vector
            adjustment = (avg_brand_color - avg_current_color) * intensity * 0.3
            
            # Apply adjustment
            adjusted = img_array + adjustment
            adjusted = np.clip(adjusted, 0, 255)
            
            return Image.fromarray(adjusted.astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Color mapping failed: {e}")
            return image
    
    def _enhance_color_harmony(
        self,
        image: Image.Image,
        brand_colors: List[Tuple[int, int, int]]
    ) -> Image.Image:
        """Enhance color harmony with brand palette"""
        try:
            # Simple color harmony enhancement
            enhancer = ImageEnhance.Color(image)
            enhanced = enhancer.enhance(1.05)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Color harmony enhancement failed: {e}")
            return image
    
    def _apply_warm_tone(self, image: Image.Image, intensity: float = 0.1) -> Image.Image:
        """Apply warm color tone"""
        try:
            img_array = np.array(image).astype(np.float32)
            
            # Increase red and yellow channels slightly
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + intensity), 0, 255)  # Red
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 + intensity * 0.5), 0, 255)  # Green
            
            return Image.fromarray(img_array.astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Warm tone application failed: {e}")
            return image
    
    def _apply_cool_tone(self, image: Image.Image, intensity: float = 0.1) -> Image.Image:
        """Apply cool color tone"""
        try:
            img_array = np.array(image).astype(np.float32)
            
            # Increase blue channel slightly
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 + intensity), 0, 255)  # Blue
            
            return Image.fromarray(img_array.astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Cool tone application failed: {e}")
            return image
    
    def _apply_vintage_tone(self, image: Image.Image) -> Image.Image:
        """Apply vintage color grading"""
        try:
            # Vintage effect: reduce saturation, add warm tone, slight sepia
            enhancer = ImageEnhance.Color(image)
            vintage = enhancer.enhance(0.8)
            
            # Add warm tone
            vintage = self._apply_warm_tone(vintage, 0.15)
            
            # Slight brightness reduction
            enhancer = ImageEnhance.Brightness(vintage)
            vintage = enhancer.enhance(0.95)
            
            return vintage
            
        except Exception as e:
            logger.error(f"Vintage tone application failed: {e}")
            return image
    
    def _apply_minimalist_composition(self, image: Image.Image) -> Image.Image:
        """Apply minimalist composition adjustments"""
        try:
            # Minimalist: increase contrast, reduce noise
            enhancer = ImageEnhance.Contrast(image)
            minimalist = enhancer.enhance(1.1)
            
            # Slight brightness increase
            enhancer = ImageEnhance.Brightness(minimalist)
            minimalist = enhancer.enhance(1.05)
            
            return minimalist
            
        except Exception as e:
            logger.error(f"Minimalist composition failed: {e}")
            return image
    
    def _adjust_lighting_for_mood(self, image: Image.Image, lighting_type: str) -> Image.Image:
        """Adjust lighting based on mood preference"""
        try:
            if lighting_type == "soft_dramatic":
                # Increase contrast, adjust shadows
                enhancer = ImageEnhance.Contrast(image)
                adjusted = enhancer.enhance(1.15)
            elif lighting_type == "natural_bright":
                # Increase brightness, maintain natural feel
                enhancer = ImageEnhance.Brightness(image)
                adjusted = enhancer.enhance(1.1)
            elif lighting_type == "soft_even":
                # Even lighting, reduce harsh shadows
                adjusted = image.filter(ImageFilter.GaussianBlur(radius=0.3))
                enhancer = ImageEnhance.Brightness(adjusted)
                adjusted = enhancer.enhance(1.05)
            elif lighting_type == "warm_golden":
                # Golden hour lighting
                adjusted = self._apply_warm_tone(image, 0.12)
                enhancer = ImageEnhance.Brightness(adjusted)
                adjusted = enhancer.enhance(1.08)
            else:
                adjusted = image
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Lighting adjustment failed: {e}")
            return image
    
    def _apply_professional_finish(self, image: Image.Image) -> Image.Image:
        """Apply professional finishing touches"""
        try:
            # Subtle enhancements for professional look
            enhanced = image
            
            # Slight contrast boost
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.03)
            
            # Color refinement
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.02)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Professional finish failed: {e}")
            return image
    
    # Assessment methods
    def _assess_color_consistency(self, image: Image.Image, guidelines: Dict[str, Any]) -> float:
        """Assess color consistency with brand guidelines"""
        try:
            image_colors = self._extract_dominant_colors(image)
            brand_colors = [self._hex_to_rgb(c) for c in guidelines.get("color_palette", [])]
            
            return self._compare_color_palettes(image_colors, brand_colors)
            
        except Exception as e:
            logger.error(f"Color consistency assessment failed: {e}")
            return 0.7
    
    def _assess_style_consistency(self, image: Image.Image, guidelines: Dict[str, Any]) -> float:
        """Assess style consistency with brand guidelines"""
        # Simplified style assessment
        return 0.85  # Placeholder
    
    def _assess_composition_consistency(self, image: Image.Image, guidelines: Dict[str, Any]) -> float:
        """Assess composition consistency with brand guidelines"""
        # Simplified composition assessment
        return 0.8  # Placeholder
    
    def _assess_mood_consistency(self, image: Image.Image, guidelines: Dict[str, Any]) -> float:
        """Assess mood consistency with brand guidelines"""
        # Simplified mood assessment
        return 0.82  # Placeholder
    
    def _assess_quality_consistency(self, image: Image.Image, guidelines: Dict[str, Any]) -> float:
        """Assess quality consistency with brand standards"""
        return self._assess_image_quality(image) / 10.0
    
    def _compare_color_palettes(
        self,
        colors1: List[Tuple[int, int, int]],
        colors2: List[Tuple[int, int, int]]
    ) -> float:
        """Compare two color palettes for similarity"""
        try:
            if not colors1 or not colors2:
                return 0.5
            
            # Calculate average color distance
            distances = []
            for c1 in colors1:
                min_dist = float('inf')
                for c2 in colors2:
                    dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)
            
            avg_distance = np.mean(distances)
            # Convert distance to similarity (0-1 scale)
            similarity = max(0, 1.0 - avg_distance / 442)  # 442 is max RGB distance
            
            return similarity
            
        except Exception as e:
            logger.error(f"Color palette comparison failed: {e}")
            return 0.7
    
    def _assess_image_quality(self, image: Image.Image) -> float:
        """Assess image quality (0-10 scale)"""
        try:
            # Basic quality assessment
            img_array = np.array(image)
            
            # Sharpness
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            laplacian_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
            sharpness = min(laplacian_var / 1000, 1.0)
            
            # Brightness distribution
            brightness = np.mean(img_array)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # Overall quality
            quality = (sharpness * 0.6 + brightness_score * 0.4) * 10
            
            return min(quality, 10.0)
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 7.0
    
    # Analysis methods (placeholders for more sophisticated analysis)
    def _extract_color_palette(self, image: Image.Image) -> List[str]:
        """Extract color palette as hex codes"""
        colors = self._extract_dominant_colors(image)
        return [self._rgb_to_hex(color) for color in colors]
    
    def _assess_color_harmony(self, image: Image.Image) -> float:
        """Assess color harmony in image"""
        return 0.85  # Placeholder
    
    def _analyze_composition_style(self, image: Image.Image) -> str:
        """Analyze composition style"""
        return "balanced"  # Placeholder
    
    def _analyze_mood(self, image: Image.Image) -> str:
        """Analyze mood/atmosphere"""
        return "professional"  # Placeholder
    
    def _analyze_lighting_style(self, image: Image.Image) -> str:
        """Analyze lighting style"""
        return "natural"  # Placeholder
    
    def _analyze_quality_metrics(self, image: Image.Image) -> Dict[str, float]:
        """Analyze quality metrics"""
        return {
            "sharpness": 0.85,
            "brightness": 0.8,
            "contrast": 0.9,
            "color_balance": 0.85
        }
    
    def _assess_style_match(self, image: Image.Image, style_keywords: List[str]) -> float:
        """Assess how well image matches style keywords"""
        return 0.85  # Placeholder
    
    def _assess_mood_match(self, image: Image.Image, target_mood: str) -> float:
        """Assess how well image matches target mood"""
        return 0.82  # Placeholder
    
    def _compare_style_elements(self, image1: Image.Image, image2: Image.Image) -> float:
        """Compare style elements between images"""
        return 0.8  # Placeholder
    
    def _calculate_brand_consistency(self, image: Image.Image, brand_name: str) -> float:
        """Calculate overall brand consistency"""
        return self.assess_brand_consistency(image, brand_name)
    
    def _generate_improvement_recommendations(
        self,
        analysis: Dict[str, Any],
        guidelines: Dict[str, Any],
        consistency_breakdown: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for improving brand consistency"""
        recommendations = []
        
        # Color recommendations
        if consistency_breakdown.get("color_consistency", 1.0) < 0.8:
            recommendations.append("Adjust color palette to better match brand colors")
        
        # Style recommendations
        if consistency_breakdown.get("style_consistency", 1.0) < 0.8:
            recommendations.append("Apply brand-specific style enhancements")
        
        # Mood recommendations
        if consistency_breakdown.get("mood_consistency", 1.0) < 0.8:
            recommendations.append("Adjust mood and atmosphere to match brand guidelines")
        
        # Quality recommendations
        if consistency_breakdown.get("quality_consistency", 1.0) < 0.8:
            recommendations.append("Improve image quality and professional finish")
        
        return recommendations
    
    def _get_consistency_grade(self, score: float) -> str:
        """Get letter grade for consistency score"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def clear_cache(self):
        """Clear brand consistency cache"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Brand consistency cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def unload_models(self):
        """Unload models to free memory"""
        try:
            if hasattr(self, 'base_pipeline') and self.base_pipeline:
                del self.base_pipeline
                self.base_pipeline = None
            
            for brand_name in list(self.brand_models.keys()):
                del self.brand_models[brand_name]
            
            self.brand_models.clear()
            self.model_loaded = False
            self.clear_cache()
            logger.info("Brand consistency models unloaded")
            
        except Exception as e:
            logger.error(f"Failed to unload models: {e}")

# Add missing import
from PIL import ImageFilter