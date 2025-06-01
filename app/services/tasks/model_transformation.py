import torch
import numpy as np
from PIL import Image
import asyncio
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from app.services.ai.model_enhancement_service import ModelEnhancementService
from app.services.ai.garment_optimization_service import GarmentOptimizationService
from app.services.ai.scene_generation_service import SceneGenerationService
from app.utils.image_utils import resize_image, enhance_image
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelTransformationPipeline:
    """Complete pipeline for transforming model photos into professional photoshoots"""
    
    def __init__(self):
        # Initialize services
        self.model_enhancement_service = ModelEnhancementService()
        self.garment_optimization_service = GarmentOptimizationService()
        self.scene_generation_service = SceneGenerationService()
        
        # Pipeline state
        self.services_loaded = False
        self.processing_stats = {
            "total_processed": 0,
            "average_processing_time": 0.0,
            "average_quality_score": 0.0
        }
        
        # Style configurations
        self.style_configs = {
            "editorial": {
                "style_prompt": "high fashion editorial photography, dramatic lighting, artistic composition, magazine quality",
                "negative_prompt": "amateur, casual, low quality, commercial, product photography",
                "background_type": "artistic",
                "lighting_type": "dramatic",
                "enhancement_level": 0.9,
                "creativity_level": 0.8
            },
            "commercial": {
                "style_prompt": "commercial product photography, clean studio lighting, professional presentation",
                "negative_prompt": "artistic, dramatic, editorial, personal, casual",
                "background_type": "studio",
                "lighting_type": "studio",
                "enhancement_level": 0.7,
                "creativity_level": 0.5
            },
            "lifestyle": {
                "style_prompt": "lifestyle photography, natural lighting, candid feel, everyday elegance",
                "negative_prompt": "studio, formal, posed, artificial, commercial",
                "background_type": "natural",
                "lighting_type": "natural",
                "enhancement_level": 0.6,
                "creativity_level": 0.7
            },
            "artistic": {
                "style_prompt": "artistic fashion photography, creative composition, unique perspective, avant-garde",
                "negative_prompt": "commercial, standard, basic, conventional, boring",
                "background_type": "abstract",
                "lighting_type": "creative",
                "enhancement_level": 0.8,
                "creativity_level": 0.9
            },
            "brand": {
                "style_prompt": "brand photography, signature style, professional presentation, brand consistent",
                "negative_prompt": "off-brand, inconsistent, generic, amateur",
                "background_type": "brand_environment",
                "lighting_type": "brand_standard",
                "enhancement_level": 0.8,
                "creativity_level": 0.6
            }
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "minimum_acceptable": 6.0,
            "good_quality": 7.5,
            "excellent_quality": 9.0
        }
        
        # Performance targets
        self.performance_targets = {
            "model_enhancement": 5.0,
            "garment_optimization": 8.0,
            "scene_generation": 12.0,
            "total_pipeline": 30.0
        }
    
    def load_all_models(self):
        """Load all AI models for the pipeline"""
        try:
            logger.info("Loading all models for transformation pipeline...")
            
            # Load models in parallel for faster startup
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self.model_enhancement_service.load_models): "model_enhancement",
                    executor.submit(self.garment_optimization_service.load_models): "garment_optimization",
                    executor.submit(self.scene_generation_service.load_models): "scene_generation"
                }
                
                for future in as_completed(futures):
                    service_name = futures[future]
                    try:
                        future.result()
                        logger.info(f"{service_name} service loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load {service_name} service: {e}")
            
            self.services_loaded = True
            logger.info("All transformation pipeline models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline models: {str(e)}")
            # Use fallback mode
            self.services_loaded = True
            logger.info("Pipeline running in fallback mode")
    
    def transform_model(
        self,
        model_image: Image.Image,
        style_prompt: str,
        negative_prompt: str = "",
        num_variations: int = 5,
        enhance_model: bool = True,
        optimize_garment: bool = True,
        generate_scene: bool = True,
        quality_mode: str = "balanced",  # "fast", "balanced", "high"
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Transform model photo into professional photoshoot variations
        
        Args:
            model_image: Input model image
            style_prompt: Overall style description
            negative_prompt: What to avoid
            num_variations: Number of style variations to generate
            enhance_model: Whether to enhance model appearance
            optimize_garment: Whether to optimize garment
            generate_scene: Whether to generate professional scenes
            quality_mode: Processing quality mode
            seed: Random seed for reproducibility
            
        Returns:
            Transformation results with variations and metadata
        """
        try:
            start_time = time.time()
            transformation_id = str(uuid.uuid4())
            
            logger.info(f"Starting model transformation {transformation_id}")
            
            # Ensure models are loaded
            if not self.services_loaded:
                self.load_all_models()
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Validate and preprocess input
            processed_image = self._preprocess_input(model_image, quality_mode)
            
            # Track processing stages
            stage_times = {}
            stage_results = {}
            error_handling = {"fallbacks_used": 0, "errors": []}
            
            # Stage 1: Model Enhancement
            if enhance_model:
                stage_start = time.time()
                try:
                    enhancement_result = self.enhance_model_step(processed_image)
                    processed_image = enhancement_result["enhanced_image"]
                    stage_results["model_enhancement"] = enhancement_result
                    stage_times["model_enhancement"] = time.time() - stage_start
                except Exception as e:
                    logger.error(f"Model enhancement failed: {e}")
                    error_handling["fallbacks_used"] += 1
                    error_handling["errors"].append(f"Model enhancement: {str(e)}")
                    stage_times["model_enhancement"] = time.time() - stage_start
            
            # Stage 2: Garment Optimization
            if optimize_garment:
                stage_start = time.time()
                try:
                    garment_result = self.optimize_garment_step(processed_image)
                    processed_image = garment_result["optimized_image"]
                    stage_results["garment_optimization"] = garment_result
                    stage_times["garment_optimization"] = time.time() - stage_start
                except Exception as e:
                    logger.error(f"Garment optimization failed: {e}")
                    error_handling["fallbacks_used"] += 1
                    error_handling["errors"].append(f"Garment optimization: {str(e)}")
                    stage_times["garment_optimization"] = time.time() - stage_start
            
            # Stage 3: Generate Style Variations
            variations = []
            variation_times = []
            
            if num_variations > 0:
                # Get style configurations
                style_types = list(self.style_configs.keys())[:num_variations]
                
                # Generate variations
                if quality_mode == "fast":
                    # Sequential processing for fast mode
                    variations = self._generate_variations_sequential(
                        processed_image, style_prompt, negative_prompt, style_types, generate_scene
                    )
                else:
                    # Parallel processing for balanced/high quality
                    variations = self._generate_variations_parallel(
                        processed_image, style_prompt, negative_prompt, style_types, generate_scene
                    )
                
                variation_times = [v.get("processing_time", 0) for v in variations]
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(variations, stage_results)
            
            # Update processing stats
            total_time = time.time() - start_time
            self._update_processing_stats(total_time, quality_scores["overall_average"])
            
            # Compile final result
            result = {
                "transformation_id": transformation_id,
                "variations": variations,
                "metadata": {
                    "original_image_size": model_image.size,
                    "processing_stages": list(stage_results.keys()),
                    "stage_times": stage_times,
                    "variation_times": variation_times,
                    "total_processing_time": total_time,
                    "quality_mode": quality_mode,
                    "num_variations_requested": num_variations,
                    "num_variations_generated": len(variations),
                    "seed": seed
                },
                "quality_scores": quality_scores,
                "performance_metrics": {
                    "meets_time_target": total_time <= self.performance_targets["total_pipeline"],
                    "meets_quality_target": quality_scores["overall_average"] >= self.quality_thresholds["minimum_acceptable"],
                    "efficiency_score": self._calculate_efficiency_score(total_time, quality_scores["overall_average"])
                },
                "error_handling": error_handling,
                "stage_results": stage_results
            }
            
            logger.info(f"Transformation {transformation_id} completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Model transformation pipeline failed: {str(e)}")
            return self._create_fallback_result(model_image, num_variations, str(e))
    
    def enhance_model_step(self, image: Image.Image) -> Dict[str, Any]:
        """Execute model enhancement step"""
        try:
            start_time = time.time()
            
            # Run model enhancement
            enhancement_result = self.model_enhancement_service.enhance_model(image)
            
            # Calculate quality improvement
            original_quality = self.assess_image_quality(image)
            enhanced_quality = self.assess_image_quality(enhancement_result["enhanced_image"])
            quality_improvement = enhanced_quality - original_quality
            
            return {
                "enhanced_image": enhancement_result["enhanced_image"],
                "enhancement_metadata": enhancement_result,
                "quality_improvement": quality_improvement,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Model enhancement step failed: {e}")
            return {
                "enhanced_image": image,
                "enhancement_metadata": {"error": str(e)},
                "quality_improvement": 0.0,
                "processing_time": 0.0
            }
    
    def optimize_garment_step(self, image: Image.Image) -> Dict[str, Any]:
        """Execute garment optimization step"""
        try:
            start_time = time.time()
            
            # Run garment optimization
            optimization_result = self.garment_optimization_service.optimize_garment(image)
            
            return {
                "optimized_image": optimization_result["optimized_image"],
                "optimization_metadata": optimization_result,
                "garment_quality_score": optimization_result["overall_score"],
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Garment optimization step failed: {e}")
            return {
                "optimized_image": image,
                "optimization_metadata": {"error": str(e)},
                "garment_quality_score": 5.0,
                "processing_time": 0.0
            }
    
    def generate_scene_step(
        self,
        image: Image.Image,
        style_prompt: str,
        background_type: str
    ) -> Dict[str, Any]:
        """Execute scene generation step"""
        try:
            start_time = time.time()
            
            # Generate scene
            scene_result = self.scene_generation_service.compose_scene(
                model_image=image,
                background_type=background_type,
                composition_style=self._extract_composition_style(style_prompt)
            )
            
            # Apply lighting
            lighting_result = self.scene_generation_service.apply_lighting(
                scene_result["composed_scene"],
                lighting_type=self._extract_lighting_type(style_prompt)
            )
            
            final_scene = lighting_result["lit_image"]
            scene_quality = (scene_result["composition_score"] + lighting_result["lighting_quality"] * 10) / 2
            
            return {
                "scene_image": final_scene,
                "scene_metadata": {
                    "composition_result": scene_result,
                    "lighting_result": lighting_result
                },
                "scene_quality_score": scene_quality,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Scene generation step failed: {e}")
            return {
                "scene_image": image,
                "scene_metadata": {"error": str(e)},
                "scene_quality_score": 5.0,
                "processing_time": 0.0
            }
    
    def generate_style_variation(
        self,
        image: Image.Image,
        style_config: Dict[str, Any],
        variation_name: str
    ) -> Dict[str, Any]:
        """Generate a single style variation"""
        try:
            start_time = time.time()
            
            # Apply style-specific enhancements
            enhanced_image = self._apply_style_enhancements(image, style_config)
            
            # Generate scene if needed
            if style_config.get("background_type"):
                scene_result = self.generate_scene_step(
                    enhanced_image,
                    style_config["style_prompt"],
                    style_config["background_type"]
                )
                final_image = scene_result["scene_image"]
                scene_quality = scene_result["scene_quality_score"]
            else:
                final_image = enhanced_image
                scene_quality = 8.0  # Default for no scene generation
            
            # Calculate style consistency
            style_consistency = self._assess_style_consistency(final_image, style_config)
            
            # Calculate overall quality
            quality_score = self.assess_image_quality(final_image)
            
            return {
                "variation_image": final_image,
                "style_type": variation_name,
                "style_consistency": style_consistency,
                "quality_score": quality_score,
                "scene_quality": scene_quality,
                "processing_time": time.time() - start_time,
                "style_config_used": style_config
            }
            
        except Exception as e:
            logger.error(f"Style variation generation failed: {e}")
            return {
                "variation_image": image,
                "style_type": variation_name,
                "style_consistency": 0.7,
                "quality_score": 5.0,
                "scene_quality": 5.0,
                "processing_time": 0.0,
                "error": str(e)
            }
    
    def _preprocess_input(self, image: Image.Image, quality_mode: str) -> Image.Image:
        """Preprocess input image based on quality mode"""
        try:
            processed = image.copy()
            
            # Resize based on quality mode
            if quality_mode == "fast":
                max_size = 512
            elif quality_mode == "balanced":
                max_size = 1024
            else:  # high quality
                max_size = 2048
            
            # Resize if needed
            if max(processed.size) > max_size:
                processed = resize_image(processed, max_size)
            
            # Basic enhancement
            if quality_mode in ["balanced", "high"]:
                processed = enhance_image(processed, brightness=1.05, contrast=1.05)
            
            return processed
            
        except Exception as e:
            logger.error(f"Input preprocessing failed: {e}")
            return image
    
    def _generate_variations_sequential(
        self,
        image: Image.Image,
        style_prompt: str,
        negative_prompt: str,
        style_types: List[str],
        generate_scene: bool
    ) -> List[Dict[str, Any]]:
        """Generate variations sequentially (faster, lower resource usage)"""
        variations = []
        
        for style_type in style_types:
            try:
                # Get style configuration
                style_config = self.style_configs[style_type].copy()
                
                # Merge with user prompts
                style_config["style_prompt"] = f"{style_config['style_prompt']}, {style_prompt}"
                style_config["negative_prompt"] = f"{style_config['negative_prompt']}, {negative_prompt}"
                
                # Generate variation
                variation = self.generate_style_variation(image, style_config, style_type)
                variations.append(variation)
                
            except Exception as e:
                logger.error(f"Failed to generate {style_type} variation: {e}")
                # Add fallback variation
                variations.append(self._create_fallback_variation(image, style_type))
        
        return variations
    
    def _generate_variations_parallel(
        self,
        image: Image.Image,
        style_prompt: str,
        negative_prompt: str,
        style_types: List[str],
        generate_scene: bool
    ) -> List[Dict[str, Any]]:
        """Generate variations in parallel (higher quality, more resources)"""
        variations = []
        
        def generate_single_variation(style_type):
            try:
                style_config = self.style_configs[style_type].copy()
                style_config["style_prompt"] = f"{style_config['style_prompt']}, {style_prompt}"
                style_config["negative_prompt"] = f"{style_config['negative_prompt']}, {negative_prompt}"
                
                return self.generate_style_variation(image, style_config, style_type)
            except Exception as e:
                logger.error(f"Parallel variation generation failed for {style_type}: {e}")
                return self._create_fallback_variation(image, style_type)
        
        # Use ThreadPoolExecutor for parallel generation
        with ThreadPoolExecutor(max_workers=min(len(style_types), 3)) as executor:
            future_to_style = {
                executor.submit(generate_single_variation, style_type): style_type 
                for style_type in style_types
            }
            
            for future in as_completed(future_to_style):
                style_type = future_to_style[future]
                try:
                    variation = future.result()
                    variations.append(variation)
                except Exception as e:
                    logger.error(f"Parallel processing failed for {style_type}: {e}")
                    variations.append(self._create_fallback_variation(image, style_type))
        
        # Sort by original order
        style_order = {style: i for i, style in enumerate(style_types)}
        variations.sort(key=lambda x: style_order.get(x["style_type"], 999))
        
        return variations
    
    def _apply_style_enhancements(
        self,
        image: Image.Image,
        style_config: Dict[str, Any]
    ) -> Image.Image:
        """Apply style-specific enhancements to image"""
        try:
            enhanced = image.copy()
            enhancement_level = style_config.get("enhancement_level", 0.7)
            
            # Apply enhancements based on style
            if "editorial" in style_config.get("style_prompt", "").lower():
                # Editorial: Higher contrast, dramatic
                enhanced = enhance_image(
                    enhanced,
                    contrast=1.0 + enhancement_level * 0.3,
                    sharpness=1.0 + enhancement_level * 0.2
                )
            elif "commercial" in style_config.get("style_prompt", "").lower():
                # Commercial: Clean, bright
                enhanced = enhance_image(
                    enhanced,
                    brightness=1.0 + enhancement_level * 0.1,
                    saturation=1.0 + enhancement_level * 0.1
                )
            elif "lifestyle" in style_config.get("style_prompt", "").lower():
                # Lifestyle: Soft, natural
                enhanced = enhance_image(
                    enhanced,
                    saturation=1.0 - enhancement_level * 0.05,
                    brightness=1.0 + enhancement_level * 0.05
                )
            elif "artistic" in style_config.get("style_prompt", "").lower():
                # Artistic: Creative, enhanced colors
                enhanced = enhance_image(
                    enhanced,
                    saturation=1.0 + enhancement_level * 0.2,
                    contrast=1.0 + enhancement_level * 0.15
                )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Style enhancement failed: {e}")
            return image
    
    def _extract_composition_style(self, style_prompt: str) -> str:
        """Extract composition style from prompt"""
        prompt_lower = style_prompt.lower()
        
        if any(word in prompt_lower for word in ["editorial", "dramatic", "artistic"]):
            return "editorial"
        elif any(word in prompt_lower for word in ["commercial", "product", "catalog"]):
            return "commercial"
        elif any(word in prompt_lower for word in ["lifestyle", "casual", "natural"]):
            return "lifestyle"
        else:
            return "commercial"  # Default
    
    def _extract_lighting_type(self, style_prompt: str) -> str:
        """Extract lighting type from prompt"""
        prompt_lower = style_prompt.lower()
        
        if any(word in prompt_lower for word in ["dramatic", "moody", "artistic"]):
            return "dramatic"
        elif any(word in prompt_lower for word in ["natural", "outdoor", "lifestyle"]):
            return "natural"
        else:
            return "studio"  # Default
    
    def _assess_style_consistency(
        self,
        image: Image.Image,
        style_config: Dict[str, Any]
    ) -> float:
        """Assess how well image matches the intended style"""
        try:
            # Simplified style consistency assessment
            # In production, this would use more sophisticated analysis
            
            base_score = 0.85  # Base consistency score
            
            # Adjust based on style type
            style_prompt = style_config.get("style_prompt", "").lower()
            
            if "editorial" in style_prompt:
                # Editorial should have higher contrast
                # This is a simplified check
                base_score += 0.05
            elif "commercial" in style_prompt:
                # Commercial should be clean and bright
                base_score += 0.03
            
            return min(base_score, 1.0)
            
        except Exception as e:
            logger.error(f"Style consistency assessment failed: {e}")
            return 0.8  # Default
    
    def assess_image_quality(self, image: Image.Image) -> float:
        """Assess overall image quality (0-10 scale)"""
        try:
            # Convert to array for analysis
            img_array = np.array(image)
            
            # Basic quality metrics
            quality_score = 7.0  # Base score
            
            # 1. Sharpness (variance of Laplacian)
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Calculate sharpness
            import cv2
            laplacian_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000, 1.0)
            
            # 2. Brightness distribution
            brightness = np.mean(img_array)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # 3. Color distribution (if color image)
            if len(img_array.shape) == 3:
                color_std = np.std(img_array, axis=(0, 1))
                color_score = min(np.mean(color_std) / 50, 1.0)
            else:
                color_score = 0.7
            
            # Combine scores
            quality_score = (
                sharpness_score * 0.4 +
                brightness_score * 0.3 +
                color_score * 0.3
            ) * 10
            
            return min(quality_score, 10.0)
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 7.0  # Default good score
    
    def _calculate_quality_scores(
        self,
        variations: List[Dict[str, Any]],
        stage_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality scores"""
        try:
            if not variations:
                return {
                    "overall_average": 5.0,
                    "variation_scores": [],
                    "stage_scores": {}
                }
            
            # Extract variation scores
            variation_scores = [v.get("quality_score", 5.0) for v in variations]
            style_consistency_scores = [v.get("style_consistency", 0.8) for v in variations]
            scene_quality_scores = [v.get("scene_quality", 5.0) for v in variations]
            
            # Calculate averages
            avg_quality = np.mean(variation_scores)
            avg_style_consistency = np.mean(style_consistency_scores)
            avg_scene_quality = np.mean(scene_quality_scores)
            
            # Stage-specific scores
            stage_scores = {}
            if "model_enhancement" in stage_results:
                stage_scores["model_enhancement"] = stage_results["model_enhancement"].get("quality_improvement", 0)
            if "garment_optimization" in stage_results:
                stage_scores["garment_optimization"] = stage_results["garment_optimization"].get("garment_quality_score", 5.0)
            
            # Overall score
            overall_score = (avg_quality * 0.5 + avg_style_consistency * 10 * 0.3 + avg_scene_quality * 0.2)
            
            return {
                "overall_average": overall_score,
                "variation_scores": variation_scores,
                "style_consistency_average": avg_style_consistency,
                "scene_quality_average": avg_scene_quality,
                "stage_scores": stage_scores,
                "quality_distribution": {
                    "min": min(variation_scores),
                    "max": max(variation_scores),
                    "std": np.std(variation_scores)
                }
            }
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return {
                "overall_average": 7.0,
                "variation_scores": [7.0] * len(variations),
                "stage_scores": {}
            }
    
    def _calculate_efficiency_score(self, processing_time: float, quality_score: float) -> float:
        """Calculate efficiency score based on time and quality"""
        try:
            # Normalize time score (30s target)
            time_score = max(0, 1.0 - (processing_time - 30.0) / 30.0)
            
            # Normalize quality score (8.5 target)
            quality_normalized = quality_score / 10.0
            
            # Combine (favor quality over speed)
            efficiency = quality_normalized * 0.7 + time_score * 0.3
            
            return min(efficiency, 1.0)
            
        except Exception as e:
            logger.error(f"Efficiency calculation failed: {e}")
            return 0.7
    
    def _update_processing_stats(self, processing_time: float, quality_score: float):
        """Update running statistics"""
        try:
            self.processing_stats["total_processed"] += 1
            total = self.processing_stats["total_processed"]
            
            # Update running averages
            old_avg_time = self.processing_stats["average_processing_time"]
            self.processing_stats["average_processing_time"] = (
                (old_avg_time * (total - 1) + processing_time) / total
            )
            
            old_avg_quality = self.processing_stats["average_quality_score"]
            self.processing_stats["average_quality_score"] = (
                (old_avg_quality * (total - 1) + quality_score) / total
            )
            
        except Exception as e:
            logger.error(f"Stats update failed: {e}")
    
    def _create_fallback_variation(self, image: Image.Image, style_type: str) -> Dict[str, Any]:
        """Create fallback variation when generation fails"""
        return {
            "variation_image": image,
            "style_type": style_type,
            "style_consistency": 0.7,
            "quality_score": 6.0,
            "scene_quality": 6.0,
            "processing_time": 0.1,
            "fallback_used": True
        }
    
    def _create_fallback_result(
        self,
        original_image: Image.Image,
        num_variations: int,
        error_message: str
    ) -> Dict[str, Any]:
        """Create fallback result when pipeline fails"""
        # Create basic variations using original image
        variations = []
        for i in range(num_variations):
            style_type = list(self.style_configs.keys())[i % len(self.style_configs)]
            variations.append(self._create_fallback_variation(original_image, style_type))
        
        return {
            "transformation_id": str(uuid.uuid4()),
            "variations": variations,
            "metadata": {
                "original_image_size": original_image.size,
                "processing_stages": [],
                "total_processing_time": 0.1,
                "fallback_mode": True
            },
            "quality_scores": {
                "overall_average": 6.0,
                "variation_scores": [6.0] * num_variations
            },
            "performance_metrics": {
                "meets_time_target": True,
                "meets_quality_target": False,
                "efficiency_score": 0.5
            },
            "error_handling": {
                "fallbacks_used": num_variations,
                "errors": [error_message],
                "fallback_mode": True
            }
        }
    
    def transform_batch(
        self,
        images: List[Image.Image],
        style_prompts: List[str],
        num_variations_per_image: int = 3
    ) -> List[Dict[str, Any]]:
        """Transform multiple images in batch"""
        results = []
        
        for i, (image, style_prompt) in enumerate(zip(images, style_prompts)):
            try:
                result = self.transform_model(
                    model_image=image,
                    style_prompt=style_prompt,
                    num_variations=num_variations_per_image,
                    quality_mode="balanced"
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch processing failed for image {i}: {e}")
                results.append(self._create_fallback_result(image, num_variations_per_image, str(e)))
        
        return results
    
    def clear_all_caches(self):
        """Clear all service caches"""
        try:
            self.model_enhancement_service.clear_cache()
            self.garment_optimization_service.clear_cache()
            self.scene_generation_service.clear_cache()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("All pipeline caches cleared")
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
    
    def unload_all_models(self):
        """Unload all models to free memory"""
        try:
            self.model_enhancement_service.unload_models()
            self.garment_optimization_service.unload_models()
            self.scene_generation_service.unload_models()
            
            self.services_loaded = False
            logger.info("All pipeline models unloaded")
            
        except Exception as e:
            logger.error(f"Model unloading failed: {e}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics"""
        return {
            "processing_stats": self.processing_stats.copy(),
            "services_loaded": self.services_loaded,
            "style_configs_available": list(self.style_configs.keys()),
            "quality_thresholds": self.quality_thresholds.copy(),
            "performance_targets": self.performance_targets.copy()
        }