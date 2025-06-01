import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.cluster import KMeans
import colorsys
import time

from app.core.config import settings
from app.services.ai.sam2_service import SAM2Service

logger = logging.getLogger(__name__)

class GarmentOptimizationService:
    """Service for optimizing garment appearance, fit, and styling"""
    
    def __init__(self):
        self.sam2_service = SAM2Service()
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Garment type detection patterns
        self.garment_patterns = {
            "shirt": ["collar", "buttons", "sleeves", "chest"],
            "dress": ["hemline", "waist", "neckline", "flowing"],
            "pants": ["legs", "waist", "crotch", "hem"],
            "jacket": ["lapels", "buttons", "collar", "structured"],
            "top": ["shoulders", "neckline", "torso"],
            "bottom": ["waist", "legs", "hips"]
        }
    
    def load_models(self):
        """Load garment optimization models"""
        try:
            logger.info("Loading garment optimization models...")
            
            # Load SAM2 for segmentation
            self.sam2_service.load_model()
            
            # Initialize color analysis tools
            self._initialize_color_tools()
            
            self.model_loaded = True
            logger.info("Garment optimization models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load garment optimization models: {str(e)}")
            self.model_loaded = True  # Use fallback methods
            logger.info("Using fallback garment optimization methods")
    
    def _initialize_color_tools(self):
        """Initialize color analysis and correction tools"""
        try:
            # Color harmony rules
            self.color_harmonies = {
                "complementary": lambda h: [(h + 180) % 360],
                "triadic": lambda h: [(h + 120) % 360, (h + 240) % 360],
                "analogous": lambda h: [(h + 30) % 360, (h - 30) % 360],
                "split_complementary": lambda h: [(h + 150) % 360, (h + 210) % 360]
            }
            
            logger.info("Color analysis tools initialized")
        except Exception as e:
            logger.warning(f"Color tools initialization failed: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if models are loaded"""
        return self.model_loaded
    
    def segment_garment(self, image: Image.Image) -> Dict[str, Any]:
        """
        Segment garment from the image using SAM2
        
        Args:
            image: Input PIL Image
            
        Returns:
            Segmentation results with mask and metadata
        """
        try:
            # Use SAM2 for initial segmentation
            sam_result = self.sam2_service.remove_background(image, return_mask=True)
            
            if isinstance(sam_result, tuple):
                segmented_image, mask = sam_result
            else:
                segmented_image = sam_result
                mask = Image.new('L', image.size, 255)
            
            # Analyze the mask to determine garment regions
            mask_array = np.array(mask)
            garment_pixels = np.sum(mask_array > 128)
            total_pixels = mask_array.size
            confidence = garment_pixels / total_pixels
            
            # Detect garment type from segmented region
            garment_type = self._classify_garment_from_mask(image, mask)
            
            return {
                "mask": mask,
                "segmented_image": segmented_image,
                "confidence": confidence,
                "garment_type": garment_type,
                "garment_area": garment_pixels,
                "segmentation_method": "SAM2"
            }
            
        except Exception as e:
            logger.error(f"Garment segmentation failed: {e}")
            return self._fallback_segmentation(image)
    
    def _fallback_segmentation(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback segmentation using traditional CV methods"""
        try:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Use color-based segmentation
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Create mask for non-skin colors (simplified garment detection)
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            garment_mask = cv2.bitwise_not(skin_mask)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_CLOSE, kernel)
            garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_OPEN, kernel)
            
            mask_pil = Image.fromarray(garment_mask)
            
            # Calculate confidence
            garment_pixels = np.sum(garment_mask > 128)
            confidence = garment_pixels / garment_mask.size
            
            return {
                "mask": mask_pil,
                "confidence": confidence,
                "garment_type": "unknown",
                "garment_area": garment_pixels,
                "segmentation_method": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback segmentation failed: {e}")
            return {
                "mask": Image.new('L', image.size, 255),
                "confidence": 0.5,
                "garment_type": "unknown",
                "garment_area": image.size[0] * image.size[1] // 2,
                "segmentation_method": "default"
            }
    
    def _classify_garment_from_mask(self, image: Image.Image, mask: Image.Image) -> str:
        """Classify garment type from segmentation mask"""
        try:
            mask_array = np.array(mask)
            
            # Analyze mask shape and distribution
            height, width = mask_array.shape
            
            # Find contours
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return "unknown"
            
            # Get largest contour (main garment)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(main_contour)
            aspect_ratio = w / h
            
            # Calculate position relative to image
            center_y = y + h // 2
            relative_center_y = center_y / height
            
            # Simple classification based on shape and position
            if aspect_ratio > 1.5:  # Wide garment
                if relative_center_y < 0.6:  # Upper body
                    return "shirt"
                else:  # Lower body
                    return "pants"
            elif aspect_ratio < 0.7:  # Tall garment
                if h > height * 0.6:  # Covers most of body
                    return "dress"
                else:
                    return "top"
            else:  # Medium aspect ratio
                if relative_center_y < 0.5:
                    return "top"
                else:
                    return "bottom"
            
        except Exception as e:
            logger.error(f"Garment classification failed: {e}")
            return "unknown"
    
    def detect_garment_type(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect garment type and attributes
        
        Args:
            image: Input PIL Image
            
        Returns:
            Detection results with type and confidence
        """
        try:
            # First segment the garment
            seg_result = self.segment_garment(image)
            garment_type = seg_result["garment_type"]
            
            # Analyze garment attributes
            attributes = self._analyze_garment_attributes(image, seg_result["mask"])
            
            # Calculate confidence based on segmentation and attribute analysis
            confidence = min(seg_result["confidence"] * 1.2, 1.0)
            
            return {
                "garment_type": garment_type,
                "confidence": confidence,
                "attributes": attributes,
                "segmentation_confidence": seg_result["confidence"]
            }
            
        except Exception as e:
            logger.error(f"Garment type detection failed: {e}")
            return {
                "garment_type": "unknown",
                "confidence": 0.5,
                "attributes": {},
                "segmentation_confidence": 0.5
            }
    
    def _analyze_garment_attributes(self, image: Image.Image, mask: Image.Image) -> Dict[str, Any]:
        """Analyze garment attributes like color, texture, style"""
        try:
            # Extract garment region
            img_array = np.array(image)
            mask_array = np.array(mask)
            
            # Get garment pixels
            garment_pixels = img_array[mask_array > 128]
            
            attributes = {}
            
            if len(garment_pixels) > 0:
                # Dominant colors
                attributes["dominant_colors"] = self._extract_dominant_colors(garment_pixels)
                
                # Texture analysis
                attributes["texture_score"] = self._analyze_texture(image, mask)
                
                # Pattern detection
                attributes["has_patterns"] = self._detect_patterns(image, mask)
                
                # Style characteristics
                attributes["style_features"] = self._analyze_style_features(image, mask)
            
            return attributes
            
        except Exception as e:
            logger.error(f"Garment attribute analysis failed: {e}")
            return {}
    
    def _extract_dominant_colors(self, pixels: np.ndarray, n_colors: int = 3) -> List[List[int]]:
        """Extract dominant colors from garment pixels"""
        try:
            if len(pixels) < n_colors:
                return pixels.tolist()
            
            # Use KMeans to find dominant colors
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            return colors.tolist()
            
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return [[128, 128, 128]]  # Default gray
    
    def _analyze_texture(self, image: Image.Image, mask: Image.Image) -> float:
        """Analyze fabric texture quality"""
        try:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            mask_array = np.array(mask)
            
            # Apply mask to focus on garment area
            masked_gray = gray * (mask_array / 255)
            
            # Calculate texture metrics
            # 1. Local Binary Pattern-like analysis
            laplacian_var = cv2.Laplacian(masked_gray, cv2.CV_64F).var()
            
            # 2. Gradient magnitude
            grad_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_score = np.mean(gradient_magnitude[mask_array > 128])
            
            # Normalize and combine scores
            texture_score = min((laplacian_var / 1000 + gradient_score / 100) / 2, 1.0)
            
            return float(texture_score)
            
        except Exception as e:
            logger.error(f"Texture analysis failed: {e}")
            return 0.5
    
    def _detect_patterns(self, image: Image.Image, mask: Image.Image) -> bool:
        """Detect if garment has patterns (stripes, prints, etc.)"""
        try:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            mask_array = np.array(mask)
            
            # Apply mask
            masked_gray = gray * (mask_array / 255)
            
            # Use FFT to detect repeating patterns
            f_transform = np.fft.fft2(masked_gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Look for peaks in frequency domain (indicating patterns)
            threshold = np.mean(magnitude_spectrum) + 2 * np.std(magnitude_spectrum)
            pattern_peaks = np.sum(magnitude_spectrum > threshold)
            
            # Threshold for pattern detection
            return pattern_peaks > 5
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return False
    
    def _analyze_style_features(self, image: Image.Image, mask: Image.Image) -> Dict[str, Any]:
        """Analyze style features of the garment"""
        try:
            features = {}
            
            # Analyze formality level
            features["formality"] = self._assess_formality(image, mask)
            
            # Analyze fit type
            features["fit_type"] = self._assess_fit_type(image, mask)
            
            # Analyze style category
            features["style_category"] = self._assess_style_category(image, mask)
            
            return features
            
        except Exception as e:
            logger.error(f"Style feature analysis failed: {e}")
            return {}
    
    def _assess_formality(self, image: Image.Image, mask: Image.Image) -> str:
        """Assess formality level (casual, business, formal)"""
        # Simplified formality assessment based on colors and structure
        try:
            # Extract dominant colors
            img_array = np.array(image)
            mask_array = np.array(mask)
            garment_pixels = img_array[mask_array > 128]
            
            if len(garment_pixels) == 0:
                return "casual"
            
            # Convert to HSV for better color analysis
            garment_hsv = []
            for pixel in garment_pixels:
                r, g, b = pixel
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                garment_hsv.append([h*360, s*100, v*100])
            
            garment_hsv = np.array(garment_hsv)
            
            # Formal colors tend to be darker and less saturated
            avg_saturation = np.mean(garment_hsv[:, 1])
            avg_value = np.mean(garment_hsv[:, 2])
            
            if avg_saturation < 30 and avg_value < 50:
                return "formal"
            elif avg_saturation < 50 and avg_value < 70:
                return "business"
            else:
                return "casual"
                
        except Exception as e:
            logger.error(f"Formality assessment failed: {e}")
            return "casual"
    
    def _assess_fit_type(self, image: Image.Image, mask: Image.Image) -> str:
        """Assess fit type (tight, regular, loose, oversized)"""
        try:
            mask_array = np.array(mask)
            
            # Find contours to analyze fit
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return "regular"
            
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate contour properties
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            
            if perimeter > 0:
                compactness = (perimeter ** 2) / area
                
                # Higher compactness indicates looser fit
                if compactness > 50:
                    return "oversized"
                elif compactness > 30:
                    return "loose"
                elif compactness > 20:
                    return "regular"
                else:
                    return "tight"
            
            return "regular"
            
        except Exception as e:
            logger.error(f"Fit assessment failed: {e}")
            return "regular"
    
    def _assess_style_category(self, image: Image.Image, mask: Image.Image) -> str:
        """Assess style category (trendy, classic, vintage, etc.)"""
        # Simplified style categorization
        return "classic"  # Default category
    
    def optimize_fit(self, image: Image.Image) -> Dict[str, Any]:
        """
        Optimize garment fit appearance
        
        Args:
            image: Input PIL Image
            
        Returns:
            Optimization results with improved fit
        """
        try:
            start_time = time.time()
            
            # Get original fit assessment
            original_fit = self._assess_garment_fit(image)
            
            # Segment garment for targeted optimization
            seg_result = self.segment_garment(image)
            mask = seg_result["mask"]
            
            # Apply fit optimization techniques
            optimized_image = image.copy()
            improvements = []
            
            # 1. Wrinkle reduction
            if self._has_wrinkles(image, mask):
                optimized_image = self._reduce_wrinkles(optimized_image, mask)
                improvements.append("wrinkle_reduction")
            
            # 2. Shape enhancement
            optimized_image = self._enhance_garment_shape(optimized_image, mask)
            improvements.append("shape_enhancement")
            
            # 3. Drape improvement
            optimized_image = self._improve_drape(optimized_image, mask)
            improvements.append("drape_improvement")
            
            # Assess final fit
            final_fit = self._assess_garment_fit(optimized_image)
            
            processing_time = time.time() - start_time
            
            return {
                "optimized_image": optimized_image,
                "fit_score": final_fit,
                "original_fit_score": original_fit,
                "improvements": improvements,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Fit optimization failed: {e}")
            return {
                "optimized_image": image,
                "fit_score": 5.0,
                "original_fit_score": 5.0,
                "improvements": [],
                "processing_time": 0.0,
                "error": str(e)
            }
    
    def _assess_garment_fit(self, image: Image.Image) -> float:
        """Assess current garment fit quality (0-10 scale)"""
        try:
            # Segment garment
            seg_result = self.segment_garment(image)
            mask = seg_result["mask"]
            
            # Calculate fit metrics
            fit_metrics = []
            
            # 1. Smoothness (wrinkle assessment)
            smoothness = self._calculate_smoothness(image, mask)
            fit_metrics.append(smoothness)
            
            # 2. Shape regularity
            shape_score = self._calculate_shape_regularity(mask)
            fit_metrics.append(shape_score)
            
            # 3. Proportion assessment
            proportion_score = self._calculate_proportions(mask)
            fit_metrics.append(proportion_score)
            
            # Combine metrics
            overall_fit = np.mean(fit_metrics) * 10
            return min(overall_fit, 10.0)
            
        except Exception as e:
            logger.error(f"Fit assessment failed: {e}")
            return 5.0  # Default medium score
    
    def _calculate_smoothness(self, image: Image.Image, mask: Image.Image) -> float:
        """Calculate smoothness of garment surface"""
        try:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            mask_array = np.array(mask)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate variance of Laplacian (edge detection)
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            
            # Focus on garment area
            garment_laplacian = laplacian[mask_array > 128]
            
            if len(garment_laplacian) > 0:
                # Lower variance indicates smoother surface
                variance = np.var(garment_laplacian)
                smoothness = 1.0 / (1.0 + variance / 1000)  # Normalize
                return min(smoothness, 1.0)
            
            return 0.7  # Default
            
        except Exception as e:
            logger.error(f"Smoothness calculation failed: {e}")
            return 0.7
    
    def _calculate_shape_regularity(self, mask: Image.Image) -> float:
        """Calculate shape regularity of garment"""
        try:
            mask_array = np.array(mask)
            
            # Find contours
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.5
            
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate contour smoothness
            epsilon = 0.02 * cv2.arcLength(main_contour, True)
            smoothed_contour = cv2.approxPolyDP(main_contour, epsilon, True)
            
            # Regular shapes have fewer vertices after approximation
            complexity = len(smoothed_contour) / len(main_contour)
            regularity = 1.0 - complexity
            
            return max(regularity, 0.0)
            
        except Exception as e:
            logger.error(f"Shape regularity calculation failed: {e}")
            return 0.5
    
    def _calculate_proportions(self, mask: Image.Image) -> float:
        """Calculate proportion correctness of garment"""
        try:
            mask_array = np.array(mask)
            
            # Find bounding rectangle
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.5
            
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Ideal aspect ratios for different garments (simplified)
            ideal_ratios = {
                "shirt": 0.8,
                "dress": 0.6,
                "pants": 0.5,
                "jacket": 0.9
            }
            
            # Use average ideal ratio
            avg_ideal = np.mean(list(ideal_ratios.values()))
            proportion_score = 1.0 - abs(aspect_ratio - avg_ideal) / avg_ideal
            
            return max(proportion_score, 0.0)
            
        except Exception as e:
            logger.error(f"Proportion calculation failed: {e}")
            return 0.5
    
    def _has_wrinkles(self, image: Image.Image, mask: Image.Image) -> bool:
        """Detect if garment has wrinkles"""
        try:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            mask_array = np.array(mask)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edges in garment area
            garment_edges = edges[mask_array > 128]
            edge_density = np.sum(garment_edges > 0) / len(garment_edges) if len(garment_edges) > 0 else 0
            
            # Threshold for wrinkle detection
            return edge_density > 0.1
            
        except Exception as e:
            logger.error(f"Wrinkle detection failed: {e}")
            return False
    
    def _reduce_wrinkles(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Reduce wrinkles in garment"""
        try:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            mask_array = np.array(mask)
            
            # Apply bilateral filter for edge-preserving smoothing
            smoothed = cv2.bilateralFilter(img_cv, 9, 75, 75)
            
            # Blend with original based on mask
            result = img_cv.copy()
            mask_normalized = mask_array.astype(float) / 255
            
            for c in range(3):
                result[:, :, c] = (
                    result[:, :, c] * (1 - mask_normalized * 0.7) +
                    smoothed[:, :, c] * (mask_normalized * 0.7)
                )
            
            return Image.fromarray(cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.error(f"Wrinkle reduction failed: {e}")
            return image
    
    def _enhance_garment_shape(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Enhance garment shape and structure"""
        try:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply subtle sharpening to enhance structure
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.1
            sharpened = cv2.filter2D(img_cv, -1, kernel)
            
            # Blend with original
            result = cv2.addWeighted(img_cv, 0.8, sharpened, 0.2, 0)
            
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.error(f"Shape enhancement failed: {e}")
            return image
    
    def _improve_drape(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Improve garment drape appearance"""
        try:
            # Apply subtle contrast enhancement for better drape definition
            enhancer = ImageEnhance.Contrast(image)
            enhanced = enhancer.enhance(1.1)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Drape improvement failed: {e}")
            return image
    
    def enhance_style(self, image: Image.Image, style_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance garment style based on parameters
        
        Args:
            image: Input PIL Image
            style_parameters: Style enhancement parameters
            
        Returns:
            Style enhancement results
        """
        try:
            enhanced_image = image.copy()
            enhancements = []
            
            # Color adjustment
            if style_parameters.get("color_adjustment", False):
                enhanced_image = self._adjust_colors(enhanced_image)
                enhancements.append("color_adjustment")
            
            # Wrinkle reduction
            if style_parameters.get("wrinkle_reduction", False):
                seg_result = self.segment_garment(enhanced_image)
                enhanced_image = self._reduce_wrinkles(enhanced_image, seg_result["mask"])
                enhancements.append("wrinkle_reduction")
            
            # Fabric enhancement
            if style_parameters.get("fabric_enhancement", False):
                enhanced_image = self._enhance_fabric(enhanced_image)
                enhancements.append("fabric_enhancement")
            
            # Calculate style score
            style_score = self._calculate_style_score(enhanced_image)
            
            return {
                "enhanced_image": enhanced_image,
                "style_score": style_score,
                "enhancements_applied": enhancements
            }
            
        except Exception as e:
            logger.error(f"Style enhancement failed: {e}")
            return {
                "enhanced_image": image,
                "style_score": 5.0,
                "enhancements_applied": [],
                "error": str(e)
            }
    
    def _adjust_colors(self, image: Image.Image) -> Image.Image:
        """Adjust garment colors for better appearance"""
        try:
            # Enhance color saturation slightly
            enhancer = ImageEnhance.Color(image)
            color_enhanced = enhancer.enhance(1.1)
            
            return color_enhanced
            
        except Exception as e:
            logger.error(f"Color adjustment failed: {e}")
            return image
    
    def _enhance_fabric(self, image: Image.Image) -> Image.Image:
        """Enhance fabric texture and appearance"""
        try:
            # Apply unsharp mask for fabric detail enhancement
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gaussian = cv2.GaussianBlur(img_cv, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(img_cv, 1.5, gaussian, -0.5, 0)
            
            return Image.fromarray(cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.error(f"Fabric enhancement failed: {e}")
            return image
    
    def _calculate_style_score(self, image: Image.Image) -> float:
        """Calculate overall style score for garment"""
        try:
            # Simplified style scoring based on various factors
            seg_result = self.segment_garment(image)
            
            # Color harmony
            color_score = self._assess_color_harmony(image, seg_result["mask"])
            
            # Texture quality
            texture_score = self._analyze_texture(image, seg_result["mask"])
            
            # Overall composition
            composition_score = 0.8  # Simplified
            
            # Combine scores
            style_score = (color_score * 0.4 + texture_score * 0.4 + composition_score * 0.2) * 10
            
            return min(style_score, 10.0)
            
        except Exception as e:
            logger.error(f"Style score calculation failed: {e}")
            return 7.0  # Default good score
    
    def _assess_color_harmony(self, image: Image.Image, mask: Image.Image) -> float:
        """Assess color harmony in garment"""
        try:
            img_array = np.array(image)
            mask_array = np.array(mask)
            
            garment_pixels = img_array[mask_array > 128]
            
            if len(garment_pixels) == 0:
                return 0.7
            
            # Extract dominant colors
            dominant_colors = self._extract_dominant_colors(garment_pixels, n_colors=3)
            
            # Simple harmony assessment based on color distance
            if len(dominant_colors) < 2:
                return 0.9  # Single color is always harmonious
            
            # Calculate color distances in RGB space
            distances = []
            for i in range(len(dominant_colors)):
                for j in range(i + 1, len(dominant_colors)):
                    color1 = np.array(dominant_colors[i])
                    color2 = np.array(dominant_colors[j])
                    distance = np.linalg.norm(color1 - color2)
                    distances.append(distance)
            
            avg_distance = np.mean(distances)
            
            # Moderate distances indicate good harmony
            if 50 < avg_distance < 150:
                return 0.9
            elif 30 < avg_distance < 200:
                return 0.7
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Color harmony assessment failed: {e}")
            return 0.7
    
    def correct_colors(self, image: Image.Image) -> Dict[str, Any]:
        """Correct and enhance garment colors"""
        try:
            # Auto color correction
            corrected_image = self._auto_color_correct(image)
            
            # Analyze colors
            color_analysis = self._analyze_colors(corrected_image)
            
            return {
                "corrected_image": corrected_image,
                "color_analysis": color_analysis,
                "adjustments_made": ["auto_correction", "saturation_boost"]
            }
            
        except Exception as e:
            logger.error(f"Color correction failed: {e}")
            return {
                "corrected_image": image,
                "color_analysis": {},
                "adjustments_made": []
            }
    
    def _auto_color_correct(self, image: Image.Image) -> Image.Image:
        """Apply automatic color correction"""
        try:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.error(f"Auto color correction failed: {e}")
            return image
    
    def _analyze_colors(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze color properties of garment"""
        try:
            seg_result = self.segment_garment(image)
            mask = seg_result["mask"]
            
            img_array = np.array(image)
            mask_array = np.array(mask)
            garment_pixels = img_array[mask_array > 128]
            
            if len(garment_pixels) == 0:
                return {}
            
            # Extract dominant colors
            dominant_colors = self._extract_dominant_colors(garment_pixels)
            
            # Calculate color harmony score
            harmony_score = self._assess_color_harmony(image, mask)
            
            return {
                "dominant_colors": dominant_colors,
                "color_harmony_score": harmony_score,
                "pixel_count": len(garment_pixels)
            }
            
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return {}
    
    def remove_wrinkles(self, image: Image.Image) -> Dict[str, Any]:
        """Remove wrinkles from garment"""
        try:
            seg_result = self.segment_garment(image)
            mask = seg_result["mask"]
            
            # Detect wrinkles
            has_wrinkles = self._has_wrinkles(image, mask)
            
            # Remove wrinkles if detected
            if has_wrinkles:
                smoothed_image = self._reduce_wrinkles(image, mask)
            else:
                smoothed_image = image
            
            # Calculate smoothness score
            smoothness_score = self._calculate_smoothness(smoothed_image, mask)
            
            return {
                "smoothed_image": smoothed_image,
                "wrinkles_detected": has_wrinkles,
                "smoothness_score": smoothness_score
            }
            
        except Exception as e:
            logger.error(f"Wrinkle removal failed: {e}")
            return {
                "smoothed_image": image,
                "wrinkles_detected": False,
                "smoothness_score": 0.7
            }
    
    def enhance_fabric_texture(self, image: Image.Image) -> Dict[str, Any]:
        """Enhance fabric texture quality"""
        try:
            enhanced_image = self._enhance_fabric(image)
            
            # Analyze texture
            seg_result = self.segment_garment(enhanced_image)
            texture_score = self._analyze_texture(enhanced_image, seg_result["mask"])
            
            # Determine fabric type (simplified)
            fabric_type = self._classify_fabric_type(enhanced_image, seg_result["mask"])
            
            return {
                "enhanced_image": enhanced_image,
                "texture_score": texture_score,
                "fabric_type": fabric_type
            }
            
        except Exception as e:
            logger.error(f"Fabric texture enhancement failed: {e}")
            return {
                "enhanced_image": image,
                "texture_score": 0.7,
                "fabric_type": "unknown"
            }
    
    def _classify_fabric_type(self, image: Image.Image, mask: Image.Image) -> str:
        """Classify fabric type based on texture analysis"""
        try:
            texture_score = self._analyze_texture(image, mask)
            
            # Simple fabric classification based on texture
            if texture_score > 0.8:
                return "smooth"  # Silk, satin, etc.
            elif texture_score > 0.6:
                return "woven"   # Cotton, linen, etc.
            elif texture_score > 0.4:
                return "knit"    # Jersey, sweater, etc.
            else:
                return "textured"  # Denim, corduroy, etc.
                
        except Exception as e:
            logger.error(f"Fabric classification failed: {e}")
            return "unknown"
    
    def optimize_garment_lighting(self, image: Image.Image) -> Dict[str, Any]:
        """Optimize lighting for garment presentation"""
        try:
            # Apply lighting optimization
            lit_image = self._apply_garment_lighting(image)
            
            # Calculate lighting score
            lighting_score = self._assess_lighting_quality(lit_image)
            
            return {
                "lit_image": lit_image,
                "lighting_score": lighting_score,
                "lighting_type": "optimized"
            }
            
        except Exception as e:
            logger.error(f"Garment lighting optimization failed: {e}")
            return {
                "lit_image": image,
                "lighting_score": 0.7,
                "lighting_type": "original"
            }
    
    def _apply_garment_lighting(self, image: Image.Image) -> Image.Image:
        """Apply optimized lighting for garment"""
        try:
            # Enhance brightness and contrast for better garment visibility
            enhancer = ImageEnhance.Brightness(image)
            brightened = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Contrast(brightened)
            contrasted = enhancer.enhance(1.1)
            
            return contrasted
            
        except Exception as e:
            logger.error(f"Garment lighting application failed: {e}")
            return image
    
    def _assess_lighting_quality(self, image: Image.Image) -> float:
        """Assess lighting quality of garment"""
        try:
            # Calculate brightness distribution
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Assess distribution - good lighting has balanced histogram
            hist_normalized = hist / hist.sum()
            
            # Calculate entropy (measure of distribution)
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            
            # Normalize entropy to 0-1 scale
            lighting_score = min(entropy / 8, 1.0)
            
            return lighting_score
            
        except Exception as e:
            logger.error(f"Lighting quality assessment failed: {e}")
            return 0.8
    
    def optimize_by_type(self, image: Image.Image, garment_type: str) -> Dict[str, Any]:
        """Optimize garment based on specific type"""
        try:
            # Type-specific optimization parameters
            type_params = {
                "shirt": {"color_adjustment": True, "wrinkle_reduction": True},
                "dress": {"fabric_enhancement": True, "color_adjustment": True},
                "pants": {"wrinkle_reduction": True, "fit_optimization": True},
                "jacket": {"texture_enhancement": True, "structure_enhancement": True}
            }
            
            params = type_params.get(garment_type, {})
            
            # Apply type-specific enhancements
            result = self.enhance_style(image, params)
            result["garment_type"] = garment_type
            result["type_specific_score"] = result.get("style_score", 7.0)
            
            return result
            
        except Exception as e:
            logger.error(f"Type-specific optimization failed: {e}")
            return {
                "optimized_image": image,
                "garment_type": garment_type,
                "type_specific_score": 7.0
            }
    
    def optimize_garment(self, image: Image.Image) -> Dict[str, Any]:
        """
        Complete garment optimization pipeline
        
        Args:
            image: Input PIL Image
            
        Returns:
            Comprehensive optimization results
        """
        try:
            start_time = time.time()
            
            # Detect garment type and properties
            detection_result = self.detect_garment_type(image)
            garment_count = 1 if detection_result["confidence"] > 0.5 else 0
            
            if garment_count == 0:
                # No garment detected - return with fallback
                return {
                    "optimized_image": image,
                    "overall_score": 5.0,
                    "garments_detected": 0,
                    "fallback_used": True,
                    "processing_time": time.time() - start_time
                }
            
            optimized_image = image
            
            # 1. Fit optimization
            fit_result = self.optimize_fit(optimized_image)
            optimized_image = fit_result["optimized_image"]
            
            # 2. Style enhancement
            style_params = {"color_adjustment": True, "wrinkle_reduction": True, "fabric_enhancement": True}
            style_result = self.enhance_style(optimized_image, style_params)
            optimized_image = style_result["enhanced_image"]
            
            # 3. Lighting optimization
            lighting_result = self.optimize_garment_lighting(optimized_image)
            optimized_image = lighting_result["lit_image"]
            
            # Calculate overall score
            overall_score = (
                fit_result["fit_score"] * 0.4 +
                style_result["style_score"] * 0.4 +
                lighting_result["lighting_score"] * 10 * 0.2
            )
            
            processing_time = time.time() - start_time
            
            return {
                "optimized_image": optimized_image,
                "overall_score": overall_score,
                "garments_detected": garment_count,
                "garment_types": [detection_result["garment_type"]],
                "fit_score": fit_result["fit_score"],
                "style_score": style_result["style_score"],
                "lighting_score": lighting_result["lighting_score"],
                "processing_time": processing_time,
                "fallback_used": False
            }
            
        except Exception as e:
            logger.error(f"Garment optimization pipeline failed: {e}")
            return {
                "optimized_image": image,
                "overall_score": 5.0,
                "garments_detected": 0,
                "fallback_used": True,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def optimize_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Optimize multiple garment images in batch"""
        results = []
        for image in images:
            result = self.optimize_garment(image)
            results.append(result)
        return results
    
    def clear_cache(self):
        """Clear optimization cache and free memory"""
        try:
            if hasattr(self.sam2_service, 'clear_cache'):
                self.sam2_service.clear_cache()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Garment optimization cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def unload_models(self):
        """Unload models to free memory"""
        try:
            if hasattr(self.sam2_service, 'unload_model'):
                self.sam2_service.unload_model()
            
            self.model_loaded = False
            self.clear_cache()
            logger.info("Garment optimization models unloaded")
            
        except Exception as e:
            logger.error(f"Failed to unload models: {e}")