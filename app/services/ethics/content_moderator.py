"""
Content Moderator for Content Safety Service

This module provides content moderation capabilities including
inappropriate content detection and professional standard validation.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image
import cv2
from dataclasses import dataclass


@dataclass
class ModerationResult:
    """Content moderation result structure"""
    inappropriate_detected: bool
    confidence: float
    categories: List[str]
    severity: str
    recommendations: List[str]


class ContentModerator:
    """
    Content moderator for inappropriate content detection
    
    Provides comprehensive content analysis for safety violations,
    professional standard validation, and age-appropriate assessment.
    """
    
    def __init__(self):
        """Initialize content moderator"""
        self.logger = logging.getLogger(__name__)
        
        # Moderation rules
        self.moderation_rules = {
            'inappropriate_keywords': [
                'explicit', 'inappropriate', 'offensive', 'sexual',
                'violent', 'hate', 'discriminatory', 'harmful'
            ],
            'professional_keywords': [
                'professional', 'business', 'formal', 'appropriate',
                'elegant', 'sophisticated', 'corporate', 'workplace'
            ],
            'quality_indicators': [
                'high_quality', 'professional_lighting', 'clear_composition',
                'appropriate_framing', 'good_resolution'
            ]
        }
        
        # Violation detectors
        self.violation_detectors = {
            'brightness': self._detect_brightness_issues,
            'composition': self._detect_composition_issues,
            'content': self._detect_content_violations,
            'quality': self._detect_quality_issues
        }
        
        # Professional standards
        self.professional_standards = {
            'attire': ['business_appropriate', 'formal', 'professional'],
            'setting': ['office', 'studio', 'professional_background'],
            'pose': ['formal', 'appropriate', 'professional'],
            'quality': ['high_resolution', 'good_lighting', 'clear']
        }
        
        self.logger.info("Content Moderator initialized")
    
    def analyze_image_content(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image content for safety and appropriateness
        
        Args:
            image_array: Numpy array of image data
            
        Returns:
            Comprehensive image content analysis
        """
        try:
            # Perform various content analyses
            brightness_analysis = self._analyze_image_brightness(image_array)
            composition_analysis = self._analyze_image_composition(image_array)
            quality_analysis = self._analyze_image_quality(image_array)
            content_analysis = self._analyze_image_content_safety(image_array)
            
            # Calculate overall safety score
            safety_score = (
                brightness_analysis['score'] * 0.15 +
                composition_analysis['score'] * 0.25 +
                quality_analysis['score'] * 0.3 +
                content_analysis['score'] * 0.3
            )
            
            result = {
                'safety_score': safety_score,
                'brightness_analysis': brightness_analysis,
                'composition_analysis': composition_analysis,
                'quality_analysis': quality_analysis,
                'content_analysis': content_analysis,
                'overall_assessment': self._generate_overall_assessment(safety_score)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image content analysis failed: {str(e)}")
            return self._create_fallback_analysis_result()
    
    def _analyze_image_brightness(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image brightness for appropriateness"""
        try:
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Calculate brightness metrics
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Assess brightness appropriateness
            if 60 <= mean_brightness <= 180:
                brightness_score = 1.0
                brightness_assessment = 'appropriate'
            elif 40 <= mean_brightness <= 200:
                brightness_score = 0.8
                brightness_assessment = 'acceptable'
            else:
                brightness_score = 0.5
                brightness_assessment = 'poor'
            
            # Check for extreme contrast
            if brightness_std > 80:
                brightness_score -= 0.1
                contrast_assessment = 'high_contrast'
            elif brightness_std < 20:
                brightness_score -= 0.1
                contrast_assessment = 'low_contrast'
            else:
                contrast_assessment = 'normal_contrast'
            
            result = {
                'score': max(0.0, min(1.0, brightness_score)),
                'mean_brightness': float(mean_brightness),
                'brightness_std': float(brightness_std),
                'brightness_assessment': brightness_assessment,
                'contrast_assessment': contrast_assessment
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Brightness analysis failed: {str(e)}")
            return {'score': 0.7, 'error': str(e)}
    
    def _analyze_image_composition(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image composition for professional standards"""
        try:
            height, width = image_array.shape[:2]
            
            # Aspect ratio analysis
            aspect_ratio = width / height
            
            if 0.8 <= aspect_ratio <= 1.2:  # Square-ish
                aspect_score = 0.9
                aspect_assessment = 'square_format'
            elif 1.2 <= aspect_ratio <= 1.8:  # Landscape
                aspect_score = 1.0
                aspect_assessment = 'landscape_format'
            elif 0.5 <= aspect_ratio <= 0.8:  # Portrait
                aspect_score = 1.0
                aspect_assessment = 'portrait_format'
            else:
                aspect_score = 0.6
                aspect_assessment = 'unusual_aspect_ratio'
            
            # Resolution analysis
            total_pixels = height * width
            
            if total_pixels >= 1024 * 1024:  # 1MP+
                resolution_score = 1.0
                resolution_assessment = 'high_resolution'
            elif total_pixels >= 512 * 512:  # 0.25MP+
                resolution_score = 0.8
                resolution_assessment = 'medium_resolution'
            else:
                resolution_score = 0.5
                resolution_assessment = 'low_resolution'
            
            # Composition score
            composition_score = (aspect_score + resolution_score) / 2
            
            result = {
                'score': composition_score,
                'aspect_ratio': aspect_ratio,
                'aspect_assessment': aspect_assessment,
                'resolution': total_pixels,
                'resolution_assessment': resolution_assessment,
                'dimensions': {'width': width, 'height': height}
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Composition analysis failed: {str(e)}")
            return {'score': 0.8, 'error': str(e)}
    
    def _analyze_image_quality(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image quality metrics"""
        try:
            # Convert to grayscale for quality analysis
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Sharpness analysis using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var > 500:
                sharpness_score = 1.0
                sharpness_assessment = 'sharp'
            elif laplacian_var > 100:
                sharpness_score = 0.8
                sharpness_assessment = 'acceptable'
            else:
                sharpness_score = 0.4
                sharpness_assessment = 'blurry'
            
            # Noise analysis
            noise_estimate = np.std(gray)
            
            if noise_estimate < 15:
                noise_score = 1.0
                noise_assessment = 'low_noise'
            elif noise_estimate < 30:
                noise_score = 0.8
                noise_assessment = 'moderate_noise'
            else:
                noise_score = 0.5
                noise_assessment = 'high_noise'
            
            # Overall quality score
            quality_score = (sharpness_score + noise_score) / 2
            
            result = {
                'score': quality_score,
                'sharpness_score': sharpness_score,
                'sharpness_assessment': sharpness_assessment,
                'laplacian_variance': float(laplacian_var),
                'noise_score': noise_score,
                'noise_assessment': noise_assessment,
                'noise_estimate': float(noise_estimate)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quality analysis failed: {str(e)}")
            return {'score': 0.7, 'error': str(e)}
    
    def _analyze_image_content_safety(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image content for safety violations"""
        try:
            # Basic content safety analysis
            # In production, use advanced ML models for content detection
            
            # Color distribution analysis
            color_safety_score = self._analyze_color_distribution(image_array)
            
            # Edge detection for inappropriate content patterns
            edge_safety_score = self._analyze_edge_patterns(image_array)
            
            # Texture analysis for content appropriateness
            texture_safety_score = self._analyze_texture_patterns(image_array)
            
            # Overall content safety score
            content_safety_score = (
                color_safety_score * 0.4 +
                edge_safety_score * 0.3 +
                texture_safety_score * 0.3
            )
            
            # Determine safety level
            if content_safety_score >= 0.9:
                safety_level = 'very_safe'
            elif content_safety_score >= 0.7:
                safety_level = 'safe'
            elif content_safety_score >= 0.5:
                safety_level = 'questionable'
            else:
                safety_level = 'unsafe'
            
            result = {
                'score': content_safety_score,
                'safety_level': safety_level,
                'color_safety': color_safety_score,
                'edge_safety': edge_safety_score,
                'texture_safety': texture_safety_score
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content safety analysis failed: {str(e)}")
            return {'score': 0.7, 'safety_level': 'questionable', 'error': str(e)}
    
    def _analyze_color_distribution(self, image_array: np.ndarray) -> float:
        """Analyze color distribution for content safety"""
        try:
            # Simple color distribution analysis
            # Professional images typically have balanced color distributions
            
            if len(image_array.shape) == 3:
                # RGB analysis
                rgb_means = np.mean(image_array, axis=(0, 1))
                rgb_stds = np.std(image_array, axis=(0, 1))
                
                # Check for balanced color distribution
                color_balance = 1.0 - (np.std(rgb_means) / 255.0)
                
                # Check for appropriate color variance
                avg_variance = np.mean(rgb_stds)
                variance_score = min(avg_variance / 50.0, 1.0)  # Normalize to 0-1
                
                color_score = (color_balance + variance_score) / 2
            else:
                # Grayscale analysis
                gray_std = np.std(image_array)
                color_score = min(gray_std / 50.0, 1.0)
            
            return max(0.0, min(1.0, color_score))
            
        except Exception as e:
            self.logger.error(f"Color distribution analysis failed: {str(e)}")
            return 0.8
    
    def _analyze_edge_patterns(self, image_array: np.ndarray) -> float:
        """Analyze edge patterns for content appropriateness"""
        try:
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Professional images typically have moderate edge density
            if 0.05 <= edge_density <= 0.25:
                edge_score = 1.0
            elif 0.02 <= edge_density <= 0.35:
                edge_score = 0.8
            else:
                edge_score = 0.6
            
            return edge_score
            
        except Exception as e:
            self.logger.error(f"Edge pattern analysis failed: {str(e)}")
            return 0.8
    
    def _analyze_texture_patterns(self, image_array: np.ndarray) -> float:
        """Analyze texture patterns for content safety"""
        try:
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Simple texture analysis using local standard deviation
            kernel = np.ones((5, 5), np.float32) / 25
            mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            texture_variance = np.var(gray.astype(np.float32) - mean_filtered)
            
            # Professional images typically have moderate texture complexity
            normalized_texture = min(texture_variance / 1000.0, 1.0)
            
            if 0.1 <= normalized_texture <= 0.8:
                texture_score = 1.0
            elif 0.05 <= normalized_texture <= 0.9:
                texture_score = 0.8
            else:
                texture_score = 0.6
            
            return texture_score
            
        except Exception as e:
            self.logger.error(f"Texture pattern analysis failed: {str(e)}")
            return 0.8
    
    def _generate_overall_assessment(self, safety_score: float) -> str:
        """Generate overall assessment based on safety score"""
        if safety_score >= 0.9:
            return 'excellent_professional_quality'
        elif safety_score >= 0.8:
            return 'good_professional_quality'
        elif safety_score >= 0.7:
            return 'acceptable_quality'
        elif safety_score >= 0.5:
            return 'needs_improvement'
        else:
            return 'inappropriate_or_poor_quality'
    
    def detect_inappropriate_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect inappropriate content in provided data
        
        Args:
            content_data: Content data to analyze for inappropriate content
            
        Returns:
            Inappropriate content detection results
        """
        try:
            # Extract content information
            image_content = content_data.get('image_content', '')
            style_description = content_data.get('style_description', '')
            target_audience = content_data.get('target_audience', 'general')
            
            # Analyze different aspects
            text_analysis = self._analyze_text_appropriateness(style_description)
            context_analysis = self._analyze_context_appropriateness(content_data)
            audience_analysis = self._analyze_audience_appropriateness(target_audience)
            
            # Determine if inappropriate content is detected
            inappropriate_detected = (
                text_analysis['inappropriate'] or
                context_analysis['inappropriate'] or
                audience_analysis['inappropriate']
            )
            
            # Collect categories
            categories = []
            categories.extend(text_analysis.get('categories', []))
            categories.extend(context_analysis.get('categories', []))
            categories.extend(audience_analysis.get('categories', []))
            
            # Calculate confidence
            confidence_scores = [
                text_analysis.get('confidence', 0.5),
                context_analysis.get('confidence', 0.5),
                audience_analysis.get('confidence', 0.5)
            ]
            overall_confidence = max(confidence_scores) if inappropriate_detected else min(confidence_scores)
            
            result = {
                'inappropriate_detected': inappropriate_detected,
                'categories': list(set(categories)),  # Remove duplicates
                'confidence': overall_confidence,
                'component_analyses': {
                    'text_analysis': text_analysis,
                    'context_analysis': context_analysis,
                    'audience_analysis': audience_analysis
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inappropriate content detection failed: {str(e)}")
            return self._create_fallback_inappropriate_result()
    
    def _analyze_text_appropriateness(self, text: str) -> Dict[str, Any]:
        """Analyze text content for appropriateness"""
        try:
            text_lower = text.lower()
            
            # Check for inappropriate keywords
            inappropriate_keywords = self.moderation_rules['inappropriate_keywords']
            inappropriate_found = [keyword for keyword in inappropriate_keywords if keyword in text_lower]
            
            # Check for professional keywords
            professional_keywords = self.moderation_rules['professional_keywords']
            professional_found = [keyword for keyword in professional_keywords if keyword in text_lower]
            
            # Determine appropriateness
            inappropriate = len(inappropriate_found) > 0
            confidence = 0.9 if inappropriate_found else 0.8
            
            # Categorize violations
            categories = []
            if 'explicit' in inappropriate_found or 'sexual' in inappropriate_found:
                categories.append('explicit_content')
            if 'violent' in inappropriate_found:
                categories.append('violent_content')
            if 'hate' in inappropriate_found or 'discriminatory' in inappropriate_found:
                categories.append('hate_speech')
            if 'offensive' in inappropriate_found:
                categories.append('offensive_content')
            
            result = {
                'inappropriate': inappropriate,
                'confidence': confidence,
                'categories': categories,
                'inappropriate_keywords': inappropriate_found,
                'professional_indicators': professional_found
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text appropriateness analysis failed: {str(e)}")
            return {'inappropriate': False, 'confidence': 0.5, 'categories': [], 'error': str(e)}
    
    def _analyze_context_appropriateness(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context appropriateness"""
        try:
            context_elements = [
                content_data.get('image_content', ''),
                content_data.get('brand_context', ''),
                content_data.get('usage_context', '')
            ]
            
            context_text = ' '.join(context_elements).lower()
            
            # Check for inappropriate contexts
            inappropriate_contexts = [
                'inappropriate', 'explicit', 'offensive', 'adult_only',
                'nsfw', 'mature_content', 'restricted'
            ]
            
            professional_contexts = [
                'business', 'corporate', 'professional', 'workplace',
                'formal', 'commercial', 'brand'
            ]
            
            inappropriate_context_found = any(ctx in context_text for ctx in inappropriate_contexts)
            professional_context_found = any(ctx in context_text for ctx in professional_contexts)
            
            # Determine appropriateness
            inappropriate = inappropriate_context_found
            confidence = 0.8 if inappropriate_context_found or professional_context_found else 0.6
            
            categories = []
            if inappropriate_context_found:
                categories.append('inappropriate_context')
            
            result = {
                'inappropriate': inappropriate,
                'confidence': confidence,
                'categories': categories,
                'professional_context': professional_context_found
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Context appropriateness analysis failed: {str(e)}")
            return {'inappropriate': False, 'confidence': 0.5, 'categories': [], 'error': str(e)}
    
    def _analyze_audience_appropriateness(self, target_audience: str) -> Dict[str, Any]:
        """Analyze target audience appropriateness"""
        try:
            audience_lower = target_audience.lower()
            
            # Check for age-restricted content markers
            adult_only_indicators = ['adult_only', 'mature', '18+', 'restricted']
            family_indicators = ['family', 'all_ages', 'children', 'general']
            
            adult_only = any(indicator in audience_lower for indicator in adult_only_indicators)
            family_friendly = any(indicator in audience_lower for indicator in family_indicators)
            
            # For family audiences, any adult content is inappropriate
            inappropriate = False
            categories = []
            
            if adult_only and family_friendly:
                inappropriate = True
                categories.append('conflicting_audience_requirements')
            
            confidence = 0.9 if adult_only or family_friendly else 0.6
            
            result = {
                'inappropriate': inappropriate,
                'confidence': confidence,
                'categories': categories,
                'adult_only': adult_only,
                'family_friendly': family_friendly
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Audience appropriateness analysis failed: {str(e)}")
            return {'inappropriate': False, 'confidence': 0.5, 'categories': [], 'error': str(e)}
    
    def identify_content_violations(self, image: Image.Image) -> List[str]:
        """
        Identify specific content violations in image
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            List of identified content violations
        """
        try:
            violations = []
            image_array = np.array(image)
            
            # Run violation detectors
            for detector_name, detector_func in self.violation_detectors.items():
                try:
                    detector_violations = detector_func(image_array)
                    violations.extend(detector_violations)
                except Exception as e:
                    self.logger.error(f"Violation detector {detector_name} failed: {str(e)}")
            
            return list(set(violations))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Content violation identification failed: {str(e)}")
            return ['analysis_error']
    
    def _detect_brightness_issues(self, image_array: np.ndarray) -> List[str]:
        """Detect brightness-related violations"""
        violations = []
        
        try:
            # Convert to grayscale if needed
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 30:
                violations.append('extremely_dark_image')
            elif mean_brightness > 220:
                violations.append('extremely_bright_image')
            
            # Check for extreme contrast
            brightness_std = np.std(gray)
            if brightness_std > 100:
                violations.append('extreme_contrast')
            elif brightness_std < 10:
                violations.append('flat_image')
                
        except Exception as e:
            self.logger.error(f"Brightness issue detection failed: {str(e)}")
        
        return violations
    
    def _detect_composition_issues(self, image_array: np.ndarray) -> List[str]:
        """Detect composition-related violations"""
        violations = []
        
        try:
            height, width = image_array.shape[:2]
            aspect_ratio = width / height
            
            # Check for problematic aspect ratios
            if aspect_ratio > 3.0:
                violations.append('extremely_wide_image')
            elif aspect_ratio < 0.3:
                violations.append('extremely_tall_image')
            
            # Check for very low resolution
            total_pixels = height * width
            if total_pixels < 128 * 128:
                violations.append('very_low_resolution')
                
        except Exception as e:
            self.logger.error(f"Composition issue detection failed: {str(e)}")
        
        return violations
    
    def _detect_content_violations(self, image_array: np.ndarray) -> List[str]:
        """Detect content-specific violations"""
        violations = []
        
        try:
            # Basic content violation detection
            # In production, use advanced ML models
            
            # Check for unusual color distributions that might indicate inappropriate content
            if len(image_array.shape) == 3:
                rgb_means = np.mean(image_array, axis=(0, 1))
                
                # Very red-dominant images might be concerning
                if rgb_means[0] > rgb_means[1] * 1.5 and rgb_means[0] > rgb_means[2] * 1.5:
                    if rgb_means[0] > 150:  # High red values
                        violations.append('unusual_color_distribution')
                
        except Exception as e:
            self.logger.error(f"Content violation detection failed: {str(e)}")
        
        return violations
    
    def _detect_quality_issues(self, image_array: np.ndarray) -> List[str]:
        """Detect quality-related violations"""
        violations = []
        
        try:
            # Convert to grayscale for quality analysis
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
            
            # Detect blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50:
                violations.append('blurry_image')
            
            # Detect excessive noise
            noise_estimate = np.std(gray)
            if noise_estimate > 50:
                violations.append('noisy_image')
                
        except Exception as e:
            self.logger.error(f"Quality issue detection failed: {str(e)}")
        
        return violations
    
    def assess_content_clarity(self, image: Image.Image) -> float:
        """
        Assess content clarity for confidence calculation
        
        Args:
            image: PIL Image to assess
            
        Returns:
            Content clarity score (0-1)
        """
        try:
            image_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
            
            # Assess sharpness
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)
            
            # Assess contrast
            contrast = np.std(gray)
            contrast_score = min(contrast / 50.0, 1.0)
            
            # Combine scores
            clarity_score = (sharpness_score + contrast_score) / 2
            
            return max(0.0, min(1.0, clarity_score))
            
        except Exception as e:
            self.logger.error(f"Content clarity assessment failed: {str(e)}")
            return 0.7
    
    def analyze_age_appropriateness(self, image: Image.Image, target_age_group: str, 
                                  content_type: str) -> Dict[str, Any]:
        """
        Analyze content for age appropriateness
        
        Args:
            image: PIL Image to analyze
            target_age_group: Target age group
            content_type: Type of content
            
        Returns:
            Age appropriateness analysis
        """
        try:
            image_array = np.array(image)
            
            # Basic age appropriateness analysis
            content_analysis = self.analyze_image_content(image_array)
            safety_score = content_analysis.get('safety_score', 0.7)
            
            # Age-specific thresholds
            age_thresholds = {
                'all_ages': 0.95,
                'family': 0.90,
                'teen': 0.80,
                'adult': 0.70
            }
            
            threshold = age_thresholds.get(target_age_group.lower(), 0.80)
            
            # Calculate age-specific scores
            mature_content_score = max(0.0, 1.0 - safety_score)
            violence_score = 0.0  # Placeholder - implement with advanced models
            suggestive_score = max(0.0, (1.0 - safety_score) * 0.5)  # Estimate
            
            # Check for inappropriate language (placeholder)
            inappropriate_language = False  # Implement with text analysis
            
            result = {
                'mature_content_score': mature_content_score,
                'violence_score': violence_score,
                'suggestive_score': suggestive_score,
                'inappropriate_language': inappropriate_language,
                'overall_score': safety_score,
                'meets_age_threshold': safety_score >= threshold
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Age appropriateness analysis failed: {str(e)}")
            return self._create_fallback_age_analysis()
    
    def analyze_family_appropriateness(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze content for family-friendly appropriateness"""
        try:
            # Family-friendly content requires highest standards
            content_analysis = self.analyze_image_content(np.array(image))
            safety_score = content_analysis.get('safety_score', 0.7)
            
            # Family-friendly threshold is very high
            family_safe_score = safety_score if safety_score >= 0.9 else safety_score * 0.8
            
            result = {
                'family_safe_score': family_safe_score,
                'family_appropriate': family_safe_score >= 0.9,
                'content_analysis': content_analysis
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Family appropriateness analysis failed: {str(e)}")
            return {'family_safe_score': 0.7, 'family_appropriate': False, 'error': str(e)}
    
    def analyze_professional_appropriateness(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze content for professional appropriateness"""
        try:
            content_analysis = self.analyze_image_content(np.array(image))
            safety_score = content_analysis.get('safety_score', 0.7)
            quality_score = content_analysis.get('quality_analysis', {}).get('score', 0.7)
            
            # Professional score considers both safety and quality
            professional_score = (safety_score * 0.6 + quality_score * 0.4)
            
            result = {
                'professional_score': professional_score,
                'professional_appropriate': professional_score >= 0.85,
                'content_analysis': content_analysis
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Professional appropriateness analysis failed: {str(e)}")
            return {'professional_score': 0.7, 'professional_appropriate': False, 'error': str(e)}
    
    def analyze_adult_appropriateness(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze content for adult audience appropriateness"""
        try:
            content_analysis = self.analyze_image_content(np.array(image))
            safety_score = content_analysis.get('safety_score', 0.7)
            
            # Adult content has more permissive thresholds but still needs to be appropriate
            adult_appropriate_score = safety_score
            
            result = {
                'adult_appropriate_score': adult_appropriate_score,
                'adult_appropriate': adult_appropriate_score >= 0.7,
                'content_analysis': content_analysis
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adult appropriateness analysis failed: {str(e)}")
            return {'adult_appropriate_score': 0.7, 'adult_appropriate': False, 'error': str(e)}
    
    def analyze_general_appropriateness(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze content for general audience appropriateness"""
        try:
            content_analysis = self.analyze_image_content(np.array(image))
            safety_score = content_analysis.get('safety_score', 0.7)
            
            # General audience requires balanced standards
            general_appropriate_score = safety_score
            
            result = {
                'general_appropriate_score': general_appropriate_score,
                'general_appropriate': general_appropriate_score >= 0.8,
                'content_analysis': content_analysis
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"General appropriateness analysis failed: {str(e)}")
            return {'general_appropriate_score': 0.7, 'general_appropriate': False, 'error': str(e)}
    
    def analyze_image_safety(self, image: Image.Image) -> Dict[str, Any]:
        """
        Comprehensive image safety analysis
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Comprehensive image safety analysis
        """
        try:
            # Perform comprehensive analysis
            content_analysis = self.analyze_image_content(np.array(image))
            
            # Extract key metrics
            safety_score = content_analysis.get('safety_score', 0.7)
            overall_assessment = content_analysis.get('overall_assessment', 'needs_review')
            
            # Additional safety checks
            violations = self.identify_content_violations(image)
            
            result = {
                'safety_score': safety_score,
                'overall_assessment': overall_assessment,
                'violations': violations,
                'safe_for_use': safety_score >= 0.8 and len(violations) == 0,
                'detailed_analysis': content_analysis
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image safety analysis failed: {str(e)}")
            return {'safety_score': 0.5, 'safe_for_use': False, 'error': str(e)}
    
    # Fallback result creation methods
    def _create_fallback_analysis_result(self) -> Dict[str, Any]:
        """Create fallback analysis result"""
        return {
            'safety_score': 0.7,
            'brightness_analysis': {'score': 0.7, 'error': True},
            'composition_analysis': {'score': 0.8, 'error': True},
            'quality_analysis': {'score': 0.7, 'error': True},
            'content_analysis': {'score': 0.7, 'safety_level': 'questionable', 'error': True},
            'overall_assessment': 'needs_review',
            'error': True
        }
    
    def _create_fallback_inappropriate_result(self) -> Dict[str, Any]:
        """Create fallback inappropriate content result"""
        return {
            'inappropriate_detected': True,
            'categories': ['analysis_error'],
            'confidence': 0.0,
            'component_analyses': {
                'text_analysis': {'inappropriate': False, 'confidence': 0.0, 'error': True},
                'context_analysis': {'inappropriate': False, 'confidence': 0.0, 'error': True},
                'audience_analysis': {'inappropriate': False, 'confidence': 0.0, 'error': True}
            },
            'error': True
        }
    
    def _create_fallback_age_analysis(self) -> Dict[str, Any]:
        """Create fallback age analysis result"""
        return {
            'mature_content_score': 0.5,
            'violence_score': 0.0,
            'suggestive_score': 0.0,
            'inappropriate_language': False,
            'overall_score': 0.5,
            'meets_age_threshold': False,
            'error': True
        }