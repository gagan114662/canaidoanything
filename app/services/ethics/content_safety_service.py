"""
Content Safety Service for Garment Creative AI

This service provides comprehensive content safety validation,
inappropriate content detection, and professional standard enforcement.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import re

from app.services.ethics.content_moderator import ContentModerator
from app.services.ethics.ethics_engine import EthicsEngine


@dataclass
class ContentSafetyResult:
    """Result structure for content safety analysis"""
    is_safe: bool
    safety_score: float
    violation_categories: List[str]
    appropriateness_score: float
    professional_standard_met: bool
    age_appropriate: bool
    recommendations: List[str]
    blocking_required: bool
    human_review_required: bool
    timestamp: datetime


@dataclass
class SafetyThresholds:
    """Configurable safety thresholds"""
    appropriateness_min: float = 0.95
    professional_standard_min: float = 0.90
    age_appropriate_min: float = 0.95
    ethics_compliance_min: float = 0.90
    overall_safety_min: float = 0.85


class ContentSafetyService:
    """
    Comprehensive content safety and moderation service
    
    Provides content appropriateness validation, professional standard
    enforcement, age-appropriate content verification, and ethics compliance.
    """
    
    def __init__(self):
        """Initialize content safety service"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.content_moderator = ContentModerator()
        self.ethics_engine = EthicsEngine()
        
        # Safety thresholds
        self.safety_thresholds = SafetyThresholds()
        
        # Content categories
        self.inappropriate_categories = [
            'sexually_explicit', 'violence', 'hate_speech', 'discrimination',
            'harassment', 'illegal_content', 'harmful_stereotypes', 'exploitation'
        ]
        
        self.professional_standards = [
            'appropriate_attire', 'professional_setting', 'respectful_poses',
            'business_appropriate', 'brand_safe', 'workplace_suitable'
        ]
        
        # Human review queue
        self.human_review_queue = []
        
        # Safety history for analytics
        self.safety_history = []
        
        self.logger.info("Content Safety Service initialized")
    
    def validate_content_appropriateness(self, image: Image.Image, 
                                       metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate overall content appropriateness
        
        Args:
            image: PIL Image to validate
            metadata: Content metadata (style, context, target audience, etc.)
            
        Returns:
            Comprehensive appropriateness validation results
        """
        try:
            # Extract metadata
            style = metadata.get('style', 'unknown')
            context = metadata.get('context', 'unknown')
            target_audience = metadata.get('target_audience', 'general')
            commercial_use = metadata.get('commercial_use', False)
            
            # Perform multi-dimensional appropriateness check
            image_appropriateness = self._assess_image_appropriateness(image)
            context_appropriateness = self._assess_context_appropriateness(context, style)
            audience_appropriateness = self._assess_audience_appropriateness(target_audience, image)
            
            # Calculate overall appropriateness score
            appropriateness_score = (
                image_appropriateness * 0.5 +
                context_appropriateness * 0.3 +
                audience_appropriateness * 0.2
            )
            
            # Determine if content is appropriate
            is_appropriate = appropriateness_score >= self.safety_thresholds.appropriateness_min
            
            # Identify violation categories
            violation_categories = self._identify_appropriateness_violations(
                image, metadata, appropriateness_score
            )
            
            # Generate recommendations
            recommendations = self._generate_appropriateness_recommendations(
                appropriateness_score, violation_categories, metadata
            )
            
            # Calculate confidence
            confidence = self._calculate_appropriateness_confidence(image, metadata)
            
            result = {
                'is_appropriate': is_appropriate,
                'appropriateness_score': appropriateness_score,
                'violation_categories': violation_categories,
                'recommendations': recommendations,
                'confidence': confidence,
                'component_scores': {
                    'image_appropriateness': image_appropriateness,
                    'context_appropriateness': context_appropriateness,
                    'audience_appropriateness': audience_appropriateness
                }
            }
            
            self.logger.info(f"Content appropriateness validation: {'appropriate' if is_appropriate else 'inappropriate'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Content appropriateness validation failed: {str(e)}")
            return self._create_fallback_appropriateness_result()
    
    def _assess_image_appropriateness(self, image: Image.Image) -> float:
        """Assess image appropriateness using visual analysis"""
        try:
            # Convert image for analysis
            image_array = np.array(image)
            
            # Analyze image properties
            brightness_score = self._assess_image_brightness(image_array)
            composition_score = self._assess_image_composition(image_array)
            content_score = self._assess_image_content_safety(image_array)
            
            # Combine scores
            image_appropriateness = (
                brightness_score * 0.2 +
                composition_score * 0.3 +
                content_score * 0.5
            )
            
            return min(max(image_appropriateness, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Image appropriateness assessment failed: {str(e)}")
            return 0.7  # Conservative fallback
    
    def _assess_image_brightness(self, image_array: np.ndarray) -> float:
        """Assess image brightness appropriateness"""
        try:
            # Calculate average brightness
            gray = np.mean(image_array, axis=2) if len(image_array.shape) == 3 else image_array
            avg_brightness = np.mean(gray)
            
            # Optimal brightness range (not too dark, not too bright)
            if 50 <= avg_brightness <= 200:
                return 1.0
            elif 30 <= avg_brightness <= 220:
                return 0.8
            else:
                return 0.6
                
        except Exception as e:
            self.logger.error(f"Brightness assessment failed: {str(e)}")
            return 0.7
    
    def _assess_image_composition(self, image_array: np.ndarray) -> float:
        """Assess image composition appropriateness"""
        try:
            # Basic composition analysis
            height, width = image_array.shape[:2]
            
            # Check image dimensions (reasonable aspect ratio)
            aspect_ratio = width / height
            if 0.5 <= aspect_ratio <= 2.0:
                composition_score = 0.9
            else:
                composition_score = 0.7
            
            # Check for sufficient image size
            if height >= 256 and width >= 256:
                composition_score += 0.1
            
            return min(composition_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Composition assessment failed: {str(e)}")
            return 0.8
    
    def _assess_image_content_safety(self, image_array: np.ndarray) -> float:
        """Assess image content for safety violations"""
        try:
            # Use content moderator for detailed analysis
            content_analysis = self.content_moderator.analyze_image_content(image_array)
            
            # Calculate safety score based on content analysis
            safety_score = content_analysis.get('safety_score', 0.8)
            
            return safety_score
            
        except Exception as e:
            self.logger.error(f"Content safety assessment failed: {str(e)}")
            return 0.7
    
    def _assess_context_appropriateness(self, context: str, style: str) -> float:
        """Assess context appropriateness"""
        try:
            context_lower = context.lower()
            style_lower = style.lower()
            
            # Professional contexts
            professional_contexts = [
                'business', 'corporate', 'professional', 'formal',
                'office', 'workplace', 'conference', 'meeting'
            ]
            
            if any(ctx in context_lower for ctx in professional_contexts):
                return 0.95
            
            # Fashion/creative contexts
            fashion_contexts = [
                'fashion', 'editorial', 'artistic', 'creative',
                'photography', 'portfolio', 'commercial'
            ]
            
            if any(ctx in context_lower for ctx in fashion_contexts):
                return 0.9
            
            # Casual/lifestyle contexts
            casual_contexts = ['casual', 'lifestyle', 'everyday', 'social']
            
            if any(ctx in context_lower for ctx in casual_contexts):
                return 0.85
            
            # Default neutral score
            return 0.8
            
        except Exception as e:
            self.logger.error(f"Context appropriateness assessment failed: {str(e)}")
            return 0.7
    
    def _assess_audience_appropriateness(self, target_audience: str, image: Image.Image) -> float:
        """Assess appropriateness for target audience"""
        try:
            audience_lower = target_audience.lower()
            
            # All ages content - strictest standards
            if 'all_ages' in audience_lower or 'family' in audience_lower:
                return self._assess_family_friendly_content(image)
            
            # Professional/business audience
            elif 'professional' in audience_lower or 'business' in audience_lower:
                return self._assess_professional_audience_content(image)
            
            # Adult/mature audience
            elif 'adult' in audience_lower or 'mature' in audience_lower:
                return self._assess_adult_audience_content(image)
            
            # General audience (default)
            else:
                return self._assess_general_audience_content(image)
                
        except Exception as e:
            self.logger.error(f"Audience appropriateness assessment failed: {str(e)}")
            return 0.8
    
    def _assess_family_friendly_content(self, image: Image.Image) -> float:
        """Assess content for family-friendly appropriateness"""
        try:
            # Strictest standards for family content
            content_analysis = self.content_moderator.analyze_family_appropriateness(image)
            return content_analysis.get('family_safe_score', 0.9)
            
        except Exception as e:
            self.logger.error(f"Family-friendly assessment failed: {str(e)}")
            return 0.8
    
    def _assess_professional_audience_content(self, image: Image.Image) -> float:
        """Assess content for professional audience"""
        try:
            professional_analysis = self.content_moderator.analyze_professional_appropriateness(image)
            return professional_analysis.get('professional_score', 0.85)
            
        except Exception as e:
            self.logger.error(f"Professional audience assessment failed: {str(e)}")
            return 0.8
    
    def _assess_adult_audience_content(self, image: Image.Image) -> float:
        """Assess content for adult audience"""
        try:
            # More permissive for adult content but still appropriate
            adult_analysis = self.content_moderator.analyze_adult_appropriateness(image)
            return adult_analysis.get('adult_appropriate_score', 0.8)
            
        except Exception as e:
            self.logger.error(f"Adult audience assessment failed: {str(e)}")
            return 0.7
    
    def _assess_general_audience_content(self, image: Image.Image) -> float:
        """Assess content for general audience"""
        try:
            # Balanced standards for general audience
            general_analysis = self.content_moderator.analyze_general_appropriateness(image)
            return general_analysis.get('general_appropriate_score', 0.85)
            
        except Exception as e:
            self.logger.error(f"General audience assessment failed: {str(e)}")
            return 0.8
    
    def _identify_appropriateness_violations(self, image: Image.Image, metadata: Dict[str, Any], 
                                          score: float) -> List[str]:
        """Identify specific appropriateness violations"""
        violations = []
        
        try:
            # Score-based violations
            if score < 0.5:
                violations.append('severe_appropriateness_violation')
            elif score < 0.7:
                violations.append('moderate_appropriateness_violation')
            
            # Content-specific violations
            content_violations = self.content_moderator.identify_content_violations(image)
            violations.extend(content_violations)
            
            # Context-specific violations
            context = metadata.get('context', '')
            if 'inappropriate' in context.lower():
                violations.append('inappropriate_context')
            
            # Audience-specific violations
            target_audience = metadata.get('target_audience', '')
            if 'all_ages' in target_audience.lower() and score < 0.9:
                violations.append('not_family_friendly')
            
        except Exception as e:
            self.logger.error(f"Violation identification failed: {str(e)}")
        
        return violations
    
    def _generate_appropriateness_recommendations(self, score: float, violations: List[str], 
                                                metadata: Dict[str, Any]) -> List[str]:
        """Generate appropriateness improvement recommendations"""
        recommendations = []
        
        try:
            # Score-based recommendations
            if score < 0.5:
                recommendations.extend([
                    "Content requires significant modification for appropriateness",
                    "Consider alternative styling or composition",
                    "Seek professional content review"
                ])
            elif score < 0.7:
                recommendations.extend([
                    "Content may need minor adjustments for full appropriateness",
                    "Review styling and context for improvements"
                ])
            
            # Violation-specific recommendations
            if 'inappropriate_context' in violations:
                recommendations.append("Adjust context or setting for better appropriateness")
            
            if 'not_family_friendly' in violations:
                recommendations.append("Modify content for family-friendly audience")
            
            # General recommendations
            recommendations.extend([
                "Ensure professional quality and composition",
                "Maintain respectful and inclusive representation",
                "Consider target audience expectations"
            ])
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
        
        return recommendations
    
    def _calculate_appropriateness_confidence(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """Calculate confidence in appropriateness assessment"""
        try:
            confidence_factors = []
            
            # Image quality factor
            image_array = np.array(image)
            if image_array.shape[0] >= 512 and image_array.shape[1] >= 512:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.7)
            
            # Metadata completeness factor
            required_fields = ['style', 'context', 'target_audience']
            completeness = sum(1 for field in required_fields if metadata.get(field))
            confidence_factors.append(completeness / len(required_fields))
            
            # Content clarity factor
            content_clarity = self.content_moderator.assess_content_clarity(image)
            confidence_factors.append(content_clarity)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.7
    
    def verify_age_appropriate_content(self, image: Image.Image, target_age_group: str, 
                                     content_type: str) -> Dict[str, Any]:
        """
        Verify content is appropriate for target age group
        
        Args:
            image: PIL Image to verify
            target_age_group: Target age group ('all_ages', 'teen', 'adult', etc.)
            content_type: Type of content ('fashion_photography', 'advertisement', etc.)
            
        Returns:
            Age appropriateness verification results
        """
        try:
            # Analyze content for age appropriateness
            age_analysis = self.content_moderator.analyze_age_appropriateness(
                image, target_age_group, content_type
            )
            
            # Determine age rating
            age_rating = self._determine_age_rating(age_analysis)
            
            # Check if appropriate for target age group
            age_appropriate = self._check_age_group_compatibility(age_rating, target_age_group)
            
            # Identify restriction reasons
            restriction_reasons = self._identify_age_restriction_reasons(
                age_analysis, target_age_group
            )
            
            # Recommend appropriate age group
            recommended_age_group = self._recommend_age_group(age_rating)
            
            result = {
                'age_appropriate': age_appropriate,
                'age_rating': age_rating,
                'restriction_reasons': restriction_reasons,
                'recommended_age_group': recommended_age_group,
                'age_analysis': age_analysis
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Age appropriateness verification failed: {str(e)}")
            return self._create_fallback_age_verification_result()
    
    def _determine_age_rating(self, age_analysis: Dict[str, Any]) -> str:
        """Determine age rating based on content analysis"""
        try:
            # Extract analysis scores
            mature_content_score = age_analysis.get('mature_content_score', 0.0)
            violence_score = age_analysis.get('violence_score', 0.0)
            suggestive_score = age_analysis.get('suggestive_score', 0.0)
            
            # Determine rating based on scores
            if mature_content_score > 0.7 or violence_score > 0.7:
                return 'Adult'
            elif suggestive_score > 0.5 or mature_content_score > 0.4:
                return 'PG-13'
            elif suggestive_score > 0.2:
                return 'PG'
            else:
                return 'G'
                
        except Exception as e:
            self.logger.error(f"Age rating determination failed: {str(e)}")
            return 'PG-13'  # Conservative default
    
    def _check_age_group_compatibility(self, age_rating: str, target_age_group: str) -> bool:
        """Check if age rating is compatible with target age group"""
        try:
            age_group_lower = target_age_group.lower()
            
            # Age group compatibility mapping
            compatibility = {
                'G': ['all_ages', 'family', 'children', 'teen', 'adult'],
                'PG': ['family', 'teen', 'adult'],
                'PG-13': ['teen', 'adult'],
                'R': ['adult'],
                'Adult': ['adult']
            }
            
            compatible_groups = compatibility.get(age_rating, ['adult'])
            
            return any(group in age_group_lower for group in compatible_groups)
            
        except Exception as e:
            self.logger.error(f"Age compatibility check failed: {str(e)}")
            return False  # Conservative default
    
    def _identify_age_restriction_reasons(self, age_analysis: Dict[str, Any], 
                                        target_age_group: str) -> List[str]:
        """Identify reasons for age restrictions"""
        reasons = []
        
        try:
            # Check various content factors
            if age_analysis.get('mature_content_score', 0.0) > 0.3:
                reasons.append('mature_content_detected')
            
            if age_analysis.get('suggestive_score', 0.0) > 0.3:
                reasons.append('suggestive_content_detected')
            
            if age_analysis.get('violence_score', 0.0) > 0.2:
                reasons.append('violent_content_detected')
            
            if age_analysis.get('inappropriate_language', False):
                reasons.append('inappropriate_language')
            
            # Target-specific restrictions
            if 'all_ages' in target_age_group.lower() and age_analysis.get('overall_score', 1.0) < 0.95:
                reasons.append('not_suitable_for_all_ages')
                
        except Exception as e:
            self.logger.error(f"Age restriction reason identification failed: {str(e)}")
        
        return reasons
    
    def _recommend_age_group(self, age_rating: str) -> str:
        """Recommend appropriate age group based on rating"""
        rating_to_age_group = {
            'G': 'all_ages',
            'PG': 'family_friendly',
            'PG-13': 'teen_and_adult',
            'R': 'adult',
            'Adult': 'mature_adult'
        }
        
        return rating_to_age_group.get(age_rating, 'adult')
    
    def validate_professional_standards(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate content against professional standards
        
        Args:
            content_data: Content information and metadata
            
        Returns:
            Professional standards validation results
        """
        try:
            # Extract content information
            image_content = content_data.get('image_content', 'unknown')
            style_description = content_data.get('style_description', '')
            brand_context = content_data.get('brand_context', 'unknown')
            commercial_use = content_data.get('commercial_use', False)
            
            # Validate different professional aspects
            attire_compliance = self._validate_professional_attire(content_data)
            setting_compliance = self._validate_professional_setting(content_data)
            composition_compliance = self._validate_professional_composition(content_data)
            brand_compliance = self._validate_brand_standards(content_data)
            
            # Calculate overall compliance score
            compliance_score = (
                attire_compliance * 0.3 +
                setting_compliance * 0.2 +
                composition_compliance * 0.3 +
                brand_compliance * 0.2
            )
            
            # Determine if standards are met
            meets_standards = compliance_score >= self.safety_thresholds.professional_standard_min
            
            # Identify standard violations
            standard_violations = self._identify_professional_violations(
                content_data, compliance_score
            )
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_professional_improvements(
                content_data, standard_violations
            )
            
            result = {
                'meets_standards': meets_standards,
                'compliance_score': compliance_score,
                'standard_violations': standard_violations,
                'improvement_suggestions': improvement_suggestions,
                'component_scores': {
                    'attire_compliance': attire_compliance,
                    'setting_compliance': setting_compliance,
                    'composition_compliance': composition_compliance,
                    'brand_compliance': brand_compliance
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Professional standards validation failed: {str(e)}")
            return self._create_fallback_professional_result()
    
    def _validate_professional_attire(self, content_data: Dict[str, Any]) -> float:
        """Validate professional attire standards"""
        try:
            style_description = content_data.get('style_description', '').lower()
            
            # Professional attire keywords
            professional_keywords = [
                'business', 'formal', 'professional', 'suit', 'blazer',
                'dress_shirt', 'tie', 'formal_dress', 'elegant'
            ]
            
            casual_keywords = [
                'casual', 'beach', 'swimwear', 'revealing', 'inappropriate'
            ]
            
            # Score based on keywords
            professional_score = 0.7  # Base score
            
            if any(keyword in style_description for keyword in professional_keywords):
                professional_score += 0.2
            
            if any(keyword in style_description for keyword in casual_keywords):
                professional_score -= 0.3
            
            return max(0.0, min(1.0, professional_score))
            
        except Exception as e:
            self.logger.error(f"Professional attire validation failed: {str(e)}")
            return 0.7
    
    def _validate_professional_setting(self, content_data: Dict[str, Any]) -> float:
        """Validate professional setting standards"""
        try:
            image_content = content_data.get('image_content', '').lower()
            
            # Professional setting keywords
            professional_settings = [
                'office', 'studio', 'boardroom', 'conference', 'workplace',
                'professional_background', 'neutral_background'
            ]
            
            inappropriate_settings = [
                'bedroom', 'bathroom', 'inappropriate_location', 'party'
            ]
            
            setting_score = 0.8  # Base score
            
            if any(setting in image_content for setting in professional_settings):
                setting_score += 0.1
            
            if any(setting in image_content for setting in inappropriate_settings):
                setting_score -= 0.4
            
            return max(0.0, min(1.0, setting_score))
            
        except Exception as e:
            self.logger.error(f"Professional setting validation failed: {str(e)}")
            return 0.8
    
    def _validate_professional_composition(self, content_data: Dict[str, Any]) -> float:
        """Validate professional composition standards"""
        try:
            style_description = content_data.get('style_description', '').lower()
            
            # Professional composition elements
            professional_composition = [
                'professional_lighting', 'clean_composition', 'formal_pose',
                'appropriate_framing', 'high_quality'
            ]
            
            poor_composition = [
                'blurry', 'poor_lighting', 'inappropriate_pose', 'low_quality'
            ]
            
            composition_score = 0.8  # Base score
            
            if any(element in style_description for element in professional_composition):
                composition_score += 0.1
            
            if any(element in style_description for element in poor_composition):
                composition_score -= 0.3
            
            return max(0.0, min(1.0, composition_score))
            
        except Exception as e:
            self.logger.error(f"Professional composition validation failed: {str(e)}")
            return 0.8
    
    def _validate_brand_standards(self, content_data: Dict[str, Any]) -> float:
        """Validate brand-specific standards"""
        try:
            brand_context = content_data.get('brand_context', '').lower()
            commercial_use = content_data.get('commercial_use', False)
            
            # Brand standard assessment
            if brand_context == 'luxury_fashion':
                return 0.95  # High standards for luxury
            elif brand_context == 'professional_services':
                return 0.90  # High professional standards
            elif brand_context == 'casual_brand':
                return 0.80  # Moderate standards
            elif commercial_use:
                return 0.85  # Commercial use standards
            else:
                return 0.80  # General standards
                
        except Exception as e:
            self.logger.error(f"Brand standards validation failed: {str(e)}")
            return 0.8
    
    def _identify_professional_violations(self, content_data: Dict[str, Any], 
                                        score: float) -> List[str]:
        """Identify professional standard violations"""
        violations = []
        
        try:
            # Score-based violations
            if score < 0.7:
                violations.append('overall_professional_standard_violation')
            
            # Specific content violations
            style_description = content_data.get('style_description', '').lower()
            
            if 'inappropriate' in style_description:
                violations.append('inappropriate_content')
            
            if 'unprofessional' in style_description:
                violations.append('unprofessional_presentation')
            
            # Commercial use violations
            if content_data.get('commercial_use', False) and score < 0.85:
                violations.append('commercial_standard_violation')
                
        except Exception as e:
            self.logger.error(f"Professional violation identification failed: {str(e)}")
        
        return violations
    
    def _generate_professional_improvements(self, content_data: Dict[str, Any], 
                                          violations: List[str]) -> List[str]:
        """Generate professional improvement suggestions"""
        suggestions = []
        
        try:
            # General improvements
            suggestions.extend([
                "Ensure professional attire and presentation",
                "Use appropriate professional setting",
                "Maintain high-quality composition and lighting"
            ])
            
            # Violation-specific suggestions
            if 'inappropriate_content' in violations:
                suggestions.append("Remove inappropriate elements and ensure professional context")
            
            if 'commercial_standard_violation' in violations:
                suggestions.append("Enhance quality to meet commercial standard requirements")
            
            # Brand-specific suggestions
            brand_context = content_data.get('brand_context', '')
            if 'luxury' in brand_context.lower():
                suggestions.append("Ensure premium quality and luxury brand standards")
                
        except Exception as e:
            self.logger.error(f"Professional improvement generation failed: {str(e)}")
        
        return suggestions
    
    def detect_inappropriate_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect inappropriate content using comprehensive analysis
        
        Args:
            content_data: Content data to analyze
            
        Returns:
            Inappropriate content detection results
        """
        try:
            # Use content moderator for detection
            detection_result = self.content_moderator.detect_inappropriate_content(content_data)
            
            # Determine severity level
            severity_level = self._determine_inappropriate_severity(detection_result)
            
            # Determine if blocking is required
            blocking_required = self._determine_blocking_requirement(
                detection_result, severity_level
            )
            
            # Generate moderation actions
            moderation_actions = self._generate_moderation_actions(
                detection_result, severity_level, blocking_required
            )
            
            result = {
                'inappropriate_detected': detection_result.get('inappropriate_detected', False),
                'inappropriate_categories': detection_result.get('categories', []),
                'severity_level': severity_level,
                'blocking_required': blocking_required,
                'moderation_actions': moderation_actions,
                'confidence': detection_result.get('confidence', 0.0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inappropriate content detection failed: {str(e)}")
            return self._create_fallback_inappropriate_result()
    
    def _determine_inappropriate_severity(self, detection_result: Dict[str, Any]) -> str:
        """Determine severity level of inappropriate content"""
        try:
            confidence = detection_result.get('confidence', 0.0)
            categories = detection_result.get('categories', [])
            
            # High severity categories
            high_severity = ['sexually_explicit', 'violence', 'hate_speech', 'illegal_content']
            
            if any(cat in categories for cat in high_severity):
                return 'critical'
            elif confidence > 0.8:
                return 'high'
            elif confidence > 0.6:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"Severity determination failed: {str(e)}")
            return 'medium'
    
    def _determine_blocking_requirement(self, detection_result: Dict[str, Any], 
                                      severity: str) -> bool:
        """Determine if content should be blocked"""
        try:
            # Block based on severity
            if severity in ['critical', 'high']:
                return True
            elif severity == 'medium' and detection_result.get('confidence', 0.0) > 0.7:
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Blocking requirement determination failed: {str(e)}")
            return True  # Conservative default
    
    def _generate_moderation_actions(self, detection_result: Dict[str, Any], 
                                   severity: str, blocking_required: bool) -> List[str]:
        """Generate moderation actions"""
        actions = []
        
        try:
            if blocking_required:
                actions.extend([
                    "Block content from publication",
                    "Flag for human review",
                    "Notify content creator of violation"
                ])
            
            if severity in ['critical', 'high']:
                actions.extend([
                    "Escalate to senior moderation team",
                    "Document violation for policy review"
                ])
            
            if severity == 'medium':
                actions.append("Queue for enhanced review")
            
            actions.append("Log moderation decision")
            
        except Exception as e:
            self.logger.error(f"Moderation action generation failed: {str(e)}")
        
        return actions
    
    def enforce_ethics_guidelines(self, ethics_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce ethics guidelines on content
        
        Args:
            ethics_scenario: Ethical scenario to evaluate
            
        Returns:
            Ethics guideline enforcement results
        """
        try:
            # Use ethics engine for evaluation
            ethics_result = self.ethics_engine.evaluate_ethics_compliance(ethics_scenario)
            
            # Determine overall compliance
            ethics_compliant = ethics_result.get('ethics_score', 0.0) >= self.safety_thresholds.ethics_compliance_min
            
            # Identify guideline violations
            guideline_violations = self._identify_ethics_violations(ethics_result)
            
            # Generate ethical recommendations
            ethical_recommendations = self._generate_ethics_recommendations(
                ethics_result, guideline_violations
            )
            
            # Determine approval status
            approval_status = self._determine_ethics_approval(ethics_result, ethics_compliant)
            
            result = {
                'ethics_compliant': ethics_compliant,
                'ethics_score': ethics_result.get('ethics_score', 0.0),
                'guideline_violations': guideline_violations,
                'ethical_recommendations': ethical_recommendations,
                'approval_status': approval_status
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ethics guideline enforcement failed: {str(e)}")
            return self._create_fallback_ethics_result()
    
    def _identify_ethics_violations(self, ethics_result: Dict[str, Any]) -> List[str]:
        """Identify specific ethics violations"""
        violations = []
        
        try:
            # Check various ethical aspects
            if ethics_result.get('representation_bias', 0.0) > 0.3:
                violations.append('representation_bias_detected')
            
            if ethics_result.get('exploitation_risk', 0.0) > 0.2:
                violations.append('potential_exploitation')
            
            if ethics_result.get('consent_issues', False):
                violations.append('consent_verification_required')
            
            if ethics_result.get('harmful_stereotypes', False):
                violations.append('harmful_stereotypes_present')
                
        except Exception as e:
            self.logger.error(f"Ethics violation identification failed: {str(e)}")
        
        return violations
    
    def _generate_ethics_recommendations(self, ethics_result: Dict[str, Any], 
                                       violations: List[str]) -> List[str]:
        """Generate ethics improvement recommendations"""
        recommendations = []
        
        try:
            # General ethical recommendations
            recommendations.extend([
                "Ensure diverse and inclusive representation",
                "Verify consent for all individuals in content",
                "Avoid harmful stereotypes and biased representation"
            ])
            
            # Violation-specific recommendations
            if 'representation_bias_detected' in violations:
                recommendations.append("Improve diversity and representation in content")
            
            if 'potential_exploitation' in violations:
                recommendations.append("Review content for potential exploitation issues")
            
            if 'consent_verification_required' in violations:
                recommendations.append("Obtain proper consent documentation")
                
        except Exception as e:
            self.logger.error(f"Ethics recommendation generation failed: {str(e)}")
        
        return recommendations
    
    def _determine_ethics_approval(self, ethics_result: Dict[str, Any], compliant: bool) -> str:
        """Determine ethics approval status"""
        try:
            ethics_score = ethics_result.get('ethics_score', 0.0)
            
            if compliant and ethics_score >= 0.95:
                return 'approved'
            elif compliant:
                return 'conditionally_approved'
            elif ethics_score >= 0.8:
                return 'requires_review'
            else:
                return 'rejected'
                
        except Exception as e:
            self.logger.error(f"Ethics approval determination failed: {str(e)}")
            return 'requires_review'
    
    def analyze_multimodal_content(self, multimodal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze multi-modal content (image + text + metadata)
        
        Args:
            multimodal_data: Multi-modal content data
            
        Returns:
            Comprehensive multi-modal analysis results
        """
        try:
            # Extract components
            image = multimodal_data.get('image')
            text_description = multimodal_data.get('text_description', '')
            style_tags = multimodal_data.get('style_tags', [])
            intended_use = multimodal_data.get('intended_use', 'unknown')
            
            # Analyze each modality
            image_analysis = self._analyze_image_component(image) if image else {}
            text_analysis = self._analyze_text_component(text_description)
            contextual_analysis = self._analyze_contextual_component(style_tags, intended_use)
            
            # Check consistency across modalities
            consistency_check = self._check_multimodal_consistency(
                image_analysis, text_analysis, contextual_analysis
            )
            
            # Calculate overall safety score
            overall_safety_score = self._calculate_multimodal_safety_score(
                image_analysis, text_analysis, contextual_analysis, consistency_check
            )
            
            result = {
                'overall_safety_score': overall_safety_score,
                'image_analysis': image_analysis,
                'text_analysis': text_analysis,
                'contextual_analysis': contextual_analysis,
                'consistency_check': consistency_check,
                'multimodal_recommendations': self._generate_multimodal_recommendations(overall_safety_score)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-modal content analysis failed: {str(e)}")
            return self._create_fallback_multimodal_result()
    
    def _analyze_image_component(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image component of multi-modal content"""
        try:
            if image is None:
                return {'safety_score': 0.5, 'analysis_available': False}
            
            # Use content moderator for image analysis
            image_analysis = self.content_moderator.analyze_image_safety(image)
            return image_analysis
            
        except Exception as e:
            self.logger.error(f"Image component analysis failed: {str(e)}")
            return {'safety_score': 0.7, 'error': str(e)}
    
    def _analyze_text_component(self, text_description: str) -> Dict[str, Any]:
        """Analyze text component of multi-modal content"""
        try:
            # Basic text safety analysis
            text_lower = text_description.lower()
            
            # Check for inappropriate keywords
            inappropriate_keywords = [
                'inappropriate', 'explicit', 'offensive', 'harmful',
                'discriminatory', 'hateful', 'violent'
            ]
            
            professional_keywords = [
                'professional', 'business', 'elegant', 'sophisticated',
                'high-quality', 'respectful', 'appropriate'
            ]
            
            safety_score = 0.8  # Base score
            
            if any(keyword in text_lower for keyword in inappropriate_keywords):
                safety_score -= 0.4
            
            if any(keyword in text_lower for keyword in professional_keywords):
                safety_score += 0.1
            
            result = {
                'safety_score': max(0.0, min(1.0, safety_score)),
                'inappropriate_keywords_found': any(keyword in text_lower for keyword in inappropriate_keywords),
                'professional_indicators': any(keyword in text_lower for keyword in professional_keywords)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text component analysis failed: {str(e)}")
            return {'safety_score': 0.7, 'error': str(e)}
    
    def _analyze_contextual_component(self, style_tags: List[str], intended_use: str) -> Dict[str, Any]:
        """Analyze contextual component of multi-modal content"""
        try:
            # Analyze style tags
            professional_tags = ['professional', 'business', 'formal', 'elegant']
            inappropriate_tags = ['inappropriate', 'explicit', 'offensive']
            
            professional_tag_count = sum(1 for tag in style_tags if tag.lower() in professional_tags)
            inappropriate_tag_count = sum(1 for tag in style_tags if tag.lower() in inappropriate_tags)
            
            # Analyze intended use
            professional_uses = ['corporate', 'business', 'professional', 'commercial']
            inappropriate_uses = ['inappropriate', 'offensive', 'harmful']
            
            use_appropriate = any(use in intended_use.lower() for use in professional_uses)
            use_inappropriate = any(use in intended_use.lower() for use in inappropriate_uses)
            
            # Calculate contextual safety score
            contextual_score = 0.8  # Base score
            
            if inappropriate_tag_count > 0 or use_inappropriate:
                contextual_score -= 0.5
            
            if professional_tag_count > 0 or use_appropriate:
                contextual_score += 0.1
            
            result = {
                'safety_score': max(0.0, min(1.0, contextual_score)),
                'professional_indicators': professional_tag_count + (1 if use_appropriate else 0),
                'inappropriate_indicators': inappropriate_tag_count + (1 if use_inappropriate else 0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Contextual component analysis failed: {str(e)}")
            return {'safety_score': 0.7, 'error': str(e)}
    
    def _check_multimodal_consistency(self, image_analysis: Dict[str, Any], 
                                    text_analysis: Dict[str, Any],
                                    contextual_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency across multi-modal components"""
        try:
            # Get safety scores
            image_score = image_analysis.get('safety_score', 0.5)
            text_score = text_analysis.get('safety_score', 0.5)
            contextual_score = contextual_analysis.get('safety_score', 0.5)
            
            # Calculate consistency
            scores = [image_score, text_score, contextual_score]
            max_variance = max(scores) - min(scores)
            
            # Consistency thresholds
            if max_variance <= 0.1:
                consistency_level = 'high'
            elif max_variance <= 0.2:
                consistency_level = 'medium'
            else:
                consistency_level = 'low'
            
            result = {
                'consistency_level': consistency_level,
                'max_variance': max_variance,
                'consistent': max_variance <= 0.2,
                'component_scores': {
                    'image': image_score,
                    'text': text_score,
                    'contextual': contextual_score
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multimodal consistency check failed: {str(e)}")
            return {'consistency_level': 'medium', 'consistent': True}
    
    def _calculate_multimodal_safety_score(self, image_analysis: Dict[str, Any],
                                         text_analysis: Dict[str, Any],
                                         contextual_analysis: Dict[str, Any],
                                         consistency_check: Dict[str, Any]) -> float:
        """Calculate overall multi-modal safety score"""
        try:
            # Get component scores
            image_score = image_analysis.get('safety_score', 0.7)
            text_score = text_analysis.get('safety_score', 0.7)
            contextual_score = contextual_analysis.get('safety_score', 0.7)
            
            # Weight components
            weighted_score = (
                image_score * 0.5 +
                text_score * 0.3 +
                contextual_score * 0.2
            )
            
            # Apply consistency bonus/penalty
            consistency_level = consistency_check.get('consistency_level', 'medium')
            if consistency_level == 'high':
                weighted_score += 0.05
            elif consistency_level == 'low':
                weighted_score -= 0.1
            
            return max(0.0, min(1.0, weighted_score))
            
        except Exception as e:
            self.logger.error(f"Multimodal safety score calculation failed: {str(e)}")
            return 0.7
    
    def _generate_multimodal_recommendations(self, overall_score: float) -> List[str]:
        """Generate multi-modal content recommendations"""
        recommendations = []
        
        try:
            if overall_score < 0.5:
                recommendations.extend([
                    "Significant improvements needed across all content components",
                    "Review image, text, and contextual elements for safety violations",
                    "Consider complete content revision"
                ])
            elif overall_score < 0.7:
                recommendations.extend([
                    "Moderate improvements needed for content safety",
                    "Review and adjust content components for better alignment"
                ])
            else:
                recommendations.extend([
                    "Content meets safety standards",
                    "Continue monitoring for consistency",
                    "Maintain current quality levels"
                ])
                
        except Exception as e:
            self.logger.error(f"Multimodal recommendation generation failed: {str(e)}")
        
        return recommendations
    
    async def moderate_content_realtime(self, image: Image.Image, 
                                      content_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real-time content moderation for immediate decisions
        
        Args:
            image: PIL Image to moderate
            content_metadata: Content metadata and context
            
        Returns:
            Real-time moderation decision and results
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Fast-track moderation analysis
            moderation_tasks = [
                self._quick_appropriateness_check(image),
                self._quick_safety_scan(content_metadata),
                self._quick_professional_check(content_metadata)
            ]
            
            # Run checks concurrently
            appropriateness_result, safety_result, professional_result = await asyncio.gather(*moderation_tasks)
            
            # Calculate overall safety score
            safety_score = (
                appropriateness_result.get('score', 0.7) * 0.4 +
                safety_result.get('score', 0.7) * 0.4 +
                professional_result.get('score', 0.7) * 0.2
            )
            
            # Make moderation decision
            moderation_decision = self._make_realtime_decision(
                safety_score, appropriateness_result, safety_result, professional_result
            )
            
            # Determine automated actions
            automated_actions = self._determine_automated_actions(moderation_decision, safety_score)
            
            # Check if human review is required
            human_review_required = self._requires_human_review_realtime(
                safety_score, moderation_decision
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            result = {
                'moderation_decision': moderation_decision,
                'safety_score': safety_score,
                'automated_actions': automated_actions,
                'human_review_required': human_review_required,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'component_results': {
                    'appropriateness': appropriateness_result,
                    'safety': safety_result,
                    'professional': professional_result
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Real-time content moderation failed: {str(e)}")
            return self._create_fallback_realtime_result()
    
    async def _quick_appropriateness_check(self, image: Image.Image) -> Dict[str, Any]:
        """Quick appropriateness check for real-time moderation"""
        try:
            # Fast image analysis
            image_array = np.array(image)
            
            # Basic appropriateness indicators
            brightness = np.mean(image_array)
            contrast = np.std(image_array)
            
            # Simple appropriateness scoring
            if 50 <= brightness <= 200 and contrast > 20:
                score = 0.9
            elif 30 <= brightness <= 220:
                score = 0.7
            else:
                score = 0.5
            
            return {'score': score, 'check_type': 'appropriateness'}
            
        except Exception as e:
            self.logger.error(f"Quick appropriateness check failed: {str(e)}")
            return {'score': 0.7, 'error': str(e)}
    
    async def _quick_safety_scan(self, content_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Quick safety scan for real-time moderation"""
        try:
            # Quick metadata safety check
            style = content_metadata.get('style', '').lower()
            context = content_metadata.get('context', '').lower()
            
            # Safety keywords
            safe_keywords = ['professional', 'business', 'elegant', 'appropriate']
            unsafe_keywords = ['inappropriate', 'explicit', 'offensive']
            
            score = 0.8  # Base score
            
            if any(keyword in style or keyword in context for keyword in safe_keywords):
                score += 0.1
            
            if any(keyword in style or keyword in context for keyword in unsafe_keywords):
                score -= 0.4
            
            return {'score': max(0.0, min(1.0, score)), 'check_type': 'safety'}
            
        except Exception as e:
            self.logger.error(f"Quick safety scan failed: {str(e)}")
            return {'score': 0.7, 'error': str(e)}
    
    async def _quick_professional_check(self, content_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Quick professional standard check for real-time moderation"""
        try:
            # Quick professional assessment
            commercial_use = content_metadata.get('commercial_use', False)
            brand_context = content_metadata.get('brand_context', '').lower()
            
            if 'luxury' in brand_context:
                score = 0.95
            elif 'professional' in brand_context:
                score = 0.90
            elif commercial_use:
                score = 0.85
            else:
                score = 0.80
            
            return {'score': score, 'check_type': 'professional'}
            
        except Exception as e:
            self.logger.error(f"Quick professional check failed: {str(e)}")
            return {'score': 0.8, 'error': str(e)}
    
    def _make_realtime_decision(self, safety_score: float, appropriateness_result: Dict[str, Any],
                              safety_result: Dict[str, Any], professional_result: Dict[str, Any]) -> str:
        """Make real-time moderation decision"""
        try:
            if safety_score >= 0.9:
                return 'approve'
            elif safety_score >= 0.7:
                return 'conditional_approve'
            elif safety_score >= 0.5:
                return 'flag_for_review'
            else:
                return 'block'
                
        except Exception as e:
            self.logger.error(f"Realtime decision making failed: {str(e)}")
            return 'flag_for_review'
    
    def _determine_automated_actions(self, decision: str, safety_score: float) -> List[str]:
        """Determine automated actions based on moderation decision"""
        actions = []
        
        try:
            if decision == 'approve':
                actions.extend(['log_approval', 'proceed_with_processing'])
            elif decision == 'conditional_approve':
                actions.extend(['log_conditional_approval', 'enhanced_monitoring'])
            elif decision == 'flag_for_review':
                actions.extend(['queue_for_human_review', 'temporary_hold'])
            elif decision == 'block':
                actions.extend(['block_content', 'notify_violation', 'escalate_review'])
            
            # Always log the decision
            actions.append('log_moderation_decision')
            
        except Exception as e:
            self.logger.error(f"Automated action determination failed: {str(e)}")
        
        return actions
    
    def _requires_human_review_realtime(self, safety_score: float, decision: str) -> bool:
        """Determine if human review is required for real-time moderation"""
        try:
            # Always require human review for blocked content
            if decision == 'block':
                return True
            
            # Require review for borderline scores
            if 0.5 <= safety_score <= 0.7:
                return True
            
            # Require review for flagged content
            if decision == 'flag_for_review':
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Human review requirement assessment failed: {str(e)}")
            return True  # Conservative default
    
    def calculate_safety_score(self, content_example: Dict[str, Any]) -> float:
        """
        Calculate comprehensive safety score for content
        
        Args:
            content_example: Content example with type and characteristics
            
        Returns:
            Overall safety score (0-1 scale)
        """
        try:
            content_type = content_example.get('type', 'unknown').lower()
            
            # Base scores by content type
            type_scores = {
                'professional_headshot': 0.95,
                'business_attire': 0.92,
                'artistic_fashion': 0.85,
                'casual_wear': 0.88,
                'formal_event': 0.90,
                'lifestyle_photography': 0.85,
                'commercial_photography': 0.88,
                'inappropriate_content': 0.15,
                'explicit_content': 0.05,
                'offensive_content': 0.10
            }
            
            base_score = type_scores.get(content_type, 0.7)
            
            # Adjust based on additional characteristics
            characteristics = content_example.get('characteristics', [])
            
            for characteristic in characteristics:
                if 'professional' in characteristic.lower():
                    base_score += 0.05
                elif 'inappropriate' in characteristic.lower():
                    base_score -= 0.3
                elif 'high_quality' in characteristic.lower():
                    base_score += 0.03
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            self.logger.error(f"Safety score calculation failed: {str(e)}")
            return 0.7
    
    def enforce_safety_thresholds(self, safety_score: float) -> Dict[str, Any]:
        """
        Enforce safety thresholds and determine actions
        
        Args:
            safety_score: Calculated safety score
            
        Returns:
            Threshold enforcement results and actions
        """
        try:
            # Determine threshold violations
            threshold_violations = []
            
            if safety_score < self.safety_thresholds.overall_safety_min:
                threshold_violations.append('overall_safety_threshold_violation')
            
            if safety_score < self.safety_thresholds.appropriateness_min:
                threshold_violations.append('appropriateness_threshold_violation')
            
            # Determine if content should be blocked
            content_blocked = len(threshold_violations) > 0 or safety_score < 0.5
            
            # Generate enforcement actions
            enforcement_actions = self._generate_enforcement_actions(
                safety_score, threshold_violations, content_blocked
            )
            
            result = {
                'content_blocked': content_blocked,
                'threshold_violations': threshold_violations,
                'enforcement_actions': enforcement_actions,
                'safety_score': safety_score,
                'threshold_met': not content_blocked
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Safety threshold enforcement failed: {str(e)}")
            return {'content_blocked': True, 'error': str(e)}
    
    def _generate_enforcement_actions(self, safety_score: float, violations: List[str], 
                                    blocked: bool) -> List[str]:
        """Generate enforcement actions based on threshold violations"""
        actions = []
        
        try:
            if blocked:
                actions.extend([
                    'block_content_publication',
                    'notify_content_creator',
                    'log_violation'
                ])
            
            if safety_score < 0.3:
                actions.extend([
                    'escalate_to_senior_review',
                    'flag_for_policy_review'
                ])
            elif safety_score < 0.7:
                actions.append('queue_for_enhanced_review')
            
            # Violation-specific actions
            if 'appropriateness_threshold_violation' in violations:
                actions.append('require_appropriateness_improvement')
            
            actions.append('update_safety_metrics')
            
        except Exception as e:
            self.logger.error(f"Enforcement action generation failed: {str(e)}")
        
        return actions
    
    def queue_for_human_review(self, edge_case_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Queue content for human review
        
        Args:
            edge_case_content: Content requiring human review
            
        Returns:
            Human review queue information
        """
        try:
            # Determine review priority
            safety_score = edge_case_content.get('safety_score', 0.5)
            content_type = edge_case_content.get('content_type', 'unknown')
            
            if safety_score < 0.3:
                priority = 'critical'
                estimated_time = '2-4 hours'
            elif safety_score < 0.5:
                priority = 'high'
                estimated_time = '4-8 hours'
            elif safety_score < 0.7:
                priority = 'medium'
                estimated_time = '1-2 days'
            else:
                priority = 'low'
                estimated_time = '2-3 days'
            
            # Determine reviewer categories
            reviewer_categories = self._determine_reviewer_categories(edge_case_content)
            
            # Determine interim action
            interim_action = self._determine_interim_action(safety_score, content_type)
            
            # Add to review queue
            review_item = {
                'content': edge_case_content,
                'priority': priority,
                'reviewer_categories': reviewer_categories,
                'submitted_at': datetime.now(),
                'estimated_review_time': estimated_time,
                'interim_action': interim_action
            }
            
            self.human_review_queue.append(review_item)
            
            result = {
                'queued_for_review': True,
                'review_priority': priority,
                'estimated_review_time': estimated_time,
                'interim_action': interim_action,
                'reviewer_categories': reviewer_categories,
                'queue_position': len(self.human_review_queue)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Human review queueing failed: {str(e)}")
            return {'queued_for_review': False, 'error': str(e)}
    
    def _determine_reviewer_categories(self, content: Dict[str, Any]) -> List[str]:
        """Determine required reviewer categories"""
        categories = ['content_moderator']  # Base category
        
        try:
            content_type = content.get('content_type', '').lower()
            safety_score = content.get('safety_score', 0.5)
            
            # Add specialists based on content
            if 'cultural' in content_type:
                categories.append('cultural_specialist')
            
            if 'artistic' in content_type:
                categories.append('artistic_reviewer')
            
            if safety_score < 0.3:
                categories.append('senior_moderator')
            
            if content.get('commercial_use', False):
                categories.append('commercial_compliance_reviewer')
                
        except Exception as e:
            self.logger.error(f"Reviewer category determination failed: {str(e)}")
        
        return categories
    
    def _determine_interim_action(self, safety_score: float, content_type: str) -> str:
        """Determine interim action while awaiting review"""
        try:
            if safety_score < 0.3:
                return 'block_pending_review'
            elif safety_score < 0.5:
                return 'hold_with_warning'
            elif safety_score < 0.7:
                return 'limited_distribution_pending_review'
            else:
                return 'proceed_with_monitoring'
                
        except Exception as e:
            self.logger.error(f"Interim action determination failed: {str(e)}")
            return 'hold_pending_review'
    
    def generate_safety_report(self, content_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive safety report from content history
        
        Args:
            content_history: Historical content processing data
            
        Returns:
            Comprehensive safety analytics report
        """
        try:
            total_content = len(content_history)
            
            if total_content == 0:
                return {'error': 'No content history provided'}
            
            # Calculate safety metrics
            safety_scores = [item.get('safety_score', 0.0) for item in content_history]
            blocked_content = [item for item in content_history if item.get('blocked', False)]
            
            # Safety score distribution
            score_distribution = self._calculate_score_distribution(safety_scores)
            
            # Blocking rate
            blocking_rate = len(blocked_content) / total_content
            
            # Violation categories analysis
            violation_categories = self._analyze_violation_categories(content_history)
            
            # Trends analysis
            trends_analysis = self._analyze_safety_trends(content_history)
            
            # Generate recommendations
            recommendations = self._generate_safety_report_recommendations(
                blocking_rate, score_distribution, violation_categories
            )
            
            report = {
                'total_content_processed': total_content,
                'safety_score_distribution': score_distribution,
                'blocking_rate': blocking_rate,
                'violation_categories': violation_categories,
                'trends_analysis': trends_analysis,
                'recommendations': recommendations,
                'report_generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Safety report generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_score_distribution(self, safety_scores: List[float]) -> Dict[str, Any]:
        """Calculate safety score distribution"""
        try:
            if not safety_scores:
                return {}
            
            # Score ranges
            excellent = sum(1 for score in safety_scores if score >= 0.9)
            good = sum(1 for score in safety_scores if 0.8 <= score < 0.9)
            acceptable = sum(1 for score in safety_scores if 0.7 <= score < 0.8)
            concerning = sum(1 for score in safety_scores if 0.5 <= score < 0.7)
            poor = sum(1 for score in safety_scores if score < 0.5)
            
            total = len(safety_scores)
            
            return {
                'excellent': {'count': excellent, 'percentage': excellent / total},
                'good': {'count': good, 'percentage': good / total},
                'acceptable': {'count': acceptable, 'percentage': acceptable / total},
                'concerning': {'count': concerning, 'percentage': concerning / total},
                'poor': {'count': poor, 'percentage': poor / total},
                'average_score': sum(safety_scores) / total
            }
            
        except Exception as e:
            self.logger.error(f"Score distribution calculation failed: {str(e)}")
            return {}
    
    def _analyze_violation_categories(self, content_history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze violation categories in content history"""
        try:
            violation_counts = {}
            
            for item in content_history:
                violations = item.get('violations', [])
                for violation in violations:
                    violation_counts[violation] = violation_counts.get(violation, 0) + 1
            
            return violation_counts
            
        except Exception as e:
            self.logger.error(f"Violation category analysis failed: {str(e)}")
            return {}
    
    def _analyze_safety_trends(self, content_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze safety trends over time"""
        try:
            # Simple trend analysis (in production, use more sophisticated time series analysis)
            recent_items = content_history[-10:] if len(content_history) >= 10 else content_history
            earlier_items = content_history[:-10] if len(content_history) >= 20 else content_history[:10]
            
            if not recent_items or not earlier_items:
                return {'trend': 'insufficient_data'}
            
            recent_avg = sum(item.get('safety_score', 0.0) for item in recent_items) / len(recent_items)
            earlier_avg = sum(item.get('safety_score', 0.0) for item in earlier_items) / len(earlier_items)
            
            trend_direction = 'improving' if recent_avg > earlier_avg else 'declining'
            trend_magnitude = abs(recent_avg - earlier_avg)
            
            return {
                'trend': trend_direction,
                'magnitude': trend_magnitude,
                'recent_average': recent_avg,
                'earlier_average': earlier_avg
            }
            
        except Exception as e:
            self.logger.error(f"Safety trends analysis failed: {str(e)}")
            return {'trend': 'unknown'}
    
    def _generate_safety_report_recommendations(self, blocking_rate: float, 
                                              score_distribution: Dict[str, Any],
                                              violations: Dict[str, int]) -> List[str]:
        """Generate recommendations based on safety report"""
        recommendations = []
        
        try:
            # Blocking rate recommendations
            if blocking_rate > 0.2:
                recommendations.append("High blocking rate detected - review content guidelines")
            elif blocking_rate > 0.1:
                recommendations.append("Moderate blocking rate - monitor content quality")
            
            # Score distribution recommendations
            if score_distribution.get('poor', {}).get('percentage', 0) > 0.1:
                recommendations.append("High percentage of poor safety scores - improve content standards")
            
            # Violation-specific recommendations
            if violations:
                top_violation = max(violations.items(), key=lambda x: x[1])
                recommendations.append(f"Most common violation: {top_violation[0]} - targeted improvement needed")
            
            # General recommendations
            recommendations.extend([
                "Continue regular safety monitoring",
                "Maintain content quality standards",
                "Update safety guidelines as needed"
            ])
            
        except Exception as e:
            self.logger.error(f"Safety report recommendation generation failed: {str(e)}")
        
        return recommendations
    
    def assess_cultural_content_safety(self, cultural_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess cultural content safety in integration with cultural sensitivity
        
        Args:
            cultural_content: Cultural content data with sensitivity information
            
        Returns:
            Cultural content safety assessment
        """
        try:
            # Extract cultural information
            cultural_elements = cultural_content.get('cultural_elements', [])
            appropriation_risk = cultural_content.get('cultural_appropriation_risk', 0.0)
            cultural_context = cultural_content.get('cultural_context', 'unknown')
            consultation = cultural_content.get('cultural_consultation', False)
            
            # Calculate cultural safety score
            cultural_safety_score = self._calculate_cultural_safety_score(
                appropriation_risk, cultural_context, consultation
            )
            
            # Identify cultural violations
            cultural_violations = self._identify_cultural_safety_violations(
                appropriation_risk, cultural_context, consultation
            )
            
            # Generate cultural recommendations
            cultural_recommendations = self._generate_cultural_safety_recommendations(
                cultural_violations, appropriation_risk
            )
            
            # Determine if culturally safe
            culturally_safe = (
                cultural_safety_score >= 0.8 and 
                appropriation_risk <= 0.3 and
                len(cultural_violations) == 0
            )
            
            result = {
                'culturally_safe': culturally_safe,
                'cultural_safety_score': cultural_safety_score,
                'cultural_violations': cultural_violations,
                'cultural_recommendations': cultural_recommendations,
                'appropriation_risk_assessment': appropriation_risk
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cultural content safety assessment failed: {str(e)}")
            return {'culturally_safe': False, 'error': str(e)}
    
    def _calculate_cultural_safety_score(self, appropriation_risk: float, 
                                       context: str, consultation: bool) -> float:
        """Calculate cultural safety score"""
        try:
            # Base score
            safety_score = 1.0 - appropriation_risk
            
            # Context adjustments
            if 'educational' in context.lower():
                safety_score += 0.1
            elif 'costume' in context.lower() or 'halloween' in context.lower():
                safety_score -= 0.4
            
            # Consultation bonus
            if consultation:
                safety_score += 0.2
            
            return max(0.0, min(1.0, safety_score))
            
        except Exception as e:
            self.logger.error(f"Cultural safety score calculation failed: {str(e)}")
            return 0.5
    
    def _identify_cultural_safety_violations(self, appropriation_risk: float, 
                                          context: str, consultation: bool) -> List[str]:
        """Identify cultural safety violations"""
        violations = []
        
        try:
            if appropriation_risk > 0.7:
                violations.append('high_cultural_appropriation_risk')
            
            if 'costume' in context.lower() and appropriation_risk > 0.3:
                violations.append('cultural_item_used_as_costume')
            
            if appropriation_risk > 0.5 and not consultation:
                violations.append('cultural_consultation_required')
                
        except Exception as e:
            self.logger.error(f"Cultural safety violation identification failed: {str(e)}")
        
        return violations
    
    def _generate_cultural_safety_recommendations(self, violations: List[str], 
                                                appropriation_risk: float) -> List[str]:
        """Generate cultural safety recommendations"""
        recommendations = []
        
        try:
            if 'high_cultural_appropriation_risk' in violations:
                recommendations.extend([
                    "High cultural appropriation risk detected",
                    "Seek cultural consultation before proceeding",
                    "Consider alternative designs without cultural elements"
                ])
            
            if 'cultural_item_used_as_costume' in violations:
                recommendations.append("Cultural items should not be used as costumes")
            
            if 'cultural_consultation_required' in violations:
                recommendations.append("Cultural consultation required for this content")
            
            if appropriation_risk > 0.3:
                recommendations.append("Enhanced cultural sensitivity monitoring recommended")
                
        except Exception as e:
            self.logger.error(f"Cultural safety recommendation generation failed: {str(e)}")
        
        return recommendations
    
    def handle_edge_case_content(self, edge_case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle edge cases in content safety processing
        
        Args:
            edge_case_data: Edge case content data
            
        Returns:
            Edge case handling results
        """
        try:
            case_type = self._identify_edge_case_type(edge_case_data)
            
            # Handle different edge case types
            if case_type == 'empty_content':
                safety_decision = 'block'
                fallback_used = True
            elif case_type == 'minimal_data':
                safety_decision = 'require_additional_information'
                fallback_used = True
            elif case_type == 'corrupted_metadata':
                safety_decision = 'use_image_only_analysis'
                fallback_used = True
            elif case_type == 'extreme_values':
                safety_decision = 'normalize_and_reprocess'
                fallback_used = True
            else:
                safety_decision = 'manual_review_required'
                fallback_used = False
            
            result = {
                'handled': True,
                'case_type': case_type,
                'fallback_used': fallback_used,
                'safety_decision': safety_decision,
                'requires_manual_review': case_type == 'unknown'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Edge case handling failed: {str(e)}")
            return {'handled': False, 'error': str(e)}
    
    def _identify_edge_case_type(self, data: Dict[str, Any]) -> str:
        """Identify type of edge case"""
        try:
            if not data or len(data) == 0:
                return 'empty_content'
            
            if len(data) == 1 and list(data.values())[0] in [None, '', 'unknown']:
                return 'minimal_data'
            
            if any(value is None for value in data.values()):
                return 'corrupted_metadata'
            
            # Check for extreme values
            numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
            if any(abs(v) > 10 for v in numeric_values):
                return 'extreme_values'
            
            return 'unknown'
            
        except Exception as e:
            self.logger.error(f"Edge case type identification failed: {str(e)}")
            return 'unknown'
    
    def process_content_batch(self, batch_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process batch of content for safety validation
        
        Args:
            batch_content: List of content items to process
            
        Returns:
            List of safety validation results
        """
        try:
            results = []
            
            for content_item in batch_content:
                # Process each item
                image = content_item.get('image')
                metadata = content_item.get('metadata', {})
                
                if image:
                    # Full validation for items with images
                    validation_result = self.validate_content_appropriateness(image, metadata)
                    safety_score = validation_result.get('appropriateness_score', 0.7)
                else:
                    # Metadata-only validation
                    safety_score = self._validate_metadata_only(metadata)
                
                results.append({
                    'safety_score': safety_score,
                    'validation_result': validation_result if image else None,
                    'metadata_validation': True
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch content processing failed: {str(e)}")
            return [{'error': str(e)} for _ in batch_content]
    
    def _validate_metadata_only(self, metadata: Dict[str, Any]) -> float:
        """Validate content based on metadata only"""
        try:
            style = metadata.get('style', '').lower()
            context = metadata.get('context', '').lower()
            
            # Basic metadata safety scoring
            safety_keywords = ['professional', 'business', 'appropriate']
            unsafe_keywords = ['inappropriate', 'explicit', 'offensive']
            
            score = 0.7  # Base score
            
            if any(keyword in style or keyword in context for keyword in safety_keywords):
                score += 0.2
            
            if any(keyword in style or keyword in context for keyword in unsafe_keywords):
                score -= 0.4
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Metadata-only validation failed: {str(e)}")
            return 0.5
    
    def get_safety_configuration(self) -> Dict[str, Any]:
        """Get current safety configuration"""
        try:
            return {
                'safety_thresholds': {
                    'appropriateness_min': self.safety_thresholds.appropriateness_min,
                    'professional_standard_min': self.safety_thresholds.professional_standard_min,
                    'age_appropriate_min': self.safety_thresholds.age_appropriate_min,
                    'ethics_compliance_min': self.safety_thresholds.ethics_compliance_min,
                    'overall_safety_min': self.safety_thresholds.overall_safety_min
                },
                'moderation_rules': {
                    'inappropriate_categories': self.inappropriate_categories,
                    'professional_standards': self.professional_standards
                },
                'review_criteria': {
                    'human_review_threshold': 0.7,
                    'blocking_threshold': 0.5,
                    'automatic_approval_threshold': 0.9
                }
            }
            
        except Exception as e:
            self.logger.error(f"Safety configuration retrieval failed: {str(e)}")
            return {'error': str(e)}
    
    def update_safety_configuration(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update safety configuration"""
        try:
            # Update safety thresholds if provided
            if 'safety_thresholds' in new_config:
                thresholds = new_config['safety_thresholds']
                
                if 'appropriateness_min' in thresholds:
                    self.safety_thresholds.appropriateness_min = thresholds['appropriateness_min']
                
                if 'professional_standard_min' in thresholds:
                    self.safety_thresholds.professional_standard_min = thresholds['professional_standard_min']
                
                # Update other thresholds as needed
            
            # Update moderation rules if provided
            if 'moderation_rules' in new_config:
                rules = new_config['moderation_rules']
                
                if 'inappropriate_categories' in rules:
                    self.inappropriate_categories = rules['inappropriate_categories']
                
                if 'professional_standards' in rules:
                    self.professional_standards = rules['professional_standards']
            
            self.logger.info("Safety configuration updated successfully")
            return {'success': True, 'updated_at': datetime.now().isoformat()}
            
        except Exception as e:
            self.logger.error(f"Safety configuration update failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    # Fallback result creation methods
    def _create_fallback_appropriateness_result(self) -> Dict[str, Any]:
        """Create fallback appropriateness result"""
        return {
            'is_appropriate': False,
            'appropriateness_score': 0.5,
            'violation_categories': ['analysis_error'],
            'recommendations': ['Manual review required due to analysis error'],
            'confidence': 0.0,
            'error': True
        }
    
    def _create_fallback_age_verification_result(self) -> Dict[str, Any]:
        """Create fallback age verification result"""
        return {
            'age_appropriate': False,
            'age_rating': 'PG-13',
            'restriction_reasons': ['analysis_error'],
            'recommended_age_group': 'adult',
            'error': True
        }
    
    def _create_fallback_professional_result(self) -> Dict[str, Any]:
        """Create fallback professional standards result"""
        return {
            'meets_standards': False,
            'compliance_score': 0.5,
            'standard_violations': ['analysis_error'],
            'improvement_suggestions': ['Manual review required'],
            'error': True
        }
    
    def _create_fallback_inappropriate_result(self) -> Dict[str, Any]:
        """Create fallback inappropriate content result"""
        return {
            'inappropriate_detected': True,
            'inappropriate_categories': ['analysis_error'],
            'severity_level': 'medium',
            'blocking_required': True,
            'moderation_actions': ['manual_review_required'],
            'error': True
        }
    
    def _create_fallback_ethics_result(self) -> Dict[str, Any]:
        """Create fallback ethics result"""
        return {
            'ethics_compliant': False,
            'ethics_score': 0.5,
            'guideline_violations': ['analysis_error'],
            'ethical_recommendations': ['Manual ethics review required'],
            'approval_status': 'requires_review',
            'error': True
        }
    
    def _create_fallback_multimodal_result(self) -> Dict[str, Any]:
        """Create fallback multi-modal result"""
        return {
            'overall_safety_score': 0.5,
            'image_analysis': {'safety_score': 0.5, 'error': True},
            'text_analysis': {'safety_score': 0.5, 'error': True},
            'contextual_analysis': {'safety_score': 0.5, 'error': True},
            'consistency_check': {'consistent': False, 'error': True},
            'multimodal_recommendations': ['Manual review required due to analysis error'],
            'error': True
        }
    
    def _create_fallback_realtime_result(self) -> Dict[str, Any]:
        """Create fallback real-time result"""
        return {
            'moderation_decision': 'flag_for_review',
            'safety_score': 0.5,
            'automated_actions': ['manual_review_required'],
            'human_review_required': True,
            'processing_time': 0.0,
            'timestamp': datetime.now().isoformat(),
            'error': True
        }