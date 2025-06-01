"""
Cultural Sensitivity Service for Garment Creative AI

This service provides comprehensive cultural sensitivity validation,
appropriation detection, and cultural appropriateness assessment.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from app.services.ethics.cultural_database import CulturalDatabase
from app.services.ethics.cultural_validator import CulturalValidator


@dataclass
class CulturalSensitivityResult:
    """Result structure for cultural sensitivity analysis"""
    cultural_sensitivity_score: float
    cultural_items_detected: List[Dict[str, Any]]
    appropriateness_assessment: Dict[str, float]
    appropriation_risk: float
    recommendations: List[str]
    expert_review_required: bool
    timestamp: datetime


@dataclass
class CulturalContext:
    """Cultural context information"""
    culture: str
    region: str
    significance_level: str
    sacred_level: str
    appropriate_contexts: List[str]
    inappropriate_contexts: List[str]


class CulturalSensitivityService:
    """
    Comprehensive cultural sensitivity and appropriation prevention service
    
    Provides cultural garment recognition, appropriateness validation,
    sacred content protection, and expert review workflows.
    """
    
    def __init__(self):
        """Initialize cultural sensitivity service"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.cultural_database = CulturalDatabase()
        self.cultural_validator = CulturalValidator()
        
        # Sensitivity thresholds
        self.sensitivity_thresholds = {
            'appropriateness_min': 0.95,
            'sacred_respect_min': 0.98,
            'cultural_accuracy_min': 0.90,
            'appropriation_risk_max': 0.3,
            'expert_review_threshold': 0.7
        }
        
        # Cultural categories
        self.cultural_significance_levels = ['low', 'medium', 'high', 'sacred', 'extremely_sacred']
        self.appropriation_risk_levels = ['low', 'medium', 'high', 'critical']
        
        # Expert review queue
        self.expert_review_queue = []
        
        self.logger.info("Cultural Sensitivity Service initialized")
    
    def recognize_cultural_garments(self, image: Image.Image) -> Dict[str, Any]:
        """
        Recognize cultural garments and accessories in image
        
        Args:
            image: PIL Image to analyze for cultural garments
            
        Returns:
            Cultural garment recognition results with confidence scores
        """
        try:
            # Convert image for analysis
            image_array = np.array(image)
            
            # Analyze for cultural garment patterns and features
            cultural_analysis = self._analyze_cultural_features(image_array)
            
            # Match against cultural database
            recognized_garments = self._match_cultural_database(cultural_analysis)
            
            # Calculate overall confidence
            if recognized_garments:
                confidences = [item.get('confidence', 0.0) for item in recognized_garments]
                overall_confidence = np.mean(confidences)
            else:
                overall_confidence = 0.0
            
            # Assess appropriation risk
            appropriation_risk = self._assess_appropriation_risk(recognized_garments)
            
            result = {
                'garments_detected': recognized_garments,
                'cultural_significance': self._assess_cultural_significance(recognized_garments),
                'confidence': overall_confidence,
                'appropriation_risk': appropriation_risk,
                'cultural_contexts': self._extract_cultural_contexts(recognized_garments)
            }
            
            self.logger.info(f"Cultural garment recognition completed: {len(recognized_garments)} items detected")
            return result
            
        except Exception as e:
            self.logger.error(f"Cultural garment recognition failed: {str(e)}")
            return self._create_fallback_recognition_result()
    
    def _analyze_cultural_features(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image for cultural garment features"""
        try:
            # Color analysis for cultural patterns
            dominant_colors = self._extract_dominant_colors(image_array)
            
            # Pattern analysis for cultural motifs
            pattern_features = self._analyze_patterns(image_array)
            
            # Shape analysis for garment types
            shape_features = self._analyze_garment_shapes(image_array)
            
            # Texture analysis for fabric types
            texture_features = self._analyze_textures(image_array)
            
            return {
                'colors': dominant_colors,
                'patterns': pattern_features,
                'shapes': shape_features,
                'textures': texture_features
            }
            
        except Exception as e:
            self.logger.error(f"Cultural feature analysis failed: {str(e)}")
            return {}
    
    def _extract_dominant_colors(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Extract dominant colors from image"""
        try:
            # Reshape image for k-means clustering
            pixels = image_array.reshape(-1, 3)
            
            # Simple color extraction (in production, use advanced clustering)
            unique_colors = []
            for i in range(0, len(pixels), len(pixels)//10):  # Sample colors
                color = pixels[i]
                color_info = {
                    'rgb': color.tolist(),
                    'cultural_significance': self._assess_color_cultural_significance(color)
                }
                unique_colors.append(color_info)
            
            return unique_colors[:5]  # Top 5 colors
            
        except Exception as e:
            self.logger.error(f"Color extraction failed: {str(e)}")
            return []
    
    def _assess_color_cultural_significance(self, color: np.ndarray) -> Dict[str, Any]:
        """Assess cultural significance of color"""
        r, g, b = color
        
        # Basic cultural color associations (expand with cultural database)
        cultural_colors = {
            'red': {'cultures': ['chinese', 'indian'], 'significance': 'celebration'},
            'saffron': {'cultures': ['indian', 'buddhist'], 'significance': 'sacred'},
            'white': {'cultures': ['many'], 'significance': 'purity'},
            'gold': {'cultures': ['many'], 'significance': 'prosperity'}
        }
        
        # Simple color matching (in production, use advanced color space analysis)
        if r > 200 and g < 100 and b < 100:  # Red
            return cultural_colors.get('red', {})
        elif r > 200 and g > 150 and b < 100:  # Saffron/Orange
            return cultural_colors.get('saffron', {})
        elif r > 200 and g > 200 and b > 200:  # White
            return cultural_colors.get('white', {})
        else:
            return {}
    
    def _analyze_patterns(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image for cultural patterns and motifs"""
        # Placeholder for advanced pattern recognition
        # In production, implement CNN-based pattern detection
        return {
            'geometric_patterns': 0.0,
            'floral_motifs': 0.0,
            'traditional_symbols': 0.0,
            'sacred_geometry': 0.0
        }
    
    def _analyze_garment_shapes(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze garment shapes for cultural identification"""
        # Placeholder for garment shape analysis
        # In production, use pose detection and garment segmentation
        return {
            'silhouette_type': 'unknown',
            'garment_structure': 'unknown',
            'cultural_cut': 'unknown'
        }
    
    def _analyze_textures(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze fabric textures for cultural identification"""
        # Placeholder for texture analysis
        # In production, use texture classification models
        return {
            'fabric_type': 'unknown',
            'weave_pattern': 'unknown',
            'cultural_textile': 'unknown'
        }
    
    def _match_cultural_database(self, cultural_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match analysis results against cultural database"""
        try:
            # Query cultural database with analysis results
            potential_matches = self.cultural_database.find_matches(cultural_analysis)
            
            # Score and filter matches
            validated_matches = []
            for match in potential_matches:
                match_score = self._calculate_match_score(cultural_analysis, match)
                if match_score > 0.6:  # Confidence threshold
                    match['confidence'] = match_score
                    validated_matches.append(match)
            
            # Sort by confidence
            validated_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            return validated_matches[:10]  # Top 10 matches
            
        except Exception as e:
            self.logger.error(f"Cultural database matching failed: {str(e)}")
            return []
    
    def _calculate_match_score(self, analysis: Dict[str, Any], cultural_item: Dict[str, Any]) -> float:
        """Calculate match score between analysis and cultural item"""
        try:
            # Placeholder scoring algorithm
            # In production, implement sophisticated matching
            base_score = 0.7  # Base confidence for any match
            
            # Color matching bonus
            if analysis.get('colors') and cultural_item.get('typical_colors'):
                base_score += 0.1
            
            # Pattern matching bonus
            if analysis.get('patterns') and cultural_item.get('traditional_patterns'):
                base_score += 0.1
            
            return min(base_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Match score calculation failed: {str(e)}")
            return 0.0
    
    def _assess_appropriation_risk(self, recognized_garments: List[Dict[str, Any]]) -> float:
        """Assess cultural appropriation risk"""
        try:
            if not recognized_garments:
                return 0.0
            
            risk_scores = []
            for garment in recognized_garments:
                sacred_level = garment.get('sacred_level', 'low')
                cultural_significance = garment.get('cultural_significance', 'low')
                
                # Calculate risk based on significance and sacred level
                if sacred_level in ['extremely_sacred', 'sacred']:
                    risk_score = 0.9
                elif sacred_level == 'high':
                    risk_score = 0.7
                elif cultural_significance == 'high':
                    risk_score = 0.5
                else:
                    risk_score = 0.2
                
                risk_scores.append(risk_score)
            
            # Return maximum risk (most conservative approach)
            return max(risk_scores)
            
        except Exception as e:
            self.logger.error(f"Appropriation risk assessment failed: {str(e)}")
            return 0.5  # Moderate risk as fallback
    
    def _assess_cultural_significance(self, recognized_garments: List[Dict[str, Any]]) -> str:
        """Assess overall cultural significance level"""
        if not recognized_garments:
            return 'none'
        
        significance_levels = [item.get('cultural_significance', 'low') for item in recognized_garments]
        
        if 'extremely_sacred' in significance_levels:
            return 'extremely_sacred'
        elif 'sacred' in significance_levels:
            return 'sacred'
        elif 'high' in significance_levels:
            return 'high'
        elif 'medium' in significance_levels:
            return 'medium'
        else:
            return 'low'
    
    def _extract_cultural_contexts(self, recognized_garments: List[Dict[str, Any]]) -> List[CulturalContext]:
        """Extract cultural contexts from recognized garments"""
        contexts = []
        for garment in recognized_garments:
            context = CulturalContext(
                culture=garment.get('culture', 'unknown'),
                region=garment.get('region', 'unknown'),
                significance_level=garment.get('cultural_significance', 'low'),
                sacred_level=garment.get('sacred_level', 'low'),
                appropriate_contexts=garment.get('appropriate_contexts', []),
                inappropriate_contexts=garment.get('inappropriate_contexts', [])
            )
            contexts.append(context)
        
        return contexts
    
    def calculate_appropriateness_score(self, cultural_item: Dict[str, Any], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate cultural appropriateness score for usage context
        
        Args:
            cultural_item: Cultural item information
            context: Usage context (intent, audience, commercial use, etc.)
            
        Returns:
            Comprehensive appropriateness assessment
        """
        try:
            # Extract context information
            usage_intent = context.get('usage_intent', 'unknown')
            target_audience = context.get('target_audience', 'unknown')
            commercial_use = context.get('commercial_use', False)
            cultural_consultation = context.get('cultural_consultation', False)
            
            # Base appropriateness calculation
            base_score = 1.0
            
            # Reduce score based on sacred level
            sacred_level = cultural_item.get('sacred_level', 'low')
            if sacred_level == 'extremely_sacred':
                base_score -= 0.8
            elif sacred_level == 'sacred':
                base_score -= 0.6
            elif sacred_level == 'high':
                base_score -= 0.3
            
            # Reduce score for commercial use without consultation
            if commercial_use and not cultural_consultation:
                base_score -= 0.4
            
            # Reduce score for inappropriate contexts
            inappropriate_contexts = cultural_item.get('inappropriate_contexts', [])
            if usage_intent in inappropriate_contexts:
                base_score -= 0.5
            
            # Bonus for appropriate contexts
            appropriate_contexts = cultural_item.get('appropriate_contexts', [])
            if usage_intent in appropriate_contexts:
                base_score += 0.1
            
            # Bonus for cultural consultation
            if cultural_consultation:
                base_score += 0.2
            
            # Ensure score is in valid range
            overall_score = max(0.0, min(1.0, base_score))
            
            # Determine risk level
            if overall_score >= 0.9:
                risk_level = 'low'
            elif overall_score >= 0.7:
                risk_level = 'medium'
            elif overall_score >= 0.5:
                risk_level = 'high'
            else:
                risk_level = 'critical'
            
            # Generate contextual recommendations
            recommendations = self._generate_appropriateness_recommendations(
                cultural_item, context, overall_score
            )
            
            result = {
                'overall_score': overall_score,
                'cultural_respect_score': self._calculate_respect_score(cultural_item, context),
                'contextual_appropriateness': self._calculate_contextual_score(cultural_item, context),
                'appropriation_risk_level': risk_level,
                'recommendations': recommendations,
                'expert_consultation_required': overall_score < 0.7 or sacred_level in ['sacred', 'extremely_sacred']
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Appropriateness score calculation failed: {str(e)}")
            return {'overall_score': 0.0, 'error': str(e)}
    
    def _calculate_respect_score(self, cultural_item: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate cultural respect score"""
        respect_score = 1.0
        
        # Check for cultural consultation
        if not context.get('cultural_consultation', False):
            respect_score -= 0.3
        
        # Check for educational component
        if context.get('educational_component', False):
            respect_score += 0.1
        
        # Check for profit from sacred items
        if cultural_item.get('sacred_level') in ['sacred', 'extremely_sacred'] and context.get('commercial_use', False):
            respect_score -= 0.5
        
        return max(0.0, min(1.0, respect_score))
    
    def _calculate_contextual_score(self, cultural_item: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate contextual appropriateness score"""
        contextual_score = 1.0
        
        usage_intent = context.get('usage_intent', 'unknown')
        appropriate_contexts = cultural_item.get('appropriate_contexts', [])
        inappropriate_contexts = cultural_item.get('inappropriate_contexts', [])
        
        if usage_intent in inappropriate_contexts:
            contextual_score = 0.0
        elif usage_intent in appropriate_contexts:
            contextual_score = 1.0
        else:
            contextual_score = 0.5  # Neutral context
        
        return contextual_score
    
    def _generate_appropriateness_recommendations(self, cultural_item: Dict[str, Any], 
                                                context: Dict[str, Any], score: float) -> List[str]:
        """Generate appropriateness recommendations"""
        recommendations = []
        
        if score < 0.3:
            recommendations.extend([
                "Usage not recommended - high cultural appropriation risk",
                "Consider alternative designs that do not use sacred cultural elements",
                "Seek guidance from cultural representatives"
            ])
        elif score < 0.7:
            recommendations.extend([
                "Proceed with caution - cultural consultation strongly recommended",
                "Ensure proper cultural context and respect",
                "Consider adding educational component about cultural significance"
            ])
        
        # Sacred item recommendations
        if cultural_item.get('sacred_level') in ['sacred', 'extremely_sacred']:
            recommendations.extend([
                "Sacred cultural item - expert consultation required",
                "Consider whether usage is appropriate for non-cultural members",
                "Ensure any usage is respectful and contextually appropriate"
            ])
        
        # Commercial use recommendations
        if context.get('commercial_use', False):
            recommendations.extend([
                "Commercial use detected - ensure cultural consultation",
                "Consider sharing profits with cultural communities",
                "Add cultural attribution and context"
            ])
        
        return recommendations
    
    def assess_sacred_content(self, cultural_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess sacred content and determine protection requirements
        
        Args:
            cultural_item: Cultural item to assess for sacred content
            
        Returns:
            Sacred content assessment with protection recommendations
        """
        try:
            sacred_level = cultural_item.get('sacred_level', 'low')
            cultural_significance = cultural_item.get('cultural_significance', 'low')
            
            # Determine if item is sacred
            is_sacred = sacred_level in ['high', 'sacred', 'extremely_sacred']
            
            # Determine protection level
            if sacred_level == 'extremely_sacred':
                protection_level = 'maximum'
            elif sacred_level == 'sacred':
                protection_level = 'high'
            elif sacred_level == 'high':
                protection_level = 'medium'
            else:
                protection_level = 'standard'
            
            # Generate usage restrictions
            usage_restrictions = self._generate_usage_restrictions(sacred_level, cultural_significance)
            
            # Determine if cultural consultation is required
            consultation_required = is_sacred or cultural_significance == 'high'
            
            result = {
                'is_sacred': is_sacred,
                'protection_level': protection_level,
                'usage_restrictions': usage_restrictions,
                'cultural_consultation_required': consultation_required,
                'sacred_level': sacred_level,
                'recommended_alternatives': self._suggest_alternatives(cultural_item)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sacred content assessment failed: {str(e)}")
            return {'is_sacred': False, 'error': str(e)}
    
    def _generate_usage_restrictions(self, sacred_level: str, significance: str) -> List[str]:
        """Generate usage restrictions based on sacred level"""
        restrictions = []
        
        if sacred_level == 'extremely_sacred':
            restrictions.extend([
                "No commercial use permitted",
                "No fashion adaptation allowed",
                "Cultural community approval required",
                "Educational use only with proper context"
            ])
        elif sacred_level == 'sacred':
            restrictions.extend([
                "Commercial use requires cultural consultation",
                "No costume or casual use",
                "Proper cultural context required"
            ])
        elif sacred_level == 'high' or significance == 'high':
            restrictions.extend([
                "Cultural consultation recommended",
                "Respectful usage context required",
                "No stereotypical representation"
            ])
        
        return restrictions
    
    def _suggest_alternatives(self, cultural_item: Dict[str, Any]) -> List[str]:
        """Suggest cultural alternatives that are more appropriate"""
        alternatives = []
        
        item_name = cultural_item.get('name', 'unknown')
        culture = cultural_item.get('culture', 'unknown')
        
        # Generic alternatives for sacred items
        if cultural_item.get('sacred_level') in ['sacred', 'extremely_sacred']:
            alternatives.extend([
                f"Modern interpretations inspired by {culture} aesthetics",
                "Contemporary designs with cultural appreciation attribution",
                "Fusion styles that honor without appropriating"
            ])
        
        # Specific alternatives based on item type
        if 'headdress' in item_name.lower():
            alternatives.extend([
                "Floral headpieces with natural elements",
                "Contemporary hair accessories with geometric patterns",
                "Modern headwear inspired by nature"
            ])
        elif 'robe' in item_name.lower():
            alternatives.extend([
                "Contemporary kimono-style jackets",
                "Modern wrap garments with cultural appreciation",
                "Inspired silhouettes with original patterns"
            ])
        
        return alternatives
    
    def validate_garment_context_combination(self, garment: str, context: str) -> Dict[str, Any]:
        """
        Validate appropriateness of garment-context combination
        
        Args:
            garment: Cultural garment name/type
            context: Usage context (e.g., 'tea_ceremony', 'halloween_party')
            
        Returns:
            Validation results with appropriateness assessment
        """
        try:
            # Get garment information from database
            garment_info = self.cultural_database.get_item_info(garment)
            
            if not garment_info:
                return {
                    'appropriateness_level': 'unknown',
                    'cultural_sensitivity_score': 0.5,
                    'recommendation': 'insufficient_cultural_data',
                    'alternative_suggestions': []
                }
            
            # Get context appropriateness
            appropriate_contexts = garment_info.get('appropriate_contexts', [])
            inappropriate_contexts = garment_info.get('inappropriate_contexts', [])
            
            # Calculate appropriateness
            if context in inappropriate_contexts:
                appropriateness_level = 'inappropriate'
                sensitivity_score = 0.1
            elif context in appropriate_contexts:
                appropriateness_level = 'appropriate'
                sensitivity_score = 0.95
            else:
                # Context not explicitly listed - analyze based on cultural significance
                sacred_level = garment_info.get('sacred_level', 'low')
                if sacred_level in ['sacred', 'extremely_sacred']:
                    appropriateness_level = 'questionable'
                    sensitivity_score = 0.3
                else:
                    appropriateness_level = 'neutral'
                    sensitivity_score = 0.7
            
            # Generate recommendation
            recommendation = self._generate_context_recommendation(
                appropriateness_level, garment_info, context
            )
            
            # Suggest alternatives if inappropriate
            alternatives = []
            if appropriateness_level in ['inappropriate', 'questionable']:
                alternatives = self._suggest_context_alternatives(garment_info, context)
            
            result = {
                'appropriateness_level': appropriateness_level,
                'cultural_sensitivity_score': sensitivity_score,
                'recommendation': recommendation,
                'alternative_suggestions': alternatives,
                'cultural_context': garment_info.get('culture', 'unknown'),
                'sacred_level': garment_info.get('sacred_level', 'low')
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Garment-context validation failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_context_recommendation(self, appropriateness: str, garment_info: Dict[str, Any], 
                                       context: str) -> str:
        """Generate context-specific recommendation"""
        if appropriateness == 'inappropriate':
            return f"Not recommended: {context} is inappropriate for {garment_info.get('name', 'this garment')}"
        elif appropriateness == 'questionable':
            return f"Proceed with caution: Cultural consultation recommended for {context} usage"
        elif appropriateness == 'appropriate':
            return f"Appropriate usage: {context} is suitable for {garment_info.get('name', 'this garment')}"
        else:
            return f"Neutral context: Consider cultural significance before proceeding"
    
    def _suggest_context_alternatives(self, garment_info: Dict[str, Any], context: str) -> List[str]:
        """Suggest alternative contexts or garments"""
        alternatives = []
        
        # Suggest appropriate contexts for this garment
        appropriate_contexts = garment_info.get('appropriate_contexts', [])
        if appropriate_contexts:
            alternatives.extend([
                f"Consider using for {ctx} instead" for ctx in appropriate_contexts[:3]
            ])
        
        # Suggest alternative garments for this context
        culture = garment_info.get('culture', 'unknown')
        alternatives.extend([
            f"Consider modern {culture}-inspired designs",
            f"Use contemporary fashion with {culture} aesthetic elements",
            "Explore fusion styles that honor without appropriating"
        ])
        
        return alternatives
    
    def queue_for_expert_review(self, cultural_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Queue cultural content for expert review
        
        Args:
            cultural_content: Cultural content requiring expert assessment
            
        Returns:
            Review queue information and interim recommendations
        """
        try:
            # Assess review priority
            appropriation_risk = cultural_content.get('appropriation_risk', 0.0)
            sacred_level = cultural_content.get('sacred_level', 'low')
            
            if appropriation_risk > 0.8 or sacred_level == 'extremely_sacred':
                priority = 'critical'
                estimated_time = '4-6 hours'
            elif appropriation_risk > 0.6 or sacred_level == 'sacred':
                priority = 'high'
                estimated_time = '1-2 days'
            elif appropriation_risk > 0.4 or sacred_level == 'high':
                priority = 'medium'
                estimated_time = '3-5 days'
            else:
                priority = 'low'
                estimated_time = '1 week'
            
            # Determine required expert categories
            expert_categories = self._determine_expert_categories(cultural_content)
            
            # Generate interim recommendation
            interim_recommendation = self._generate_interim_recommendation(
                appropriation_risk, sacred_level
            )
            
            # Add to review queue
            review_item = {
                'content': cultural_content,
                'priority': priority,
                'expert_categories': expert_categories,
                'submitted_at': datetime.now(),
                'estimated_review_time': estimated_time,
                'status': 'queued'
            }
            
            self.expert_review_queue.append(review_item)
            
            result = {
                'queued_for_review': True,
                'review_priority': priority,
                'estimated_review_time': estimated_time,
                'interim_recommendation': interim_recommendation,
                'expert_categories': expert_categories,
                'queue_position': len(self.expert_review_queue)
            }
            
            self.logger.info(f"Cultural content queued for expert review (priority: {priority})")
            return result
            
        except Exception as e:
            self.logger.error(f"Expert review queueing failed: {str(e)}")
            return {'queued_for_review': False, 'error': str(e)}
    
    def _determine_expert_categories(self, cultural_content: Dict[str, Any]) -> List[str]:
        """Determine required expert categories for review"""
        expert_categories = []
        
        culture = cultural_content.get('culture', 'unknown')
        sacred_level = cultural_content.get('sacred_level', 'low')
        
        # Cultural anthropologist for all reviews
        expert_categories.append('cultural_anthropologist')
        
        # Culture-specific experts
        if culture != 'unknown':
            expert_categories.append(f'{culture}_cultural_expert')
        
        # Religious/spiritual expert for sacred items
        if sacred_level in ['sacred', 'extremely_sacred']:
            expert_categories.append('religious_studies_expert')
        
        # Fashion historian for garment-specific issues
        if cultural_content.get('garment'):
            expert_categories.append('fashion_historian')
        
        return expert_categories
    
    def _generate_interim_recommendation(self, risk: float, sacred_level: str) -> str:
        """Generate interim recommendation while awaiting expert review"""
        if risk > 0.8 or sacred_level == 'extremely_sacred':
            return "Block usage pending expert review"
        elif risk > 0.6 or sacred_level == 'sacred':
            return "Proceed with extreme caution - expert approval required"
        elif risk > 0.4:
            return "Enhanced monitoring recommended pending review"
        else:
            return "Standard monitoring while awaiting expert guidance"
    
    async def monitor_cultural_sensitivity_realtime(self, image: Image.Image,
                                                  transformation_metadata: Dict[str, Any]) -> CulturalSensitivityResult:
        """
        Real-time cultural sensitivity monitoring
        
        Args:
            image: Input image for cultural analysis
            transformation_metadata: Transformation context and settings
            
        Returns:
            Real-time cultural sensitivity assessment
        """
        try:
            # Recognize cultural garments
            recognition_result = self.recognize_cultural_garments(image)
            
            # Calculate cultural appropriateness
            appropriateness_scores = {}
            for item in recognition_result.get('garments_detected', []):
                context = {
                    'usage_intent': transformation_metadata.get('style', 'fashion_photography'),
                    'commercial_use': True,
                    'cultural_consultation': False
                }
                
                appropriateness = self.calculate_appropriateness_score(item, context)
                appropriateness_scores[item.get('name', 'unknown')] = appropriateness['overall_score']
            
            # Calculate overall cultural sensitivity score
            if appropriateness_scores:
                cultural_sensitivity_score = min(appropriateness_scores.values())
            else:
                cultural_sensitivity_score = 1.0  # No cultural items detected
            
            # Determine if expert review is required
            expert_review_required = (
                cultural_sensitivity_score < self.sensitivity_thresholds['expert_review_threshold'] or
                any(item.get('sacred_level') in ['sacred', 'extremely_sacred'] 
                    for item in recognition_result.get('garments_detected', []))
            )
            
            # Generate recommendations
            recommendations = self._generate_realtime_recommendations(
                recognition_result, appropriateness_scores, cultural_sensitivity_score
            )
            
            # Create result
            result = CulturalSensitivityResult(
                cultural_sensitivity_score=cultural_sensitivity_score,
                cultural_items_detected=recognition_result.get('garments_detected', []),
                appropriateness_assessment=appropriateness_scores,
                appropriation_risk=recognition_result.get('appropriation_risk', 0.0),
                recommendations=recommendations,
                expert_review_required=expert_review_required,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Real-time cultural monitoring failed: {str(e)}")
            return self._create_fallback_sensitivity_result()
    
    def _generate_realtime_recommendations(self, recognition_result: Dict[str, Any],
                                         appropriateness_scores: Dict[str, float],
                                         overall_score: float) -> List[str]:
        """Generate real-time cultural sensitivity recommendations"""
        recommendations = []
        
        if overall_score < 0.3:
            recommendations.extend([
                "High cultural appropriation risk detected",
                "Block transformation pending cultural review",
                "Consider alternative styling without cultural elements"
            ])
        elif overall_score < 0.7:
            recommendations.extend([
                "Cultural sensitivity concerns detected",
                "Enhanced monitoring recommended",
                "Consider cultural consultation before proceeding"
            ])
        
        # Item-specific recommendations
        for item in recognition_result.get('garments_detected', []):
            if item.get('sacred_level') in ['sacred', 'extremely_sacred']:
                recommendations.append(f"Sacred item detected: {item.get('name', 'unknown')} - expert review required")
        
        return recommendations
    
    def _create_fallback_recognition_result(self) -> Dict[str, Any]:
        """Create fallback recognition result on error"""
        return {
            'garments_detected': [],
            'cultural_significance': 'unknown',
            'confidence': 0.0,
            'appropriation_risk': 0.5,  # Moderate risk as conservative fallback
            'cultural_contexts': [],
            'error': True
        }
    
    def _create_fallback_sensitivity_result(self) -> CulturalSensitivityResult:
        """Create fallback sensitivity result on error"""
        return CulturalSensitivityResult(
            cultural_sensitivity_score=0.5,  # Moderate score when uncertain
            cultural_items_detected=[],
            appropriateness_assessment={},
            appropriation_risk=0.5,
            recommendations=['Manual cultural review required due to analysis error'],
            expert_review_required=True,  # Conservative approach
            timestamp=datetime.now()
        )
    
    def get_cultural_database_stats(self) -> Dict[str, Any]:
        """Get cultural database statistics"""
        return self.cultural_database.get_database_stats()
    
    def get_cultural_items_by_culture(self, culture: str) -> List[Dict[str, Any]]:
        """Get cultural items for specific culture"""
        return self.cultural_database.search_by_culture(culture)
    
    def get_regional_cultural_info(self, garment: str, region: str) -> Dict[str, Any]:
        """Get regional cultural information for garment"""
        return self.cultural_database.get_regional_info(garment, region)
    
    def provide_cultural_education(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide cultural education and alternatives for problematic requests
        
        Args:
            request: Problematic cultural request requiring education
            
        Returns:
            Educational content and respectful alternatives
        """
        try:
            desired_style = request.get('desired_style', 'unknown')
            garment_interest = request.get('garment_interest', 'unknown')
            
            # Get cultural information
            cultural_info = self.cultural_database.get_educational_info(garment_interest)
            
            # Generate educational content
            education_result = {
                'cultural_background': cultural_info.get('background', 'Cultural background information not available'),
                'significance_explanation': cultural_info.get('significance', 'Cultural significance information not available'),
                'why_problematic': self._explain_why_problematic(cultural_info, request),
                'respectful_alternatives': self._generate_respectful_alternatives(cultural_info, request),
                'cultural_appreciation_guidance': self._provide_appreciation_guidance(cultural_info)
            }
            
            return education_result
            
        except Exception as e:
            self.logger.error(f"Cultural education provision failed: {str(e)}")
            return {'error': str(e)}
    
    def _explain_why_problematic(self, cultural_info: Dict[str, Any], request: Dict[str, Any]) -> str:
        """Explain why the request is culturally problematic"""
        sacred_level = cultural_info.get('sacred_level', 'low')
        usage = request.get('usage', 'unknown')
        
        explanations = []
        
        if sacred_level in ['sacred', 'extremely_sacred']:
            explanations.append("This item has sacred religious or spiritual significance")
        
        if usage == 'fashion_photoshoot' and cultural_info.get('traditional_use'):
            explanations.append("Using traditional ceremonial items for fashion can trivialize their cultural importance")
        
        if not explanations:
            explanations.append("This usage may not respect the cultural context and significance of the item")
        
        return ". ".join(explanations)
    
    def _generate_respectful_alternatives(self, cultural_info: Dict[str, Any], request: Dict[str, Any]) -> List[str]:
        """Generate respectful alternatives to problematic requests"""
        alternatives = []
        
        culture = cultural_info.get('culture', 'unknown')
        
        alternatives.extend([
            f"Contemporary designs inspired by {culture} aesthetics",
            f"Modern interpretations that honor {culture} artistic traditions",
            f"Fusion styles that appreciate without appropriating {culture} elements",
            "Original designs that celebrate cultural diversity respectfully"
        ])
        
        return alternatives
    
    def _provide_appreciation_guidance(self, cultural_info: Dict[str, Any]) -> List[str]:
        """Provide guidance for cultural appreciation vs appropriation"""
        guidance = [
            "Learn about the cultural significance and history",
            "Consult with cultural representatives or experts",
            "Give proper attribution and context",
            "Avoid using sacred or ceremonial items for fashion",
            "Support cultural communities and artisans",
            "Approach with respect, humility, and willingness to learn"
        ]
        
        return guidance
    
    def validate_multicultural_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate scenes with multiple cultural elements
        
        Args:
            scene_data: Scene containing multiple cultural elements
            
        Returns:
            Multicultural scene validation results
        """
        try:
            garments = scene_data.get('garments', [])
            background = scene_data.get('background', 'neutral')
            context = scene_data.get('context', 'fashion')
            
            # Validate each cultural element individually
            individual_scores = {}
            for garment in garments:
                garment_info = self.cultural_database.get_item_info(garment)
                if garment_info:
                    context_data = {'usage_intent': context, 'commercial_use': True}
                    appropriateness = self.calculate_appropriateness_score(garment_info, context_data)
                    individual_scores[garment] = appropriateness['overall_score']
            
            # Calculate cultural harmony (how well cultures work together)
            cultural_harmony_score = self._assess_cultural_harmony(garments)
            
            # Calculate fusion appropriateness
            fusion_appropriateness = self._assess_fusion_appropriateness(scene_data)
            
            # Overall appropriateness
            if individual_scores:
                min_individual_score = min(individual_scores.values())
                overall_appropriateness = min(min_individual_score, cultural_harmony_score, fusion_appropriateness)
            else:
                overall_appropriateness = 1.0
            
            # Determine if expert consultation is needed
            expert_consultation_needed = (
                overall_appropriateness < 0.7 or
                any(score < 0.5 for score in individual_scores.values())
            )
            
            result = {
                'overall_appropriateness': overall_appropriateness,
                'individual_element_scores': individual_scores,
                'cultural_harmony_score': cultural_harmony_score,
                'fusion_appropriateness': fusion_appropriateness,
                'expert_consultation_needed': expert_consultation_needed,
                'recommendations': self._generate_multicultural_recommendations(overall_appropriateness)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multicultural scene validation failed: {str(e)}")
            return {'error': str(e)}
    
    def _assess_cultural_harmony(self, garments: List[str]) -> float:
        """Assess how well multiple cultures harmonize in one scene"""
        if len(garments) <= 1:
            return 1.0
        
        # Get cultures for each garment
        cultures = []
        for garment in garments:
            garment_info = self.cultural_database.get_item_info(garment)
            if garment_info:
                cultures.append(garment_info.get('culture', 'unknown'))
        
        # Simple harmony assessment
        unique_cultures = set(cultures)
        
        if len(unique_cultures) <= 2:
            return 0.9  # Two cultures can harmonize well
        elif len(unique_cultures) <= 3:
            return 0.7  # Three cultures require more care
        else:
            return 0.5  # Many cultures together need expert review
    
    def _assess_fusion_appropriateness(self, scene_data: Dict[str, Any]) -> float:
        """Assess appropriateness of cultural fusion approach"""
        styling_approach = scene_data.get('styling_approach', 'unknown')
        
        if styling_approach == 'respectful_fusion':
            return 0.9
        elif styling_approach == 'educational':
            return 0.95
        elif styling_approach == 'artistic':
            return 0.8
        else:
            return 0.6  # Unknown approach needs review
    
    def _generate_multicultural_recommendations(self, appropriateness: float) -> List[str]:
        """Generate recommendations for multicultural scenes"""
        recommendations = []
        
        if appropriateness < 0.5:
            recommendations.extend([
                "Multicultural combination not recommended",
                "Consider focusing on single cultural tradition",
                "Seek expert cultural consultation"
            ])
        elif appropriateness < 0.7:
            recommendations.extend([
                "Proceed with caution in multicultural styling",
                "Ensure respectful representation of each culture",
                "Add educational context about cultural fusion"
            ])
        else:
            recommendations.extend([
                "Multicultural approach appears respectful",
                "Continue with cultural sensitivity monitoring",
                "Consider adding cultural attribution"
            ])
        
        return recommendations