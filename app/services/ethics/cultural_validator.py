"""
Cultural Validator for Cultural Sensitivity Service

This module provides validation logic for cultural appropriateness,
sacred content usage, and context sensitivity assessments.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class ValidationResult:
    """Cultural validation result structure"""
    is_appropriate: bool
    confidence: float
    violations: List[str]
    recommendations: List[str]
    risk_level: str
    cultural_consultation_required: bool


class CulturalValidator:
    """
    Cultural appropriateness validator
    
    Provides validation logic for determining cultural appropriateness,
    sacred content protection, and context sensitivity.
    """
    
    def __init__(self):
        """Initialize cultural validator"""
        self.logger = logging.getLogger(__name__)
        
        # Validation rules and thresholds
        self.appropriateness_thresholds = {
            'extremely_sacred': 0.95,
            'sacred': 0.90,
            'high_significance': 0.80,
            'medium_significance': 0.70,
            'general': 0.60
        }
        
        # Risk level mappings
        self.risk_levels = {
            (0.0, 0.3): 'critical',
            (0.3, 0.5): 'high',
            (0.5, 0.7): 'medium',
            (0.7, 0.9): 'low',
            (0.9, 1.0): 'minimal'
        }
        
        # Sacred usage restrictions
        self.sacred_restrictions = {
            'extremely_sacred': [
                'no_commercial_use',
                'no_fashion_adaptation',
                'cultural_community_approval_required',
                'educational_use_only'
            ],
            'sacred': [
                'cultural_consultation_required',
                'no_costume_use',
                'proper_cultural_context_required'
            ],
            'high': [
                'cultural_consultation_recommended',
                'respectful_usage_required',
                'no_stereotypical_representation'
            ]
        }
        
        # Context appropriateness mappings
        self.context_mappings = self._initialize_context_mappings()
        
        self.logger.info("Cultural Validator initialized")
    
    def _initialize_context_mappings(self) -> Dict[str, Dict[str, float]]:
        """Initialize context appropriateness mappings"""
        return {
            'ceremonial': {
                'cultural_ceremony': 1.0,
                'religious_ceremony': 1.0,
                'wedding': 0.9,
                'graduation': 0.8,
                'formal_event': 0.7,
                'fashion_show': 0.3,
                'costume_party': 0.1,
                'halloween': 0.0
            },
            'traditional': {
                'cultural_celebration': 1.0,
                'cultural_education': 0.9,
                'artistic_performance': 0.8,
                'formal_event': 0.7,
                'fashion_photography': 0.4,
                'casual_wear': 0.2,
                'costume': 0.0
            },
            'religious': {
                'religious_observance': 1.0,
                'religious_ceremony': 1.0,
                'cultural_education': 0.8,
                'respectful_artistic_expression': 0.6,
                'fashion': 0.1,
                'costume': 0.0,
                'commercial_use': 0.2
            },
            'everyday': {
                'daily_wear': 1.0,
                'casual_social': 0.9,
                'work_appropriate': 0.8,
                'formal_event': 0.7,
                'fashion': 0.6,
                'artistic': 0.5
            }
        }
    
    def validate_appropriateness(self, cultural_combo: Dict[str, Any]) -> ValidationResult:
        """
        Validate cultural appropriateness of garment-context combination
        
        Args:
            cultural_combo: Dictionary containing garment, context, and usage info
            
        Returns:
            Comprehensive validation result
        """
        try:
            garment = cultural_combo.get('garment', 'unknown')
            context = cultural_combo.get('context', 'unknown')
            culture_source = cultural_combo.get('culture_source', 'unknown')
            usage_culture = cultural_combo.get('usage_culture', 'unknown')
            commercial_use = cultural_combo.get('commercial_use', False)
            cultural_consultation = cultural_combo.get('cultural_consultation', False)
            
            # Calculate base appropriateness score
            base_score = self._calculate_base_appropriateness(cultural_combo)
            
            # Apply context-specific adjustments
            context_score = self._assess_context_appropriateness(garment, context)
            
            # Apply cultural source considerations
            cultural_source_score = self._assess_cultural_source_appropriateness(
                culture_source, usage_culture
            )
            
            # Apply commercial use considerations
            commercial_adjustment = self._assess_commercial_use_impact(
                cultural_combo, commercial_use, cultural_consultation
            )
            
            # Calculate final appropriateness score
            final_score = (
                base_score * 0.4 +
                context_score * 0.3 +
                cultural_source_score * 0.2 +
                commercial_adjustment * 0.1
            )
            
            # Determine if appropriate
            is_appropriate = final_score >= 0.7
            
            # Assess confidence
            confidence = self._calculate_validation_confidence(cultural_combo, final_score)
            
            # Identify violations
            violations = self._identify_violations(cultural_combo, final_score)
            
            # Generate recommendations
            recommendations = self._generate_validation_recommendations(
                cultural_combo, final_score, violations
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(final_score)
            
            # Determine if cultural consultation is required
            consultation_required = self._requires_cultural_consultation(
                cultural_combo, final_score
            )
            
            result = ValidationResult(
                is_appropriate=is_appropriate,
                confidence=confidence,
                violations=violations,
                recommendations=recommendations,
                risk_level=risk_level,
                cultural_consultation_required=consultation_required
            )
            
            self.logger.info(f"Cultural validation completed: {risk_level} risk, appropriate: {is_appropriate}")
            return result
            
        except Exception as e:
            self.logger.error(f"Cultural validation failed: {str(e)}")
            return self._create_fallback_validation_result()
    
    def _calculate_base_appropriateness(self, cultural_combo: Dict[str, Any]) -> float:
        """Calculate base appropriateness score"""
        try:
            # Extract cultural significance information
            sacred_level = cultural_combo.get('sacred_level', 'low')
            cultural_significance = cultural_combo.get('cultural_significance', 'low')
            
            # Base score depends on significance
            if sacred_level == 'extremely_sacred':
                base_score = 0.2  # Very restrictive
            elif sacred_level == 'sacred':
                base_score = 0.4
            elif sacred_level == 'high' or cultural_significance == 'high':
                base_score = 0.6
            elif cultural_significance == 'medium':
                base_score = 0.8
            else:
                base_score = 0.9  # Low significance items more permissive
            
            return base_score
            
        except Exception as e:
            self.logger.error(f"Base appropriateness calculation failed: {str(e)}")
            return 0.5
    
    def _assess_context_appropriateness(self, garment: str, context: str) -> float:
        """Assess context appropriateness"""
        try:
            # Determine garment category
            garment_category = self._categorize_garment(garment)
            
            # Get context mappings for this category
            context_scores = self.context_mappings.get(garment_category, {})
            
            # Find matching context
            context_score = 0.5  # Default neutral score
            for ctx_pattern, score in context_scores.items():
                if self._context_matches(context, ctx_pattern):
                    context_score = score
                    break
            
            return context_score
            
        except Exception as e:
            self.logger.error(f"Context appropriateness assessment failed: {str(e)}")
            return 0.5
    
    def _categorize_garment(self, garment: str) -> str:
        """Categorize garment for context mapping"""
        garment_lower = garment.lower()
        
        # Ceremonial garments
        ceremonial_keywords = ['headdress', 'ceremonial', 'ritual', 'sacred', 'jingle_dress']
        if any(keyword in garment_lower for keyword in ceremonial_keywords):
            return 'ceremonial'
        
        # Religious garments
        religious_keywords = ['hijab', 'habit', 'cassock', 'religious', 'prayer']
        if any(keyword in garment_lower for keyword in religious_keywords):
            return 'religious'
        
        # Traditional but not ceremonial
        traditional_keywords = ['kimono', 'sari', 'hanbok', 'traditional', 'cultural']
        if any(keyword in garment_lower for keyword in traditional_keywords):
            return 'traditional'
        
        # Default to everyday
        return 'everyday'
    
    def _context_matches(self, context: str, pattern: str) -> bool:
        """Check if context matches pattern"""
        context_lower = context.lower()
        pattern_lower = pattern.lower()
        
        # Direct match
        if pattern_lower in context_lower:
            return True
        
        # Synonym matching
        context_synonyms = {
            'halloween': ['costume_party', 'halloween_party', 'costume'],
            'fashion': ['fashion_show', 'fashion_photography', 'modeling'],
            'cultural_ceremony': ['cultural_celebration', 'cultural_event'],
            'religious_ceremony': ['religious_service', 'worship', 'prayer']
        }
        
        for main_context, synonyms in context_synonyms.items():
            if main_context == pattern_lower and any(syn in context_lower for syn in synonyms):
                return True
        
        return False
    
    def _assess_cultural_source_appropriateness(self, culture_source: str, usage_culture: str) -> float:
        """Assess appropriateness based on cultural source vs usage culture"""
        try:
            if culture_source == 'unknown' or usage_culture == 'unknown':
                return 0.7  # Neutral when information unavailable
            
            # Same culture usage
            if culture_source.lower() == usage_culture.lower():
                return 1.0
            
            # Global/universal cultures
            global_cultures = ['global', 'universal', 'western', 'international']
            if culture_source.lower() in global_cultures:
                return 0.9
            
            # Cross-cultural usage requires more consideration
            return 0.6
            
        except Exception as e:
            self.logger.error(f"Cultural source assessment failed: {str(e)}")
            return 0.7
    
    def _assess_commercial_use_impact(self, cultural_combo: Dict[str, Any], 
                                    commercial_use: bool, consultation: bool) -> float:
        """Assess impact of commercial use"""
        try:
            if not commercial_use:
                return 1.0  # No commercial impact
            
            sacred_level = cultural_combo.get('sacred_level', 'low')
            
            # Sacred items heavily penalized for commercial use
            if sacred_level in ['extremely_sacred', 'sacred']:
                if consultation:
                    return 0.3  # Still problematic but consultation helps
                else:
                    return 0.1  # Very problematic
            
            # High significance items
            elif sacred_level == 'high':
                if consultation:
                    return 0.7
                else:
                    return 0.4
            
            # Medium/low significance
            else:
                if consultation:
                    return 0.9
                else:
                    return 0.7
            
        except Exception as e:
            self.logger.error(f"Commercial use assessment failed: {str(e)}")
            return 0.5
    
    def _calculate_validation_confidence(self, cultural_combo: Dict[str, Any], score: float) -> float:
        """Calculate confidence in validation result"""
        try:
            confidence_factors = []
            
            # Information completeness
            required_fields = ['garment', 'context', 'sacred_level']
            completeness = sum(1 for field in required_fields if cultural_combo.get(field))
            confidence_factors.append(completeness / len(required_fields))
            
            # Score extremes are more confident
            if score < 0.2 or score > 0.8:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.7)
            
            # Cultural consultation information available
            if 'cultural_consultation' in cultural_combo:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.7)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.5
    
    def _identify_violations(self, cultural_combo: Dict[str, Any], score: float) -> List[str]:
        """Identify specific cultural violations"""
        violations = []
        
        try:
            sacred_level = cultural_combo.get('sacred_level', 'low')
            context = cultural_combo.get('context', 'unknown')
            commercial_use = cultural_combo.get('commercial_use', False)
            cultural_consultation = cultural_combo.get('cultural_consultation', False)
            
            # Sacred item violations
            if sacred_level in ['extremely_sacred', 'sacred']:
                if 'costume' in context.lower() or 'halloween' in context.lower():
                    violations.append('sacred_item_used_as_costume')
                
                if commercial_use and not cultural_consultation:
                    violations.append('commercial_use_of_sacred_item_without_consultation')
            
            # Context violations
            inappropriate_contexts = ['costume', 'halloween', 'stereotypical']
            if any(ctx in context.lower() for ctx in inappropriate_contexts):
                violations.append('inappropriate_context_usage')
            
            # Cultural consultation violations
            if score < 0.5 and not cultural_consultation:
                violations.append('cultural_consultation_required_but_not_obtained')
            
            # General appropriateness violations
            if score < 0.3:
                violations.append('overall_cultural_appropriateness_violation')
            
        except Exception as e:
            self.logger.error(f"Violation identification failed: {str(e)}")
        
        return violations
    
    def _generate_validation_recommendations(self, cultural_combo: Dict[str, Any], 
                                           score: float, violations: List[str]) -> List[str]:
        """Generate validation recommendations"""
        recommendations = []
        
        try:
            # Score-based recommendations
            if score < 0.3:
                recommendations.extend([
                    "Usage not recommended - high cultural appropriation risk",
                    "Consider alternative designs without cultural elements",
                    "Seek guidance from cultural representatives"
                ])
            elif score < 0.7:
                recommendations.extend([
                    "Proceed with caution - cultural consultation recommended",
                    "Ensure proper cultural context and attribution",
                    "Consider educational component about cultural significance"
                ])
            
            # Violation-specific recommendations
            if 'sacred_item_used_as_costume' in violations:
                recommendations.append("Sacred items should not be used as costumes")
            
            if 'commercial_use_of_sacred_item_without_consultation' in violations:
                recommendations.extend([
                    "Cultural consultation required for commercial use",
                    "Consider profit-sharing with cultural communities"
                ])
            
            if 'cultural_consultation_required_but_not_obtained' in violations:
                recommendations.append("Obtain cultural consultation before proceeding")
            
            # Sacred level recommendations
            sacred_level = cultural_combo.get('sacred_level', 'low')
            if sacred_level in ['extremely_sacred', 'sacred']:
                recommendations.extend([
                    "Consider whether usage is appropriate for non-cultural members",
                    "Ensure any usage is respectful and contextually appropriate"
                ])
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
        
        return recommendations
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score"""
        for (min_score, max_score), risk_level in self.risk_levels.items():
            if min_score <= score <= max_score:
                return risk_level
        
        return 'unknown'
    
    def _requires_cultural_consultation(self, cultural_combo: Dict[str, Any], score: float) -> bool:
        """Determine if cultural consultation is required"""
        try:
            sacred_level = cultural_combo.get('sacred_level', 'low')
            commercial_use = cultural_combo.get('commercial_use', False)
            
            # Always required for sacred items
            if sacred_level in ['extremely_sacred', 'sacred']:
                return True
            
            # Required for high significance + commercial use
            if sacred_level == 'high' and commercial_use:
                return True
            
            # Required for low scores
            if score < 0.7:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Consultation requirement assessment failed: {str(e)}")
            return True  # Conservative default
    
    def validate_sacred_usage(self, sacred_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate usage of sacred cultural content
        
        Args:
            sacred_content: Sacred content information and proposed usage
            
        Returns:
            Sacred usage validation results
        """
        try:
            item = sacred_content.get('item', 'unknown')
            sacred_level = sacred_content.get('sacred_level', 'low')
            proposed_usage = sacred_content.get('proposed_usage', 'unknown')
            cultural_permission = sacred_content.get('cultural_permission', False)
            
            # Determine if usage is approved
            usage_approved = self._assess_sacred_usage_approval(
                sacred_level, proposed_usage, cultural_permission
            )
            
            # Determine consultation requirements
            consultation_required = sacred_level in ['high', 'sacred', 'extremely_sacred']
            
            # Get usage restrictions
            restrictions = self.sacred_restrictions.get(sacred_level, [])
            
            # Generate sacred-specific recommendations
            recommendations = self._generate_sacred_recommendations(
                sacred_level, proposed_usage, usage_approved
            )
            
            result = {
                'usage_approved': usage_approved,
                'requires_consultation': consultation_required,
                'cultural_permission_required': sacred_level in ['sacred', 'extremely_sacred'],
                'usage_restrictions': restrictions,
                'recommendations': recommendations,
                'sacred_level': sacred_level,
                'risk_assessment': 'critical' if not usage_approved else 'low'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sacred usage validation failed: {str(e)}")
            return {'usage_approved': False, 'error': str(e)}
    
    def _assess_sacred_usage_approval(self, sacred_level: str, usage: str, permission: bool) -> bool:
        """Assess if sacred usage should be approved"""
        try:
            # Extremely sacred - very restrictive
            if sacred_level == 'extremely_sacred':
                approved_uses = ['cultural_ceremony', 'educational_with_permission']
                return any(use in usage.lower() for use in approved_uses) and permission
            
            # Sacred - restrictive
            elif sacred_level == 'sacred':
                forbidden_uses = ['costume', 'fashion', 'commercial', 'casual']
                return not any(use in usage.lower() for use in forbidden_uses)
            
            # High significance - moderate restrictions
            elif sacred_level == 'high':
                forbidden_uses = ['costume', 'halloween', 'stereotypical']
                return not any(use in usage.lower() for use in forbidden_uses)
            
            # Medium/low - fewer restrictions
            else:
                forbidden_uses = ['offensive', 'stereotypical']
                return not any(use in usage.lower() for use in forbidden_uses)
            
        except Exception as e:
            self.logger.error(f"Sacred usage approval assessment failed: {str(e)}")
            return False  # Conservative default
    
    def _generate_sacred_recommendations(self, sacred_level: str, usage: str, approved: bool) -> List[str]:
        """Generate recommendations for sacred content usage"""
        recommendations = []
        
        if not approved:
            recommendations.extend([
                "Sacred content usage not approved",
                "Consider alternative non-sacred designs",
                "Seek proper cultural consultation"
            ])
        
        if sacred_level == 'extremely_sacred':
            recommendations.extend([
                "Extremely sacred item - use only with explicit cultural permission",
                "Educational context required if any usage permitted",
                "No commercial adaptation permitted"
            ])
        elif sacred_level == 'sacred':
            recommendations.extend([
                "Sacred item - cultural consultation required",
                "Ensure proper cultural context",
                "Avoid costume or casual usage"
            ])
        
        return recommendations
    
    def validate_context_sensitivity(self, sensitive_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate context sensitivity for cultural items
        
        Args:
            sensitive_context: Context sensitivity information
            
        Returns:
            Context sensitivity validation results
        """
        try:
            cultural_item = sensitive_context.get('cultural_item', 'unknown')
            proposed_context = sensitive_context.get('proposed_context', 'unknown')
            cultural_significance = sensitive_context.get('cultural_significance', 'low')
            
            # Assess context appropriateness
            context_appropriate = self._assess_context_sensitivity(
                cultural_item, proposed_context, cultural_significance
            )
            
            # Check for sensitivity violations
            sensitivity_violation = self._check_sensitivity_violations(
                proposed_context, cultural_significance
            )
            
            # Generate context recommendations
            recommendations = self._generate_context_recommendations(
                cultural_item, proposed_context, context_appropriate
            )
            
            result = {
                'context_appropriate': context_appropriate,
                'sensitivity_violation': sensitivity_violation,
                'cultural_significance': cultural_significance,
                'recommendations': recommendations,
                'approval_required': not context_appropriate or sensitivity_violation
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Context sensitivity validation failed: {str(e)}")
            return {'context_appropriate': False, 'error': str(e)}
    
    def _assess_context_sensitivity(self, item: str, context: str, significance: str) -> bool:
        """Assess context sensitivity"""
        try:
            # High significance items are more context-sensitive
            if significance in ['extremely_high', 'high']:
                appropriate_contexts = [
                    'cultural_ceremony', 'educational', 'respectful_artistic',
                    'cultural_celebration', 'formal_cultural_event'
                ]
                return any(ctx in context.lower() for ctx in appropriate_contexts)
            
            # Medium significance - moderate sensitivity
            elif significance == 'medium':
                inappropriate_contexts = ['costume', 'halloween', 'mockery', 'stereotypical']
                return not any(ctx in context.lower() for ctx in inappropriate_contexts)
            
            # Low significance - minimal restrictions
            else:
                very_inappropriate = ['mockery', 'offensive', 'discriminatory']
                return not any(ctx in context.lower() for ctx in very_inappropriate)
            
        except Exception as e:
            self.logger.error(f"Context sensitivity assessment failed: {str(e)}")
            return False
    
    def _check_sensitivity_violations(self, context: str, significance: str) -> bool:
        """Check for specific sensitivity violations"""
        try:
            context_lower = context.lower()
            
            # Universal violations regardless of significance
            universal_violations = ['mockery', 'offensive', 'discriminatory', 'hateful']
            if any(violation in context_lower for violation in universal_violations):
                return True
            
            # Significance-specific violations
            if significance in ['extremely_high', 'high']:
                high_significance_violations = ['costume', 'halloween', 'casual', 'trivial']
                if any(violation in context_lower for violation in high_significance_violations):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Sensitivity violation check failed: {str(e)}")
            return True  # Conservative default
    
    def _generate_context_recommendations(self, item: str, context: str, appropriate: bool) -> List[str]:
        """Generate context-specific recommendations"""
        recommendations = []
        
        if not appropriate:
            recommendations.extend([
                f"Context '{context}' not appropriate for {item}",
                "Consider more respectful cultural contexts",
                "Seek cultural guidance for appropriate usage"
            ])
        else:
            recommendations.extend([
                "Context appears culturally appropriate",
                "Continue with cultural sensitivity monitoring",
                "Ensure respectful representation"
            ])
        
        return recommendations
    
    def _create_fallback_validation_result(self) -> ValidationResult:
        """Create fallback validation result on error"""
        return ValidationResult(
            is_appropriate=False,
            confidence=0.0,
            violations=['validation_error'],
            recommendations=['Manual cultural review required due to validation error'],
            risk_level='unknown',
            cultural_consultation_required=True
        )
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get current validation rules and thresholds"""
        return {
            'appropriateness_thresholds': self.appropriateness_thresholds,
            'risk_levels': {str(k): v for k, v in self.risk_levels.items()},
            'sacred_restrictions': self.sacred_restrictions,
            'context_mappings': self.context_mappings
        }
    
    def update_validation_rules(self, new_rules: Dict[str, Any]):
        """Update validation rules and thresholds"""
        if 'appropriateness_thresholds' in new_rules:
            self.appropriateness_thresholds.update(new_rules['appropriateness_thresholds'])
        
        if 'sacred_restrictions' in new_rules:
            self.sacred_restrictions.update(new_rules['sacred_restrictions'])
        
        self.logger.info("Cultural validation rules updated")
    
    def validate_batch_cultural_items(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate batch of cultural items for appropriateness
        
        Args:
            items: List of cultural items to validate
            
        Returns:
            Batch validation results
        """
        try:
            results = []
            
            for item in items:
                validation_result = self.validate_appropriateness(item)
                results.append({
                    'item': item.get('garment', 'unknown'),
                    'validation': validation_result,
                    'approved': validation_result.is_appropriate
                })
            
            # Calculate batch statistics
            total_items = len(results)
            approved_items = sum(1 for r in results if r['approved'])
            approval_rate = approved_items / total_items if total_items > 0 else 0
            
            # Identify high-risk items
            high_risk_items = [
                r for r in results 
                if r['validation'].risk_level in ['critical', 'high']
            ]
            
            batch_result = {
                'total_items': total_items,
                'approved_items': approved_items,
                'approval_rate': approval_rate,
                'high_risk_items': len(high_risk_items),
                'batch_recommendations': self._generate_batch_recommendations(results),
                'individual_results': results
            }
            
            return batch_result
            
        except Exception as e:
            self.logger.error(f"Batch validation failed: {str(e)}")
            return {'error': str(e)}