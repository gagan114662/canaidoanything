"""
Ethics Engine for Content Safety Service

This module provides ethics guideline enforcement and ethical
principle validation for AI-generated content.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EthicalPrinciple:
    """Ethical principle definition"""
    name: str
    description: str
    weight: float
    threshold: float
    violation_severity: str


@dataclass
class EthicsViolation:
    """Ethics violation structure"""
    principle: str
    severity: str
    description: str
    recommendation: str


class EthicsEngine:
    """
    Ethics engine for ethical guideline enforcement
    
    Provides comprehensive ethical principle validation,
    bias detection, and ethical compliance assessment.
    """
    
    def __init__(self):
        """Initialize ethics engine with ethical principles"""
        self.logger = logging.getLogger(__name__)
        
        # Core ethical principles
        self.ethical_principles = self._initialize_ethical_principles()
        
        # Ethics guidelines
        self.ethics_guidelines = self._initialize_ethics_guidelines()
        
        # Bias detection parameters
        self.bias_thresholds = {
            'representation_bias_max': 0.3,
            'demographic_imbalance_max': 0.4,
            'stereotyping_threshold': 0.2,
            'exploitation_risk_max': 0.1
        }
        
        # Ethical compliance thresholds
        self.compliance_thresholds = {
            'overall_ethics_min': 0.8,
            'principle_compliance_min': 0.7,
            'bias_tolerance_max': 0.3,
            'harm_prevention_min': 0.9
        }
        
        self.logger.info("Ethics Engine initialized with comprehensive principles")
    
    def _initialize_ethical_principles(self) -> Dict[str, EthicalPrinciple]:
        """Initialize core ethical principles"""
        principles = {
            'respect_for_persons': EthicalPrinciple(
                name='Respect for Persons',
                description='Treat all individuals with dignity and respect their autonomy',
                weight=0.25,
                threshold=0.8,
                violation_severity='high'
            ),
            'beneficence': EthicalPrinciple(
                name='Beneficence',
                description='Act in ways that promote wellbeing and avoid harm',
                weight=0.20,
                threshold=0.8,
                violation_severity='high'
            ),
            'justice': EthicalPrinciple(
                name='Justice',
                description='Ensure fair treatment and equal representation',
                weight=0.20,
                threshold=0.75,
                violation_severity='medium'
            ),
            'non_maleficence': EthicalPrinciple(
                name='Non-maleficence',
                description='Do no harm and prevent potential harm to individuals or groups',
                weight=0.15,
                threshold=0.9,
                violation_severity='critical'
            ),
            'transparency': EthicalPrinciple(
                name='Transparency',
                description='Be open about AI involvement and decision-making processes',
                weight=0.10,
                threshold=0.7,
                violation_severity='medium'
            ),
            'accountability': EthicalPrinciple(
                name='Accountability',
                description='Take responsibility for AI decisions and their consequences',
                weight=0.10,
                threshold=0.8,
                violation_severity='medium'
            )
        }
        
        return principles
    
    def _initialize_ethics_guidelines(self) -> Dict[str, Dict[str, Any]]:
        """Initialize specific ethics guidelines"""
        guidelines = {
            'representation': {
                'diverse_representation': {
                    'requirement': 'Include diverse demographics in generated content',
                    'threshold': 0.8,
                    'measurement': 'diversity_index'
                },
                'avoid_stereotypes': {
                    'requirement': 'Avoid harmful stereotypes and biased representations',
                    'threshold': 0.9,
                    'measurement': 'stereotype_detection_score'
                },
                'inclusive_messaging': {
                    'requirement': 'Promote inclusive and positive messaging',
                    'threshold': 0.8,
                    'measurement': 'inclusivity_score'
                }
            },
            'consent_and_autonomy': {
                'informed_consent': {
                    'requirement': 'Ensure informed consent for all individuals in content',
                    'threshold': 1.0,
                    'measurement': 'consent_verification'
                },
                'privacy_protection': {
                    'requirement': 'Protect individual privacy and personal information',
                    'threshold': 0.95,
                    'measurement': 'privacy_compliance_score'
                },
                'autonomy_respect': {
                    'requirement': 'Respect individual choices and self-determination',
                    'threshold': 0.9,
                    'measurement': 'autonomy_respect_score'
                }
            },
            'harm_prevention': {
                'no_exploitation': {
                    'requirement': 'Prevent exploitation of individuals or groups',
                    'threshold': 0.95,
                    'measurement': 'exploitation_risk_score'
                },
                'psychological_safety': {
                    'requirement': 'Ensure content does not cause psychological harm',
                    'threshold': 0.9,
                    'measurement': 'psychological_safety_score'
                },
                'social_responsibility': {
                    'requirement': 'Consider broader social impact of generated content',
                    'threshold': 0.8,
                    'measurement': 'social_impact_score'
                }
            },
            'fairness_and_justice': {
                'equal_treatment': {
                    'requirement': 'Provide equal quality treatment across all demographics',
                    'threshold': 0.9,
                    'measurement': 'treatment_equality_score'
                },
                'opportunity_equity': {
                    'requirement': 'Ensure equal opportunities for representation',
                    'threshold': 0.8,
                    'measurement': 'opportunity_equity_score'
                },
                'bias_mitigation': {
                    'requirement': 'Actively work to identify and mitigate biases',
                    'threshold': 0.85,
                    'measurement': 'bias_mitigation_effectiveness'
                }
            }
        }
        
        return guidelines
    
    def evaluate_ethics_compliance(self, ethics_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate ethics compliance for given scenario
        
        Args:
            ethics_scenario: Scenario data for ethical evaluation
            
        Returns:
            Comprehensive ethics compliance evaluation
        """
        try:
            # Extract scenario information
            representation = ethics_scenario.get('representation', 'unknown')
            message = ethics_scenario.get('message', 'unknown')
            commercial_intent = ethics_scenario.get('commercial_intent', False)
            target_audience = ethics_scenario.get('target_audience', 'general')
            consent_verified = ethics_scenario.get('consent', False)
            
            # Evaluate each ethical principle
            principle_scores = {}
            for principle_name, principle in self.ethical_principles.items():
                score = self._evaluate_principle_compliance(principle, ethics_scenario)
                principle_scores[principle_name] = score
            
            # Calculate weighted ethics score
            ethics_score = sum(
                principle_scores[name] * principle.weight
                for name, principle in self.ethical_principles.items()
            )
            
            # Evaluate specific guidelines
            guideline_compliance = self._evaluate_guideline_compliance(ethics_scenario)
            
            # Detect ethical violations
            ethical_violations = self._detect_ethical_violations(
                principle_scores, guideline_compliance, ethics_scenario
            )
            
            # Assess representation bias
            representation_bias = self._assess_representation_bias(ethics_scenario)
            
            # Check for exploitation risks
            exploitation_risk = self._assess_exploitation_risk(ethics_scenario)
            
            # Generate ethical recommendations
            ethical_recommendations = self._generate_ethical_recommendations(
                principle_scores, ethical_violations, ethics_scenario
            )
            
            result = {
                'ethics_score': ethics_score,
                'principle_scores': principle_scores,
                'guideline_compliance': guideline_compliance,
                'ethical_violations': [violation.__dict__ for violation in ethical_violations],
                'representation_bias': representation_bias,
                'exploitation_risk': exploitation_risk,
                'consent_issues': not consent_verified,
                'harmful_stereotypes': self._detect_harmful_stereotypes(ethics_scenario),
                'ethical_recommendations': ethical_recommendations,
                'compliance_level': self._determine_compliance_level(ethics_score)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ethics compliance evaluation failed: {str(e)}")
            return self._create_fallback_ethics_result()
    
    def _evaluate_principle_compliance(self, principle: EthicalPrinciple, 
                                     scenario: Dict[str, Any]) -> float:
        """Evaluate compliance with specific ethical principle"""
        try:
            principle_name = principle.name.lower().replace(' ', '_').replace('-', '_')
            
            if principle_name == 'respect_for_persons':
                return self._evaluate_respect_for_persons(scenario)
            elif principle_name == 'beneficence':
                return self._evaluate_beneficence(scenario)
            elif principle_name == 'justice':
                return self._evaluate_justice(scenario)
            elif principle_name == 'non_maleficence':
                return self._evaluate_non_maleficence(scenario)
            elif principle_name == 'transparency':
                return self._evaluate_transparency(scenario)
            elif principle_name == 'accountability':
                return self._evaluate_accountability(scenario)
            else:
                return 0.7  # Default neutral score
                
        except Exception as e:
            self.logger.error(f"Principle compliance evaluation failed for {principle.name}: {str(e)}")
            return 0.5
    
    def _evaluate_respect_for_persons(self, scenario: Dict[str, Any]) -> float:
        """Evaluate respect for persons principle"""
        try:
            score = 0.8  # Base score
            
            # Check consent verification
            if scenario.get('consent', False):
                score += 0.1
            else:
                score -= 0.3
            
            # Check for dignified representation
            representation = scenario.get('representation', '').lower()
            if 'dignified' in representation or 'respectful' in representation:
                score += 0.1
            elif 'objectifying' in representation or 'exploitative' in representation:
                score -= 0.4
            
            # Check for autonomy respect
            if scenario.get('autonomy_respected', True):
                score += 0.05
            else:
                score -= 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Respect for persons evaluation failed: {str(e)}")
            return 0.7
    
    def _evaluate_beneficence(self, scenario: Dict[str, Any]) -> float:
        """Evaluate beneficence principle"""
        try:
            score = 0.8  # Base score
            
            # Check for positive messaging
            message = scenario.get('message', '').lower()
            positive_indicators = ['positive', 'empowering', 'uplifting', 'inspiring']
            negative_indicators = ['negative', 'harmful', 'degrading', 'offensive']
            
            if any(indicator in message for indicator in positive_indicators):
                score += 0.1
            elif any(indicator in message for indicator in negative_indicators):
                score -= 0.3
            
            # Check for wellbeing promotion
            if scenario.get('promotes_wellbeing', False):
                score += 0.1
            
            # Check for harm potential
            harm_potential = scenario.get('harm_potential', 0.0)
            score -= harm_potential * 0.5
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Beneficence evaluation failed: {str(e)}")
            return 0.7
    
    def _evaluate_justice(self, scenario: Dict[str, Any]) -> float:
        """Evaluate justice principle"""
        try:
            score = 0.8  # Base score
            
            # Check for fair representation
            representation = scenario.get('representation', '').lower()
            if 'diverse' in representation or 'inclusive' in representation:
                score += 0.1
            elif 'biased' in representation or 'discriminatory' in representation:
                score -= 0.3
            
            # Check for equal treatment
            if scenario.get('equal_treatment', True):
                score += 0.1
            else:
                score -= 0.2
            
            # Check demographic balance
            demographic_diversity = scenario.get('demographic_diversity', 0.5)
            if demographic_diversity >= 0.8:
                score += 0.1
            elif demographic_diversity < 0.3:
                score -= 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Justice evaluation failed: {str(e)}")
            return 0.7
    
    def _evaluate_non_maleficence(self, scenario: Dict[str, Any]) -> float:
        """Evaluate non-maleficence principle"""
        try:
            score = 0.9  # High base score for harm prevention
            
            # Check for potential harm
            harm_indicators = ['harmful', 'dangerous', 'toxic', 'offensive', 'discriminatory']
            content_description = str(scenario.get('content_type', '') + ' ' + 
                                   scenario.get('message', '') + ' ' + 
                                   scenario.get('representation', '')).lower()
            
            harm_found = any(indicator in content_description for indicator in harm_indicators)
            if harm_found:
                score -= 0.4
            
            # Check exploitation risk
            exploitation_risk = scenario.get('exploitation_risk', 0.0)
            score -= exploitation_risk * 0.3
            
            # Check for protective measures
            if scenario.get('protective_measures', False):
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Non-maleficence evaluation failed: {str(e)}")
            return 0.8
    
    def _evaluate_transparency(self, scenario: Dict[str, Any]) -> float:
        """Evaluate transparency principle"""
        try:
            score = 0.7  # Base score
            
            # Check for AI disclosure
            if scenario.get('ai_disclosed', False):
                score += 0.2
            else:
                score -= 0.1
            
            # Check for process transparency
            if scenario.get('process_transparent', False):
                score += 0.1
            
            # Check for clear attribution
            if scenario.get('clear_attribution', False):
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Transparency evaluation failed: {str(e)}")
            return 0.7
    
    def _evaluate_accountability(self, scenario: Dict[str, Any]) -> float:
        """Evaluate accountability principle"""
        try:
            score = 0.8  # Base score
            
            # Check for responsibility acknowledgment
            if scenario.get('responsibility_acknowledged', False):
                score += 0.1
            
            # Check for audit trail
            if scenario.get('audit_trail', False):
                score += 0.1
            
            # Check for feedback mechanisms
            if scenario.get('feedback_mechanism', False):
                score += 0.05
            
            # Commercial use increases accountability requirements
            if scenario.get('commercial_intent', False):
                if not scenario.get('responsibility_acknowledged', False):
                    score -= 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Accountability evaluation failed: {str(e)}")
            return 0.7
    
    def _evaluate_guideline_compliance(self, scenario: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate compliance with specific ethics guidelines"""
        try:
            compliance_scores = {}
            
            for category, guidelines in self.ethics_guidelines.items():
                category_scores = {}
                
                for guideline_name, guideline_data in guidelines.items():
                    score = self._evaluate_specific_guideline(guideline_data, scenario)
                    category_scores[guideline_name] = score
                
                # Calculate category average
                compliance_scores[category] = sum(category_scores.values()) / len(category_scores)
            
            return compliance_scores
            
        except Exception as e:
            self.logger.error(f"Guideline compliance evaluation failed: {str(e)}")
            return {}
    
    def _evaluate_specific_guideline(self, guideline: Dict[str, Any], 
                                   scenario: Dict[str, Any]) -> float:
        """Evaluate compliance with a specific guideline"""
        try:
            measurement = guideline.get('measurement', 'generic')
            threshold = guideline.get('threshold', 0.8)
            
            # Evaluate based on measurement type
            if measurement == 'diversity_index':
                return scenario.get('demographic_diversity', 0.5)
            elif measurement == 'stereotype_detection_score':
                return 1.0 - scenario.get('stereotype_risk', 0.0)
            elif measurement == 'inclusivity_score':
                return scenario.get('inclusivity_score', 0.7)
            elif measurement == 'consent_verification':
                return 1.0 if scenario.get('consent', False) else 0.0
            elif measurement == 'privacy_compliance_score':
                return scenario.get('privacy_compliance', 0.8)
            elif measurement == 'exploitation_risk_score':
                return 1.0 - scenario.get('exploitation_risk', 0.0)
            elif measurement == 'psychological_safety_score':
                return scenario.get('psychological_safety', 0.8)
            elif measurement == 'social_impact_score':
                return scenario.get('social_impact', 0.7)
            elif measurement == 'treatment_equality_score':
                return scenario.get('equal_treatment_score', 0.8)
            elif measurement == 'bias_mitigation_effectiveness':
                return scenario.get('bias_mitigation', 0.7)
            else:
                return 0.7  # Default score
                
        except Exception as e:
            self.logger.error(f"Specific guideline evaluation failed: {str(e)}")
            return 0.5
    
    def _detect_ethical_violations(self, principle_scores: Dict[str, float],
                                 guideline_compliance: Dict[str, float],
                                 scenario: Dict[str, Any]) -> List[EthicsViolation]:
        """Detect specific ethical violations"""
        violations = []
        
        try:
            # Check principle violations
            for principle_name, principle in self.ethical_principles.items():
                score = principle_scores.get(principle_name, 0.5)
                
                if score < principle.threshold:
                    violation = EthicsViolation(
                        principle=principle_name,
                        severity=principle.violation_severity,
                        description=f"Score {score:.2f} below threshold {principle.threshold}",
                        recommendation=f"Improve {principle.description.lower()}"
                    )
                    violations.append(violation)
            
            # Check guideline violations
            for category, score in guideline_compliance.items():
                if score < self.compliance_thresholds.get(f'{category}_min', 0.8):
                    violation = EthicsViolation(
                        principle=f"guideline_{category}",
                        severity='medium',
                        description=f"Guideline compliance {score:.2f} below required standard",
                        recommendation=f"Review and improve {category} compliance"
                    )
                    violations.append(violation)
            
            # Check specific scenario violations
            if scenario.get('exploitation_risk', 0.0) > self.bias_thresholds['exploitation_risk_max']:
                violation = EthicsViolation(
                    principle='exploitation_prevention',
                    severity='critical',
                    description='High exploitation risk detected',
                    recommendation='Eliminate exploitative elements and ensure consent'
                )
                violations.append(violation)
            
            if not scenario.get('consent', False) and scenario.get('commercial_intent', False):
                violation = EthicsViolation(
                    principle='informed_consent',
                    severity='high',
                    description='Commercial use without verified consent',
                    recommendation='Obtain proper informed consent for commercial use'
                )
                violations.append(violation)
                
        except Exception as e:
            self.logger.error(f"Ethical violation detection failed: {str(e)}")
        
        return violations
    
    def _assess_representation_bias(self, scenario: Dict[str, Any]) -> float:
        """Assess representation bias in scenario"""
        try:
            bias_score = 0.0  # Lower is better
            
            # Check demographic diversity
            demographic_diversity = scenario.get('demographic_diversity', 0.5)
            if demographic_diversity < 0.5:
                bias_score += 0.3
            elif demographic_diversity < 0.7:
                bias_score += 0.1
            
            # Check for stereotypical representation
            stereotype_risk = scenario.get('stereotype_risk', 0.0)
            bias_score += stereotype_risk
            
            # Check for inclusive messaging
            inclusivity_score = scenario.get('inclusivity_score', 0.7)
            if inclusivity_score < 0.5:
                bias_score += 0.2
            elif inclusivity_score < 0.7:
                bias_score += 0.1
            
            return min(bias_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Representation bias assessment failed: {str(e)}")
            return 0.3
    
    def _assess_exploitation_risk(self, scenario: Dict[str, Any]) -> float:
        """Assess exploitation risk in scenario"""
        try:
            risk_score = 0.0
            
            # Check consent status
            if not scenario.get('consent', False):
                risk_score += 0.4
            
            # Check commercial intent without consent
            if scenario.get('commercial_intent', False) and not scenario.get('consent', False):
                risk_score += 0.3
            
            # Check for vulnerable populations
            if scenario.get('vulnerable_population', False):
                risk_score += 0.2
            
            # Check for objectifying content
            representation = scenario.get('representation', '').lower()
            if 'objectifying' in representation or 'exploitative' in representation:
                risk_score += 0.3
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Exploitation risk assessment failed: {str(e)}")
            return 0.1
    
    def _detect_harmful_stereotypes(self, scenario: Dict[str, Any]) -> bool:
        """Detect harmful stereotypes in scenario"""
        try:
            # Check for stereotype indicators
            stereotype_indicators = [
                'stereotypical', 'biased', 'reductive', 'limiting',
                'discriminatory', 'prejudiced', 'narrow'
            ]
            
            content_text = str(scenario.get('representation', '') + ' ' + 
                             scenario.get('message', '') + ' ' + 
                             scenario.get('content_type', '')).lower()
            
            stereotype_found = any(indicator in content_text for indicator in stereotype_indicators)
            
            # Check stereotype risk score
            stereotype_risk = scenario.get('stereotype_risk', 0.0)
            
            return stereotype_found or stereotype_risk > self.bias_thresholds['stereotyping_threshold']
            
        except Exception as e:
            self.logger.error(f"Harmful stereotype detection failed: {str(e)}")
            return False
    
    def _generate_ethical_recommendations(self, principle_scores: Dict[str, float],
                                        violations: List[EthicsViolation],
                                        scenario: Dict[str, Any]) -> List[str]:
        """Generate ethical improvement recommendations"""
        recommendations = []
        
        try:
            # Violation-specific recommendations
            for violation in violations:
                recommendations.append(violation.recommendation)
            
            # Principle-specific recommendations
            for principle_name, score in principle_scores.items():
                if score < 0.7:
                    principle = self.ethical_principles[principle_name]
                    recommendations.append(f"Improve {principle.description.lower()}")
            
            # Scenario-specific recommendations
            if scenario.get('commercial_intent', False):
                recommendations.extend([
                    'Ensure proper informed consent for commercial use',
                    'Consider profit-sharing with affected communities',
                    'Implement additional ethical review for commercial content'
                ])
            
            if scenario.get('demographic_diversity', 0.5) < 0.7:
                recommendations.append('Increase demographic diversity in representation')
            
            if scenario.get('exploitation_risk', 0.0) > 0.1:
                recommendations.extend([
                    'Review content for potential exploitation',
                    'Ensure all individuals are treated with dignity',
                    'Verify consent and autonomy of all participants'
                ])
            
            # General ethical recommendations
            recommendations.extend([
                'Maintain high ethical standards throughout process',
                'Regular ethical review and assessment',
                'Consider broader social impact of content',
                'Ensure transparency about AI involvement'
            ])
            
            # Remove duplicates and return
            return list(set(recommendations))
            
        except Exception as e:
            self.logger.error(f"Ethical recommendation generation failed: {str(e)}")
            return ['Comprehensive ethical review required']
    
    def _determine_compliance_level(self, ethics_score: float) -> str:
        """Determine overall compliance level"""
        if ethics_score >= 0.9:
            return 'excellent'
        elif ethics_score >= 0.8:
            return 'good'
        elif ethics_score >= 0.7:
            return 'acceptable'
        elif ethics_score >= 0.6:
            return 'concerning'
        else:
            return 'poor'
    
    def detect_representation_bias(self, content_representation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect representation bias in content
        
        Args:
            content_representation: Representation characteristics to analyze
            
        Returns:
            Bias detection results
        """
        try:
            # Extract representation metrics
            demographic_diversity = content_representation.get('demographic_diversity', 0.5)
            inclusive_messaging = content_representation.get('inclusive_messaging', False)
            stereotypical_elements = content_representation.get('stereotypical_elements', False)
            body_type_diversity = content_representation.get('body_type_diversity', 0.5)
            
            # Calculate bias indicators
            bias_indicators = []
            bias_score = 0.0
            
            # Demographic diversity bias
            if demographic_diversity < 0.5:
                bias_indicators.append('low_demographic_diversity')
                bias_score += 0.3
            elif demographic_diversity < 0.7:
                bias_score += 0.1
            
            # Body type diversity bias
            if body_type_diversity < 0.6:
                bias_indicators.append('limited_body_type_representation')
                bias_score += 0.2
            
            # Stereotypical elements
            if stereotypical_elements:
                bias_indicators.append('stereotypical_representation')
                bias_score += 0.4
            
            # Lack of inclusive messaging
            if not inclusive_messaging:
                bias_indicators.append('non_inclusive_messaging')
                bias_score += 0.2
            
            # Determine if bias is detected
            bias_detected = bias_score > self.bias_thresholds['representation_bias_max']
            
            # Categorize bias types
            bias_categories = self._categorize_representation_bias(bias_indicators)
            
            result = {
                'bias_detected': bias_detected,
                'bias_score': min(bias_score, 1.0),
                'bias_categories': bias_categories,
                'bias_indicators': bias_indicators,
                'diversity_scores': {
                    'demographic_diversity': demographic_diversity,
                    'body_type_diversity': body_type_diversity,
                    'overall_diversity': (demographic_diversity + body_type_diversity) / 2
                },
                'recommendations': self._generate_bias_mitigation_recommendations(bias_indicators)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Representation bias detection failed: {str(e)}")
            return self._create_fallback_bias_result()
    
    def _categorize_representation_bias(self, bias_indicators: List[str]) -> List[str]:
        """Categorize types of representation bias"""
        categories = []
        
        if 'low_demographic_diversity' in bias_indicators:
            categories.append('demographic_underrepresentation')
        
        if 'limited_body_type_representation' in bias_indicators:
            categories.append('body_type_bias')
        
        if 'stereotypical_representation' in bias_indicators:
            categories.append('stereotyping_bias')
        
        if 'non_inclusive_messaging' in bias_indicators:
            categories.append('exclusionary_messaging')
        
        return categories
    
    def _generate_bias_mitigation_recommendations(self, bias_indicators: List[str]) -> List[str]:
        """Generate bias mitigation recommendations"""
        recommendations = []
        
        if 'low_demographic_diversity' in bias_indicators:
            recommendations.extend([
                'Increase demographic diversity in content',
                'Ensure representative sampling across ethnic groups',
                'Review casting and selection processes for bias'
            ])
        
        if 'limited_body_type_representation' in bias_indicators:
            recommendations.extend([
                'Include diverse body types and sizes',
                'Avoid narrow beauty standards',
                'Promote body positivity and acceptance'
            ])
        
        if 'stereotypical_representation' in bias_indicators:
            recommendations.extend([
                'Eliminate stereotypical portrayals',
                'Consult with cultural representatives',
                'Review content for implicit biases'
            ])
        
        if 'non_inclusive_messaging' in bias_indicators:
            recommendations.extend([
                'Adopt inclusive language and messaging',
                'Consider impact on marginalized communities',
                'Promote positive and empowering messaging'
            ])
        
        # General recommendations
        recommendations.extend([
            'Implement regular bias auditing',
            'Train content creators on bias awareness',
            'Establish diverse review panels'
        ])
        
        return recommendations
    
    def get_ethical_principles(self) -> Dict[str, Dict[str, Any]]:
        """Get current ethical principles and their details"""
        try:
            principles_dict = {}
            for name, principle in self.ethical_principles.items():
                principles_dict[name] = {
                    'name': principle.name,
                    'description': principle.description,
                    'weight': principle.weight,
                    'threshold': principle.threshold,
                    'violation_severity': principle.violation_severity
                }
            
            return principles_dict
            
        except Exception as e:
            self.logger.error(f"Ethical principles retrieval failed: {str(e)}")
            return {}
    
    def update_ethical_principles(self, updated_principles: Dict[str, Dict[str, Any]]):
        """Update ethical principles configuration"""
        try:
            for principle_name, principle_data in updated_principles.items():
                if principle_name in self.ethical_principles:
                    principle = self.ethical_principles[principle_name]
                    
                    if 'weight' in principle_data:
                        principle.weight = principle_data['weight']
                    if 'threshold' in principle_data:
                        principle.threshold = principle_data['threshold']
                    if 'violation_severity' in principle_data:
                        principle.violation_severity = principle_data['violation_severity']
            
            self.logger.info("Ethical principles updated successfully")
            
        except Exception as e:
            self.logger.error(f"Ethical principles update failed: {str(e)}")
    
    def get_ethics_guidelines(self) -> Dict[str, Any]:
        """Get current ethics guidelines"""
        return self.ethics_guidelines.copy()
    
    def get_bias_thresholds(self) -> Dict[str, float]:
        """Get current bias detection thresholds"""
        return self.bias_thresholds.copy()
    
    def update_bias_thresholds(self, new_thresholds: Dict[str, float]):
        """Update bias detection thresholds"""
        try:
            self.bias_thresholds.update(new_thresholds)
            self.logger.info("Bias thresholds updated successfully")
        except Exception as e:
            self.logger.error(f"Bias threshold update failed: {str(e)}")
    
    # Fallback result creation methods
    def _create_fallback_ethics_result(self) -> Dict[str, Any]:
        """Create fallback ethics evaluation result"""
        return {
            'ethics_score': 0.5,
            'principle_scores': {},
            'guideline_compliance': {},
            'ethical_violations': [],
            'representation_bias': 0.5,
            'exploitation_risk': 0.5,
            'consent_issues': True,
            'harmful_stereotypes': False,
            'ethical_recommendations': ['Comprehensive ethical review required due to analysis error'],
            'compliance_level': 'unknown',
            'error': True
        }
    
    def _create_fallback_bias_result(self) -> Dict[str, Any]:
        """Create fallback bias detection result"""
        return {
            'bias_detected': True,
            'bias_score': 0.5,
            'bias_categories': ['analysis_error'],
            'bias_indicators': ['analysis_error'],
            'diversity_scores': {
                'demographic_diversity': 0.5,
                'body_type_diversity': 0.5,
                'overall_diversity': 0.5
            },
            'recommendations': ['Manual bias review required due to analysis error'],
            'error': True
        }