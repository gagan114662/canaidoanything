"""
Bias Detection Service for Garment Creative AI

This service provides comprehensive bias detection and fairness monitoring
across all AI transformations to ensure ethical and inclusive outputs.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from dataclasses import dataclass
from datetime import datetime

from app.services.ethics.demographic_classifier import DemographicClassifier
from app.services.ethics.fairness_metrics import FairnessMetrics
from app.core.config import settings


@dataclass
class BiasDetectionResult:
    """Result structure for bias detection analysis"""
    bias_score: float
    demographic_classification: Dict[str, Any]
    fairness_metrics: Dict[str, float]
    bias_detected: bool
    bias_severity: str
    recommendations: List[str]
    timestamp: datetime


@dataclass
class BiasThresholds:
    """Configurable bias detection thresholds"""
    demographic_parity_max_variance: float = 0.02  # 2% maximum variance
    equalized_odds_max_variance: float = 0.03      # 3% maximum variance
    quality_max_variance: float = 0.15             # 15% maximum quality variance
    individual_fairness_min: float = 0.95          # 95% minimum individual fairness
    cultural_sensitivity_min: float = 0.95         # 95% minimum cultural sensitivity


class BiasDetectionService:
    """
    Comprehensive bias detection and fairness monitoring service
    
    Provides real-time bias detection, demographic performance monitoring,
    and fairness metrics calculation for AI transformation pipeline.
    """
    
    def __init__(self):
        """Initialize bias detection service with classifiers and metrics"""
        self.logger = logging.getLogger(__name__)
        self.demographic_classifier = DemographicClassifier()
        self.fairness_metrics = FairnessMetrics()
        self.bias_thresholds = BiasThresholds()
        
        # Performance tracking
        self.performance_history = []
        self.bias_alerts_history = []
        
        # Initialize bias detection models
        self._initialize_bias_models()
        
        self.logger.info("Bias Detection Service initialized successfully")
    
    def _initialize_bias_models(self):
        """Initialize bias detection and demographic classification models"""
        try:
            # Initialize demographic classification models
            self.demographic_classifier.load_models()
            
            # Initialize fairness calculation components
            self.fairness_metrics.initialize()
            
            self.logger.info("Bias detection models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bias models: {str(e)}")
            # Fallback to basic demographic classification
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize fallback models when primary models fail"""
        self.logger.warning("Using fallback bias detection models")
        self.demographic_classifier.use_fallback_models()
    
    def classify_demographics(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify demographic characteristics of person in image
        
        Args:
            image: PIL Image containing person to classify
            
        Returns:
            Dictionary with demographic classifications and confidence scores
        """
        try:
            # Use demographic classifier for comprehensive analysis
            classification_result = self.demographic_classifier.classify_comprehensive(image)
            
            # Validate classification quality
            if classification_result['confidence'] < 0.7:
                self.logger.warning("Low confidence demographic classification")
                classification_result['requires_review'] = True
            
            return classification_result
            
        except Exception as e:
            self.logger.error(f"Demographic classification failed: {str(e)}")
            return self._fallback_demographic_classification(image)
    
    def _fallback_demographic_classification(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback demographic classification using basic analysis"""
        return {
            'age': 'unknown',
            'gender': 'unknown', 
            'ethnicity': 'unknown',
            'body_type': 'unknown',
            'confidence': 0.1,
            'requires_review': True,
            'fallback_used': True
        }
    
    def detect_performance_bias(self, performance_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Detect bias in performance across demographic groups
        
        Args:
            performance_data: Dictionary mapping demographic groups to performance metrics
            
        Returns:
            Bias detection results with severity and recommendations
        """
        if not performance_data or len(performance_data) < 2:
            return {
                'bias_detected': False,
                'bias_severity': 'none',
                'reason': 'insufficient_data'
            }
        
        try:
            # Calculate performance variances
            quality_scores = [group['quality'] for group in performance_data.values()]
            success_rates = [group.get('success_rate', 1.0) for group in performance_data.values()]
            processing_times = [group.get('processing_time', 0.0) for group in performance_data.values()]
            
            quality_variance = max(quality_scores) - min(quality_scores)
            success_variance = max(success_rates) - min(success_rates)
            time_variance = max(processing_times) - min(processing_times)
            
            # Assess bias severity
            bias_detected = (
                quality_variance > self.bias_thresholds.quality_max_variance or
                success_variance > self.bias_thresholds.demographic_parity_max_variance
            )
            
            bias_severity = self._calculate_bias_severity(
                quality_variance, success_variance, time_variance
            )
            
            # Generate bias report
            bias_report = {
                'bias_detected': bias_detected,
                'bias_severity': bias_severity,
                'demographic_parity': success_variance <= self.bias_thresholds.demographic_parity_max_variance,
                'quality_variance': quality_variance,
                'success_rate_variance': success_variance,
                'processing_time_variance': time_variance,
                'affected_groups': self._identify_affected_groups(performance_data),
                'recommendations': self._generate_bias_recommendations(bias_severity, quality_variance)
            }
            
            # Log bias detection results
            if bias_detected:
                self.logger.warning(f"Bias detected with severity: {bias_severity}")
                self._record_bias_alert(bias_report)
            
            return bias_report
            
        except Exception as e:
            self.logger.error(f"Performance bias detection failed: {str(e)}")
            return {'bias_detected': False, 'error': str(e)}
    
    def _calculate_bias_severity(self, quality_var: float, success_var: float, time_var: float) -> str:
        """Calculate bias severity based on variance metrics"""
        if quality_var > 0.3 or success_var > 0.1:
            return 'critical'
        elif quality_var > 0.2 or success_var > 0.05:
            return 'high'
        elif quality_var > 0.1 or success_var > 0.03:
            return 'medium'
        elif quality_var > 0.05 or success_var > 0.02:
            return 'low'
        else:
            return 'none'
    
    def _identify_affected_groups(self, performance_data: Dict[str, Dict[str, float]]) -> List[str]:
        """Identify demographic groups with lower performance"""
        if not performance_data:
            return []
        
        avg_quality = np.mean([group['quality'] for group in performance_data.values()])
        affected_groups = []
        
        for group_name, metrics in performance_data.items():
            if metrics['quality'] < avg_quality - 0.1:  # 10% below average
                affected_groups.append(group_name)
        
        return affected_groups
    
    def _generate_bias_recommendations(self, severity: str, quality_variance: float) -> List[str]:
        """Generate recommendations for bias mitigation"""
        recommendations = []
        
        if severity in ['critical', 'high']:
            recommendations.append("Immediate review and bias mitigation required")
            recommendations.append("Consider halting affected transformations")
            recommendations.append("Engage diversity and inclusion experts")
        
        if quality_variance > 0.15:
            recommendations.append("Retrain models with more diverse data")
            recommendations.append("Implement bias-aware loss functions")
        
        if severity in ['medium', 'low']:
            recommendations.append("Monitor bias trends closely")
            recommendations.append("Increase testing frequency")
        
        recommendations.append("Document bias mitigation efforts")
        return recommendations
    
    def measure_individual_fairness(self, image1: Image.Image, image2: Image.Image, 
                                  outcome1: float = None, outcome2: float = None) -> float:
        """
        Measure individual fairness between similar individuals
        
        Args:
            image1, image2: Images of similar individuals
            outcome1, outcome2: AI transformation outcomes (optional)
            
        Returns:
            Individual fairness score (0-1, higher is more fair)
        """
        try:
            return self.fairness_metrics.calculate_individual_fairness(
                image1, image2, outcome1, outcome2
            )
        except Exception as e:
            self.logger.error(f"Individual fairness measurement failed: {str(e)}")
            return 0.0  # Conservative fallback
    
    async def monitor_bias_realtime(self, image: Image.Image, 
                                  transformation_metadata: Dict[str, Any]) -> BiasDetectionResult:
        """
        Real-time bias monitoring for transformation pipeline
        
        Args:
            image: Input image for transformation
            transformation_metadata: Metadata about the transformation
            
        Returns:
            Comprehensive bias detection result
        """
        start_time = time.time()
        
        try:
            # Classify demographics
            demographic_classification = self.classify_demographics(image)
            
            # Calculate fairness metrics
            fairness_metrics = await self._calculate_fairness_metrics_async(
                image, demographic_classification, transformation_metadata
            )
            
            # Determine overall bias score
            bias_score = self._calculate_overall_bias_score(
                demographic_classification, fairness_metrics
            )
            
            # Assess bias severity
            bias_detected = bias_score > 0.3  # Configurable threshold
            bias_severity = self._assess_bias_severity_from_score(bias_score)
            
            # Generate recommendations
            recommendations = self._generate_realtime_recommendations(
                bias_score, demographic_classification, fairness_metrics
            )
            
            # Create result object
            result = BiasDetectionResult(
                bias_score=bias_score,
                demographic_classification=demographic_classification,
                fairness_metrics=fairness_metrics,
                bias_detected=bias_detected,
                bias_severity=bias_severity,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Performance logging
            processing_time = time.time() - start_time
            if processing_time > 2.0:  # Performance threshold
                self.logger.warning(f"Bias detection exceeded time limit: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Real-time bias monitoring failed: {str(e)}")
            return self._create_fallback_bias_result()
    
    async def _calculate_fairness_metrics_async(self, image: Image.Image, 
                                              demographic_data: Dict[str, Any],
                                              transformation_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate fairness metrics asynchronously"""
        try:
            # Run fairness calculations
            fairness_task = asyncio.create_task(
                self.fairness_metrics.calculate_comprehensive_fairness(
                    image, demographic_data, transformation_metadata
                )
            )
            
            fairness_result = await fairness_task
            return fairness_result
            
        except Exception as e:
            self.logger.error(f"Fairness metrics calculation failed: {str(e)}")
            return {'overall_fairness': 0.0, 'error': str(e)}
    
    def _calculate_overall_bias_score(self, demographic_data: Dict[str, Any], 
                                    fairness_metrics: Dict[str, float]) -> float:
        """Calculate overall bias score from multiple factors"""
        try:
            # Weight different bias factors
            demographic_confidence = demographic_data.get('confidence', 0.0)
            fairness_score = fairness_metrics.get('overall_fairness', 0.0)
            
            # Lower confidence in demographics may indicate bias
            demographic_bias = 1.0 - demographic_confidence
            
            # Lower fairness indicates higher bias
            fairness_bias = 1.0 - fairness_score
            
            # Weighted combination
            overall_bias = (demographic_bias * 0.3) + (fairness_bias * 0.7)
            
            return min(overall_bias, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Bias score calculation failed: {str(e)}")
            return 0.5  # Neutral fallback
    
    def _assess_bias_severity_from_score(self, bias_score: float) -> str:
        """Assess bias severity from overall bias score"""
        if bias_score >= 0.8:
            return 'critical'
        elif bias_score >= 0.6:
            return 'high'
        elif bias_score >= 0.4:
            return 'medium'
        elif bias_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_realtime_recommendations(self, bias_score: float, 
                                        demographic_data: Dict[str, Any],
                                        fairness_metrics: Dict[str, float]) -> List[str]:
        """Generate real-time bias mitigation recommendations"""
        recommendations = []
        
        if bias_score > 0.7:
            recommendations.append("Block transformation - critical bias detected")
            recommendations.append("Route to expert review immediately")
        elif bias_score > 0.5:
            recommendations.append("Flag for enhanced monitoring")
            recommendations.append("Apply bias mitigation algorithms")
        elif bias_score > 0.3:
            recommendations.append("Monitor closely for bias patterns")
        
        # Specific recommendations based on classification confidence
        if demographic_data.get('confidence', 1.0) < 0.7:
            recommendations.append("Low demographic classification confidence - increase monitoring")
        
        return recommendations
    
    def _create_fallback_bias_result(self) -> BiasDetectionResult:
        """Create fallback bias detection result on error"""
        return BiasDetectionResult(
            bias_score=0.5,  # Neutral score when uncertain
            demographic_classification={'confidence': 0.0, 'error': True},
            fairness_metrics={'overall_fairness': 0.0},
            bias_detected=True,  # Conservative approach
            bias_severity='unknown',
            recommendations=['Manual review required due to bias detection error'],
            timestamp=datetime.now()
        )
    
    def generate_mitigation_recommendations(self, bias_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate comprehensive bias mitigation recommendations
        
        Args:
            bias_data: Detected bias information and severity
            
        Returns:
            Structured mitigation recommendations by category
        """
        recommendations = {
            'immediate_actions': [],
            'long_term_improvements': [],
            'alternative_algorithms': [],
            'monitoring_frequency': [],
            'expert_consultation': []
        }
        
        severity = bias_data.get('bias_severity', 'unknown')
        affected_groups = bias_data.get('affected_groups', [])
        
        # Immediate actions based on severity
        if severity in ['critical', 'high']:
            recommendations['immediate_actions'].extend([
                'Halt affected transformations immediately',
                'Activate expert review protocol',
                'Implement emergency bias mitigation',
                'Notify compliance and ethics teams'
            ])
        elif severity == 'medium':
            recommendations['immediate_actions'].extend([
                'Increase monitoring frequency to real-time',
                'Apply additional bias filters',
                'Flag affected outputs for review'
            ])
        
        # Long-term improvements
        recommendations['long_term_improvements'].extend([
            'Expand training data diversity',
            'Implement bias-aware training objectives',
            'Regular bias auditing schedule',
            'Continuous bias monitoring integration'
        ])
        
        # Alternative algorithms for affected groups
        if affected_groups:
            recommendations['alternative_algorithms'].extend([
                f'Group-specific enhancement for {group}' for group in affected_groups
            ])
        
        # Monitoring frequency
        if severity in ['critical', 'high']:
            recommendations['monitoring_frequency'].append('Real-time monitoring required')
        elif severity == 'medium':
            recommendations['monitoring_frequency'].append('Hourly bias checks')
        else:
            recommendations['monitoring_frequency'].append('Daily bias monitoring')
        
        return recommendations
    
    def analyze_batch_bias(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze bias patterns across batch of transformations
        
        Args:
            batch_data: List of transformation results with demographics
            
        Returns:
            Comprehensive batch bias analysis
        """
        if not batch_data:
            return {'error': 'No data provided for batch analysis'}
        
        try:
            # Group by demographics
            demographic_groups = {}
            for item in batch_data:
                demographics = item.get('demographics', {})
                ethnicity = demographics.get('ethnicity', 'unknown')
                
                if ethnicity not in demographic_groups:
                    demographic_groups[ethnicity] = []
                
                demographic_groups[ethnicity].append(item)
            
            # Calculate performance metrics per group
            group_performance = {}
            for group, items in demographic_groups.items():
                qualities = [item.get('quality', 0.0) for item in items]
                group_performance[group] = {
                    'avg_quality': np.mean(qualities),
                    'quality_std': np.std(qualities),
                    'count': len(items)
                }
            
            # Detect bias in batch
            bias_analysis = self.detect_performance_bias(group_performance)
            
            # Calculate overall bias score
            overall_bias_score = self._calculate_batch_bias_score(group_performance)
            
            # Trend analysis
            trend_analysis = self._analyze_bias_trends(batch_data)
            
            return {
                'overall_bias_score': overall_bias_score,
                'demographic_breakdown': group_performance,
                'bias_analysis': bias_analysis,
                'trend_analysis': trend_analysis,
                'recommendations': self.generate_mitigation_recommendations(bias_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Batch bias analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_batch_bias_score(self, group_performance: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall bias score for batch"""
        if len(group_performance) < 2:
            return 0.0
        
        qualities = [group['avg_quality'] for group in group_performance.values()]
        quality_variance = max(qualities) - min(qualities)
        
        # Normalize variance to 0-1 scale
        return min(quality_variance / 3.0, 1.0)  # Assuming max expected variance of 3.0
    
    def _analyze_bias_trends(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze bias trends in batch data"""
        # Placeholder for trend analysis
        return {
            'trend_direction': 'stable',
            'bias_increasing': False,
            'pattern_detected': False
        }
    
    def check_bias_alert(self, performance_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Check if bias alert should be triggered
        
        Args:
            performance_data: Performance metrics across demographic groups
            
        Returns:
            Alert information and triggering details
        """
        bias_result = self.detect_performance_bias(performance_data)
        
        alert = {
            'alert_triggered': False,
            'severity': 'none',
            'immediate_action_required': False,
            'notification_sent': False,
            'escalation_required': False
        }
        
        if bias_result.get('bias_detected', False):
            severity = bias_result.get('bias_severity', 'none')
            
            alert['alert_triggered'] = True
            alert['severity'] = severity
            
            if severity in ['critical', 'high']:
                alert['immediate_action_required'] = True
                alert['escalation_required'] = True
                alert['notification_sent'] = self._send_bias_alert_notification(bias_result)
            
            # Record alert
            self._record_bias_alert(alert)
        
        return alert
    
    def _send_bias_alert_notification(self, bias_result: Dict[str, Any]) -> bool:
        """Send bias alert notification to relevant teams"""
        try:
            # Placeholder for notification system integration
            self.logger.critical(f"BIAS ALERT: {bias_result['bias_severity']} bias detected")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send bias alert: {str(e)}")
            return False
    
    def _record_bias_alert(self, alert_data: Dict[str, Any]):
        """Record bias alert for audit trail"""
        alert_record = {
            'timestamp': datetime.now(),
            'alert_data': alert_data
        }
        self.bias_alerts_history.append(alert_record)
        
        # Keep only recent alerts (last 1000)
        if len(self.bias_alerts_history) > 1000:
            self.bias_alerts_history = self.bias_alerts_history[-1000:]
    
    def validate_demographic_distribution(self, distribution: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate demographic distribution for balance and diversity
        
        Args:
            distribution: Dictionary mapping demographic groups to proportions
            
        Returns:
            Distribution validation results
        """
        try:
            total_proportion = sum(distribution.values())
            
            # Normalize if needed
            if abs(total_proportion - 1.0) > 0.01:
                distribution = {k: v/total_proportion for k, v in distribution.items()}
            
            # Calculate diversity metrics
            num_groups = len(distribution)
            entropy = -sum(p * np.log(p) for p in distribution.values() if p > 0)
            max_entropy = np.log(num_groups)
            diversity_index = entropy / max_entropy if max_entropy > 0 else 0
            
            # Check balance (no group dominates)
            max_proportion = max(distribution.values())
            is_balanced = max_proportion < 0.5  # No single group > 50%
            
            return {
                'is_balanced': is_balanced,
                'diversity_index': diversity_index,
                'max_proportion': max_proportion,
                'group_count': num_groups,
                'distribution_quality': 'good' if diversity_index > 0.8 else 'poor'
            }
            
        except Exception as e:
            self.logger.error(f"Distribution validation failed: {str(e)}")
            return {'error': str(e)}
    
    def get_bias_detection_stats(self) -> Dict[str, Any]:
        """Get bias detection service statistics"""
        return {
            'total_alerts': len(self.bias_alerts_history),
            'recent_alerts': len([a for a in self.bias_alerts_history 
                                if (datetime.now() - a['timestamp']).days < 7]),
            'service_status': 'active',
            'models_loaded': self.demographic_classifier.models_loaded,
            'performance_history_size': len(self.performance_history)
        }