"""
Fairness Metrics Calculator for Bias Detection

This module provides comprehensive fairness metrics calculation including
demographic parity, equalized odds, and individual fairness measures.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
import cv2
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F


@dataclass
class FairnessResult:
    """Structured fairness metrics result"""
    demographic_parity: float
    equalized_odds: float
    individual_fairness: float
    overall_fairness: float
    group_performance: Dict[str, float]
    bias_indicators: List[str]


class FairnessMetrics:
    """
    Comprehensive fairness metrics calculator
    
    Provides calculation of key fairness metrics including demographic parity,
    equalized odds, individual fairness, and group fairness measures.
    """
    
    def __init__(self):
        """Initialize fairness metrics calculator"""
        self.logger = logging.getLogger(__name__)
        
        # Fairness thresholds
        self.fairness_thresholds = {
            'demographic_parity_min': 0.97,  # 3% max difference
            'equalized_odds_min': 0.95,     # 5% max difference
            'individual_fairness_min': 0.95, # 95% similarity for similar inputs
            'group_fairness_min': 0.90      # 90% minimum group performance
        }
        
        # Performance tracking
        self.fairness_history = []
        
        self.logger.info("Fairness Metrics calculator initialized")
    
    def initialize(self):
        """Initialize fairness calculation components"""
        try:
            # Initialize any required models or components
            self.logger.info("Fairness metrics components initialized")
        except Exception as e:
            self.logger.error(f"Fairness metrics initialization failed: {str(e)}")
    
    def calculate_demographic_parity(self, group_outcomes: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Calculate demographic parity across groups
        
        Demographic parity requires that the positive outcome rates are
        similar across different demographic groups.
        
        Args:
            group_outcomes: Dictionary mapping groups to outcome counts
                          Format: {"group": {"positive_outcomes": int, "total": int}}
            
        Returns:
            Demographic parity analysis results
        """
        try:
            if len(group_outcomes) < 2:
                return {
                    'parity_achieved': True,
                    'max_variance': 0.0,
                    'group_rates': {},
                    'reason': 'insufficient_groups'
                }
            
            # Calculate positive outcome rates for each group
            group_rates = {}
            for group, outcomes in group_outcomes.items():
                positive = outcomes.get('positive_outcomes', 0)
                total = outcomes.get('total', 1)
                rate = positive / max(total, 1)  # Avoid division by zero
                group_rates[group] = rate
            
            # Calculate parity metrics
            rates = list(group_rates.values())
            max_rate = max(rates)
            min_rate = min(rates)
            max_variance = max_rate - min_rate
            
            # Check if parity is achieved
            parity_threshold = 1.0 - self.fairness_thresholds['demographic_parity_min']
            parity_achieved = max_variance <= parity_threshold
            
            # Identify problematic groups
            avg_rate = np.mean(rates)
            underperforming_groups = [
                group for group, rate in group_rates.items()
                if rate < avg_rate - (parity_threshold / 2)
            ]
            
            result = {
                'parity_achieved': parity_achieved,
                'max_variance': max_variance,
                'group_rates': group_rates,
                'average_rate': avg_rate,
                'underperforming_groups': underperforming_groups,
                'parity_score': 1.0 - max_variance  # Higher is better
            }
            
            self.logger.info(f"Demographic parity: {'achieved' if parity_achieved else 'violated'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Demographic parity calculation failed: {str(e)}")
            return {'error': str(e), 'parity_achieved': False}
    
    def calculate_equalized_odds(self, classification_data: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Calculate equalized odds across demographic groups
        
        Equalized odds requires that true positive rates and false positive rates
        are similar across groups.
        
        Args:
            classification_data: Dictionary with confusion matrix data per group
                               Format: {"group": {"tp": int, "fp": int, "tn": int, "fn": int}}
            
        Returns:
            Equalized odds analysis results
        """
        try:
            if len(classification_data) < 2:
                return {
                    'equalized_odds_achieved': True,
                    'tpr_variance': 0.0,
                    'fpr_variance': 0.0,
                    'group_metrics': {},
                    'reason': 'insufficient_groups'
                }
            
            # Calculate TPR and FPR for each group
            group_metrics = {}
            tpr_values = []
            fpr_values = []
            
            for group, data in classification_data.items():
                tp = data.get('tp', 0)
                fp = data.get('fp', 0)
                tn = data.get('tn', 0)
                fn = data.get('fn', 0)
                
                # Calculate rates
                tpr = tp / max(tp + fn, 1)  # True Positive Rate (Sensitivity)
                fpr = fp / max(fp + tn, 1)  # False Positive Rate
                
                group_metrics[group] = {
                    'tpr': tpr,
                    'fpr': fpr,
                    'accuracy': (tp + tn) / max(tp + tn + fp + fn, 1),
                    'precision': tp / max(tp + fp, 1),
                    'recall': tpr
                }
                
                tpr_values.append(tpr)
                fpr_values.append(fpr)
            
            # Calculate variances
            tpr_variance = max(tpr_values) - min(tpr_values)
            fpr_variance = max(fpr_values) - min(fpr_values)
            
            # Check equalized odds achievement
            odds_threshold = 1.0 - self.fairness_thresholds['equalized_odds_min']
            equalized_odds_achieved = (
                tpr_variance <= odds_threshold and 
                fpr_variance <= odds_threshold
            )
            
            result = {
                'equalized_odds_achieved': equalized_odds_achieved,
                'tpr_variance': tpr_variance,
                'fpr_variance': fpr_variance,
                'group_metrics': group_metrics,
                'average_tpr': np.mean(tpr_values),
                'average_fpr': np.mean(fpr_values),
                'equalized_odds_score': 1.0 - max(tpr_variance, fpr_variance)
            }
            
            self.logger.info(f"Equalized odds: {'achieved' if equalized_odds_achieved else 'violated'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Equalized odds calculation failed: {str(e)}")
            return {'error': str(e), 'equalized_odds_achieved': False}
    
    def calculate_individual_fairness(self, image1: Image.Image, image2: Image.Image,
                                    outcome1: float = None, outcome2: float = None) -> float:
        """
        Calculate individual fairness between similar individuals
        
        Individual fairness requires that similar individuals receive similar outcomes.
        
        Args:
            image1, image2: Images of individuals to compare
            outcome1, outcome2: AI system outcomes for comparison (optional)
            
        Returns:
            Individual fairness score (0-1, higher is more fair)
        """
        try:
            # Calculate image similarity
            image_similarity = self._calculate_image_similarity(image1, image2)
            
            # If outcomes provided, calculate outcome similarity
            if outcome1 is not None and outcome2 is not None:
                outcome_similarity = self._calculate_outcome_similarity(outcome1, outcome2)
                
                # Individual fairness: similar images should have similar outcomes
                # If images are very similar, outcomes should also be similar
                fairness_score = self._compute_individual_fairness_score(
                    image_similarity, outcome_similarity
                )
            else:
                # Without outcomes, base fairness on image similarity
                fairness_score = image_similarity
            
            self.logger.debug(f"Individual fairness score: {fairness_score:.3f}")
            return float(fairness_score)
            
        except Exception as e:
            self.logger.error(f"Individual fairness calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_image_similarity(self, image1: Image.Image, image2: Image.Image) -> float:
        """Calculate similarity between two images"""
        try:
            # Resize images to same size for comparison
            size = (256, 256)
            img1_resized = image1.resize(size)
            img2_resized = image2.resize(size)
            
            # Convert to numpy arrays
            arr1 = np.array(img1_resized, dtype=np.float32) / 255.0
            arr2 = np.array(img2_resized, dtype=np.float32) / 255.0
            
            # Calculate structural similarity
            # Simple method: normalized cross-correlation
            correlation = np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0
            
            # Convert correlation to similarity (0-1 range)
            similarity = (correlation + 1.0) / 2.0
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.error(f"Image similarity calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_outcome_similarity(self, outcome1: float, outcome2: float) -> float:
        """Calculate similarity between two outcomes"""
        try:
            # Normalize outcomes if they're not in 0-1 range
            if max(abs(outcome1), abs(outcome2)) > 1.0:
                # Assume outcomes are quality scores out of 10
                outcome1 = outcome1 / 10.0
                outcome2 = outcome2 / 10.0
            
            # Calculate absolute difference
            difference = abs(outcome1 - outcome2)
            
            # Convert to similarity (closer outcomes = higher similarity)
            similarity = 1.0 - difference
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.error(f"Outcome similarity calculation failed: {str(e)}")
            return 0.0
    
    def _compute_individual_fairness_score(self, image_similarity: float, 
                                         outcome_similarity: float) -> float:
        """Compute individual fairness score from similarities"""
        try:
            # Individual fairness principle: similar inputs should yield similar outputs
            # If images are similar (high image_similarity), outcomes should be similar too
            
            if image_similarity > 0.8:  # Very similar images
                # Outcomes should be very similar too
                fairness_score = outcome_similarity
            elif image_similarity > 0.6:  # Moderately similar images
                # Some outcome variation is acceptable
                fairness_score = min(outcome_similarity + 0.2, 1.0)
            else:  # Different images
                # Large outcome differences are acceptable
                fairness_score = min(outcome_similarity + 0.4, 1.0)
            
            return fairness_score
            
        except Exception as e:
            self.logger.error(f"Individual fairness score computation failed: {str(e)}")
            return 0.0
    
    async def calculate_comprehensive_fairness(self, image: Image.Image,
                                             demographic_data: Dict[str, Any],
                                             transformation_metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive fairness metrics for a transformation
        
        Args:
            image: Input image
            demographic_data: Demographic classification results
            transformation_metadata: Results and metadata from transformation
            
        Returns:
            Comprehensive fairness metrics
        """
        try:
            # Initialize fairness scores
            fairness_scores = {
                'demographic_fairness': 0.0,
                'quality_fairness': 0.0,
                'processing_fairness': 0.0,
                'overall_fairness': 0.0
            }
            
            # Calculate demographic-based fairness
            demographic_fairness = await self._assess_demographic_fairness(
                demographic_data, transformation_metadata
            )
            fairness_scores['demographic_fairness'] = demographic_fairness
            
            # Calculate quality fairness
            quality_fairness = self._assess_quality_fairness(
                demographic_data, transformation_metadata
            )
            fairness_scores['quality_fairness'] = quality_fairness
            
            # Calculate processing fairness
            processing_fairness = self._assess_processing_fairness(
                demographic_data, transformation_metadata
            )
            fairness_scores['processing_fairness'] = processing_fairness
            
            # Calculate overall fairness
            overall_fairness = np.mean([
                demographic_fairness,
                quality_fairness,
                processing_fairness
            ])
            fairness_scores['overall_fairness'] = overall_fairness
            
            return fairness_scores
            
        except Exception as e:
            self.logger.error(f"Comprehensive fairness calculation failed: {str(e)}")
            return {'overall_fairness': 0.0, 'error': str(e)}
    
    async def _assess_demographic_fairness(self, demographic_data: Dict[str, Any],
                                         transformation_metadata: Dict[str, Any]) -> float:
        """Assess fairness based on demographic characteristics"""
        try:
            # Check if demographic classification has sufficient confidence
            confidence = demographic_data.get('confidence', 0.0)
            
            if confidence < 0.7:
                # Low confidence may indicate bias in demographic classification
                return 0.6  # Moderate fairness score
            
            # Check for demographic-specific quality issues
            ethnicity = demographic_data.get('ethnicity', 'unknown')
            quality_score = transformation_metadata.get('quality_score', 5.0)
            
            # Historical performance check (placeholder)
            expected_quality = self._get_expected_quality_for_demographic(ethnicity)
            
            if quality_score >= expected_quality * 0.95:  # Within 5% of expected
                return 0.95
            elif quality_score >= expected_quality * 0.90:  # Within 10%
                return 0.85
            else:
                return 0.70  # Below expected performance
            
        except Exception as e:
            self.logger.error(f"Demographic fairness assessment failed: {str(e)}")
            return 0.5
    
    def _get_expected_quality_for_demographic(self, ethnicity: str) -> float:
        """Get expected quality score for demographic group"""
        # Placeholder implementation - in production, use historical data
        # All groups should have equal expected quality (fairness principle)
        return 8.5  # Target quality score
    
    def _assess_quality_fairness(self, demographic_data: Dict[str, Any],
                               transformation_metadata: Dict[str, Any]) -> float:
        """Assess fairness in quality outcomes"""
        try:
            quality_score = transformation_metadata.get('quality_score', 5.0)
            
            # Check if quality meets minimum standards regardless of demographics
            if quality_score >= 8.5:  # High quality
                return 1.0
            elif quality_score >= 7.5:  # Good quality
                return 0.9
            elif quality_score >= 6.5:  # Acceptable quality
                return 0.8
            else:
                return 0.6  # Below acceptable quality
            
        except Exception as e:
            self.logger.error(f"Quality fairness assessment failed: {str(e)}")
            return 0.5
    
    def _assess_processing_fairness(self, demographic_data: Dict[str, Any],
                                  transformation_metadata: Dict[str, Any]) -> float:
        """Assess fairness in processing time and resources"""
        try:
            processing_time = transformation_metadata.get('processing_time', 30.0)
            
            # Processing time should be consistent regardless of demographics
            if processing_time <= 25.0:  # Fast processing
                return 1.0
            elif processing_time <= 30.0:  # Target processing
                return 0.95
            elif processing_time <= 35.0:  # Acceptable processing
                return 0.85
            else:
                return 0.7  # Slow processing
            
        except Exception as e:
            self.logger.error(f"Processing fairness assessment failed: {str(e)}")
            return 0.5
    
    def calculate_group_fairness(self, group_performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate group fairness metrics across multiple demographic groups
        
        Args:
            group_performance_data: Performance data per demographic group
                                  Format: {"group": [score1, score2, ...]}
            
        Returns:
            Group fairness analysis
        """
        try:
            if len(group_performance_data) < 2:
                return {
                    'group_fairness_achieved': True,
                    'reason': 'insufficient_groups'
                }
            
            # Calculate statistics for each group
            group_stats = {}
            all_means = []
            
            for group, scores in group_performance_data.items():
                if scores:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    
                    group_stats[group] = {
                        'mean': mean_score,
                        'std': std_score,
                        'count': len(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    }
                    all_means.append(mean_score)
            
            # Calculate group fairness metrics
            if all_means:
                overall_mean = np.mean(all_means)
                max_deviation = max(abs(mean - overall_mean) for mean in all_means)
                
                # Group fairness threshold
                fairness_threshold = overall_mean * 0.1  # 10% deviation allowed
                group_fairness_achieved = max_deviation <= fairness_threshold
                
                # Calculate fairness score
                group_fairness_score = 1.0 - (max_deviation / overall_mean)
                group_fairness_score = max(0.0, min(1.0, group_fairness_score))
                
                result = {
                    'group_fairness_achieved': group_fairness_achieved,
                    'group_stats': group_stats,
                    'overall_mean': overall_mean,
                    'max_deviation': max_deviation,
                    'group_fairness_score': group_fairness_score,
                    'underperforming_groups': [
                        group for group, stats in group_stats.items()
                        if stats['mean'] < overall_mean - fairness_threshold
                    ]
                }
                
                return result
            else:
                return {'error': 'No valid performance data'}
            
        except Exception as e:
            self.logger.error(f"Group fairness calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def calculate_intersectional_fairness(self, performance_data: Dict[Tuple[str, str], List[float]]) -> Dict[str, Any]:
        """
        Calculate intersectional fairness across multiple demographic dimensions
        
        Args:
            performance_data: Performance data for intersectional groups
                            Format: {(ethnicity, gender): [scores...]}
            
        Returns:
            Intersectional fairness analysis
        """
        try:
            if len(performance_data) < 4:  # Need multiple intersectional groups
                return {
                    'intersectional_fairness_achieved': True,
                    'reason': 'insufficient_intersectional_groups'
                }
            
            # Calculate performance for each intersectional group
            intersectional_stats = {}
            all_means = []
            
            for group_key, scores in performance_data.items():
                if scores:
                    ethnicity, gender = group_key
                    group_name = f"{ethnicity}_{gender}"
                    
                    mean_score = np.mean(scores)
                    intersectional_stats[group_name] = {
                        'mean': mean_score,
                        'count': len(scores),
                        'ethnicity': ethnicity,
                        'gender': gender
                    }
                    all_means.append(mean_score)
            
            # Analyze intersectional fairness
            if all_means:
                overall_mean = np.mean(all_means)
                max_deviation = max(abs(mean - overall_mean) for mean in all_means)
                
                # Stricter threshold for intersectional fairness
                fairness_threshold = overall_mean * 0.05  # 5% deviation allowed
                intersectional_fairness_achieved = max_deviation <= fairness_threshold
                
                intersectional_fairness_score = 1.0 - (max_deviation / overall_mean)
                intersectional_fairness_score = max(0.0, min(1.0, intersectional_fairness_score))
                
                return {
                    'intersectional_fairness_achieved': intersectional_fairness_achieved,
                    'intersectional_stats': intersectional_stats,
                    'overall_mean': overall_mean,
                    'max_deviation': max_deviation,
                    'intersectional_fairness_score': intersectional_fairness_score
                }
            else:
                return {'error': 'No valid intersectional data'}
            
        except Exception as e:
            self.logger.error(f"Intersectional fairness calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def generate_fairness_report(self, all_fairness_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive fairness report
        
        Args:
            all_fairness_metrics: Combined fairness metrics from various calculations
            
        Returns:
            Comprehensive fairness report with recommendations
        """
        try:
            # Extract key metrics
            demographic_parity = all_fairness_metrics.get('demographic_parity', {})
            equalized_odds = all_fairness_metrics.get('equalized_odds', {})
            individual_fairness = all_fairness_metrics.get('individual_fairness', 0.0)
            group_fairness = all_fairness_metrics.get('group_fairness', {})
            
            # Calculate overall fairness score
            fairness_components = []
            
            if demographic_parity.get('parity_score'):
                fairness_components.append(demographic_parity['parity_score'])
            
            if equalized_odds.get('equalized_odds_score'):
                fairness_components.append(equalized_odds['equalized_odds_score'])
            
            if individual_fairness:
                fairness_components.append(individual_fairness)
            
            if group_fairness.get('group_fairness_score'):
                fairness_components.append(group_fairness['group_fairness_score'])
            
            overall_fairness_score = np.mean(fairness_components) if fairness_components else 0.0
            
            # Determine fairness level
            if overall_fairness_score >= 0.95:
                fairness_level = 'excellent'
            elif overall_fairness_score >= 0.90:
                fairness_level = 'good'
            elif overall_fairness_score >= 0.80:
                fairness_level = 'acceptable'
            elif overall_fairness_score >= 0.70:
                fairness_level = 'concerning'
            else:
                fairness_level = 'poor'
            
            # Generate recommendations
            recommendations = self._generate_fairness_recommendations(
                all_fairness_metrics, fairness_level
            )
            
            # Compile report
            fairness_report = {
                'overall_fairness_score': overall_fairness_score,
                'fairness_level': fairness_level,
                'component_scores': {
                    'demographic_parity': demographic_parity.get('parity_score', 0.0),
                    'equalized_odds': equalized_odds.get('equalized_odds_score', 0.0),
                    'individual_fairness': individual_fairness,
                    'group_fairness': group_fairness.get('group_fairness_score', 0.0)
                },
                'fairness_violations': self._identify_fairness_violations(all_fairness_metrics),
                'recommendations': recommendations,
                'next_review_date': self._calculate_next_review_date(fairness_level)
            }
            
            return fairness_report
            
        except Exception as e:
            self.logger.error(f"Fairness report generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_fairness_recommendations(self, metrics: Dict[str, Any], level: str) -> List[str]:
        """Generate fairness improvement recommendations"""
        recommendations = []
        
        if level in ['poor', 'concerning']:
            recommendations.extend([
                "Immediate bias mitigation required",
                "Increase monitoring frequency to real-time",
                "Review and retrain models with diverse data",
                "Implement bias-aware loss functions"
            ])
        
        if level in ['acceptable', 'concerning', 'poor']:
            recommendations.extend([
                "Expand demographic representation in training data",
                "Implement regular fairness auditing",
                "Consider group-specific model adjustments"
            ])
        
        # Specific recommendations based on violations
        demographic_parity = metrics.get('demographic_parity', {})
        if not demographic_parity.get('parity_achieved', True):
            recommendations.append("Address demographic parity violations")
        
        equalized_odds = metrics.get('equalized_odds', {})
        if not equalized_odds.get('equalized_odds_achieved', True):
            recommendations.append("Improve equalized odds across groups")
        
        return recommendations
    
    def _identify_fairness_violations(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify specific fairness violations"""
        violations = []
        
        # Check demographic parity
        demographic_parity = metrics.get('demographic_parity', {})
        if not demographic_parity.get('parity_achieved', True):
            violations.append("demographic_parity_violation")
        
        # Check equalized odds
        equalized_odds = metrics.get('equalized_odds', {})
        if not equalized_odds.get('equalized_odds_achieved', True):
            violations.append("equalized_odds_violation")
        
        # Check individual fairness
        individual_fairness = metrics.get('individual_fairness', 1.0)
        if individual_fairness < self.fairness_thresholds['individual_fairness_min']:
            violations.append("individual_fairness_violation")
        
        # Check group fairness
        group_fairness = metrics.get('group_fairness', {})
        if not group_fairness.get('group_fairness_achieved', True):
            violations.append("group_fairness_violation")
        
        return violations
    
    def _calculate_next_review_date(self, fairness_level: str) -> str:
        """Calculate when next fairness review should occur"""
        from datetime import datetime, timedelta
        
        if fairness_level in ['poor', 'concerning']:
            days = 7  # Weekly review
        elif fairness_level == 'acceptable':
            days = 30  # Monthly review
        else:
            days = 90  # Quarterly review
        
        next_review = datetime.now() + timedelta(days=days)
        return next_review.strftime('%Y-%m-%d')
    
    def get_fairness_thresholds(self) -> Dict[str, float]:
        """Get current fairness thresholds"""
        return self.fairness_thresholds.copy()
    
    def update_fairness_thresholds(self, new_thresholds: Dict[str, float]):
        """Update fairness thresholds"""
        self.fairness_thresholds.update(new_thresholds)
        self.logger.info("Fairness thresholds updated")
    
    def get_fairness_history(self) -> List[Dict[str, Any]]:
        """Get fairness calculation history"""
        return self.fairness_history.copy()