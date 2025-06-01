import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch
import asyncio
from typing import Dict, List, Any

from app.services.ethics.bias_detection_service import BiasDetectionService
from app.services.ethics.demographic_classifier import DemographicClassifier
from app.services.ethics.fairness_metrics import FairnessMetrics


class TestBiasDetectionService:
    """Test suite for bias detection service using TDD methodology"""
    
    @pytest.fixture
    def bias_detection_service(self):
        """Create bias detection service instance for testing"""
        return BiasDetectionService()
    
    @pytest.fixture
    def sample_demographic_data(self):
        """Sample demographic classification data"""
        return {
            "age": "25-35",
            "gender": "female",
            "ethnicity": "asian",
            "body_type": "average",
            "skin_tone": "medium",
            "confidence": 0.92
        }
    
    @pytest.fixture
    def sample_performance_data(self):
        """Sample performance data across demographics"""
        return {
            "asian": {"quality": 8.5, "processing_time": 24.2, "success_rate": 0.96},
            "caucasian": {"quality": 8.7, "processing_time": 23.8, "success_rate": 0.97},
            "african": {"quality": 8.3, "processing_time": 25.1, "success_rate": 0.94},
            "hispanic": {"quality": 8.6, "processing_time": 24.5, "success_rate": 0.96},
            "middle_eastern": {"quality": 8.4, "processing_time": 24.8, "success_rate": 0.95},
            "indigenous": {"quality": 8.2, "processing_time": 25.3, "success_rate": 0.93}
        }

    # Test 1: Service Initialization
    def test_bias_detection_service_initialization(self, bias_detection_service):
        """Test bias detection service initializes correctly"""
        assert bias_detection_service is not None
        assert hasattr(bias_detection_service, 'demographic_classifier')
        assert hasattr(bias_detection_service, 'fairness_metrics')
        assert hasattr(bias_detection_service, 'bias_thresholds')
        assert bias_detection_service.bias_thresholds['demographic_parity_max_variance'] == 0.02
        assert bias_detection_service.bias_thresholds['equalized_odds_max_variance'] == 0.03
    
    # Test 2: Demographic Classification
    def test_demographic_classification_accuracy(self, bias_detection_service, sample_model_image):
        """Test demographic classification meets accuracy requirements"""
        result = bias_detection_service.classify_demographics(sample_model_image)
        
        # Verify classification completeness
        assert 'age' in result
        assert 'gender' in result  
        assert 'ethnicity' in result
        assert 'body_type' in result
        assert 'confidence' in result
        
        # Verify confidence threshold
        assert result['confidence'] >= 0.7
        
        # Verify valid classification categories
        valid_ethnicities = ['asian', 'caucasian', 'african', 'hispanic', 'middle_eastern', 'indigenous', 'mixed']
        assert result['ethnicity'] in valid_ethnicities
        
        valid_body_types = ['petite', 'average', 'athletic', 'plus_size', 'tall', 'curvy', 'lean', 'broad']
        assert result['body_type'] in valid_body_types

    # Test 3: Bias Detection in Performance
    def test_demographic_performance_bias_detection(self, bias_detection_service, sample_performance_data):
        """Test bias detection identifies performance disparities across demographics"""
        bias_report = bias_detection_service.detect_performance_bias(sample_performance_data)
        
        # Verify bias detection completeness
        assert 'demographic_parity' in bias_report
        assert 'quality_variance' in bias_report
        assert 'processing_time_variance' in bias_report
        assert 'success_rate_variance' in bias_report
        assert 'bias_detected' in bias_report
        assert 'bias_severity' in bias_report
        
        # Verify bias threshold enforcement
        max_quality_variance = max(sample_performance_data.values(), key=lambda x: x['quality'])['quality'] - \
                              min(sample_performance_data.values(), key=lambda x: x['quality'])['quality']
        
        if max_quality_variance > bias_detection_service.bias_thresholds['quality_max_variance']:
            assert bias_report['bias_detected'] is True
            assert bias_report['bias_severity'] in ['low', 'medium', 'high']
    
    # Test 4: Individual Fairness Measurement
    def test_individual_fairness_measurement(self, bias_detection_service, sample_model_image):
        """Test individual fairness scoring for similar inputs"""
        # Create slightly modified similar image
        similar_image = sample_model_image.copy()
        similar_array = np.array(similar_image)
        similar_array += np.random.normal(0, 5, similar_array.shape).astype(np.uint8)
        similar_image = Image.fromarray(np.clip(similar_array, 0, 255))
        
        fairness_score = bias_detection_service.measure_individual_fairness(
            sample_model_image, similar_image
        )
        
        # Verify fairness score properties
        assert 0.0 <= fairness_score <= 1.0
        assert fairness_score >= 0.95  # High similarity should result in high fairness
        
        # Test with dissimilar image
        dissimilar_image = Image.new('RGB', (512, 512), (255, 0, 0))  # Solid red
        low_fairness_score = bias_detection_service.measure_individual_fairness(
            sample_model_image, dissimilar_image
        )
        assert low_fairness_score < fairness_score

    # Test 5: Bias Threshold Enforcement
    def test_bias_threshold_enforcement(self, bias_detection_service):
        """Test bias detection enforces predefined thresholds"""
        # Test data with high bias
        biased_data = {
            "group_a": {"quality": 9.0, "success_rate": 0.98},
            "group_b": {"quality": 7.0, "success_rate": 0.85}  # Significant disparity
        }
        
        bias_result = bias_detection_service.detect_performance_bias(biased_data)
        assert bias_result['bias_detected'] is True
        assert bias_result['bias_severity'] == 'high'
        
        # Test data without bias
        fair_data = {
            "group_a": {"quality": 8.5, "success_rate": 0.96},
            "group_b": {"quality": 8.6, "success_rate": 0.97}  # Minimal disparity
        }
        
        fair_result = bias_detection_service.detect_performance_bias(fair_data)
        assert fair_result['bias_detected'] is False

    # Test 6: Real-time Bias Monitoring
    @pytest.mark.asyncio
    async def test_realtime_bias_monitoring(self, bias_detection_service, sample_model_image):
        """Test real-time bias monitoring with performance requirements"""
        start_time = asyncio.get_event_loop().time()
        
        monitoring_result = await bias_detection_service.monitor_bias_realtime(
            image=sample_model_image,
            transformation_metadata={"style": "editorial", "quality": 8.7}
        )
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Verify performance requirement (<2s)
        assert processing_time < 2.0
        
        # Verify monitoring result structure
        assert 'bias_score' in monitoring_result
        assert 'demographic_classification' in monitoring_result
        assert 'fairness_metrics' in monitoring_result
        assert 'recommendation' in monitoring_result
        assert 'timestamp' in monitoring_result
        
        # Verify bias score range
        assert 0.0 <= monitoring_result['bias_score'] <= 1.0

    # Test 7: Bias Mitigation Recommendations
    def test_bias_mitigation_recommendations(self, bias_detection_service):
        """Test bias mitigation recommendation system"""
        bias_data = {
            "ethnicity_bias": True,
            "gender_bias": False,
            "age_bias": True,
            "bias_severity": "medium",
            "affected_groups": ["asian", "elderly"]
        }
        
        recommendations = bias_detection_service.generate_mitigation_recommendations(bias_data)
        
        # Verify recommendation structure
        assert 'immediate_actions' in recommendations
        assert 'long_term_improvements' in recommendations
        assert 'alternative_algorithms' in recommendations
        assert 'monitoring_frequency' in recommendations
        
        # Verify actionable recommendations
        assert len(recommendations['immediate_actions']) > 0
        assert len(recommendations['alternative_algorithms']) > 0
        
        # Verify severity-based recommendations
        if bias_data['bias_severity'] == 'high':
            assert 'block_generation' in recommendations['immediate_actions']

    # Test 8: Performance Impact Assessment
    def test_performance_impact_assessment(self, bias_detection_service, sample_model_image):
        """Test bias detection performance impact on pipeline"""
        # Measure baseline processing time
        baseline_start = asyncio.get_event_loop().time()
        # Simulate model enhancement without bias detection
        await asyncio.sleep(0.1)  # Simulate processing
        baseline_time = asyncio.get_event_loop().time() - baseline_start
        
        # Measure with bias detection
        enhanced_start = asyncio.get_event_loop().time()
        bias_result = await bias_detection_service.monitor_bias_realtime(
            image=sample_model_image,
            transformation_metadata={"quality": 8.5}
        )
        enhanced_time = asyncio.get_event_loop().time() - enhanced_start
        
        # Verify performance overhead is acceptable (<10%)
        overhead_percentage = ((enhanced_time - baseline_time) / baseline_time) * 100
        assert overhead_percentage < 10.0

    # Test 9: Batch Bias Analysis
    def test_batch_bias_analysis(self, bias_detection_service):
        """Test batch bias analysis for historical data"""
        batch_data = [
            {"demographics": {"ethnicity": "asian", "gender": "female"}, "quality": 8.5},
            {"demographics": {"ethnicity": "caucasian", "gender": "female"}, "quality": 8.7},
            {"demographics": {"ethnicity": "african", "gender": "male"}, "quality": 8.3},
            {"demographics": {"ethnicity": "hispanic", "gender": "male"}, "quality": 8.6},
        ]
        
        batch_analysis = bias_detection_service.analyze_batch_bias(batch_data)
        
        # Verify batch analysis structure
        assert 'overall_bias_score' in batch_analysis
        assert 'demographic_breakdown' in batch_analysis
        assert 'trend_analysis' in batch_analysis
        assert 'recommendations' in batch_analysis
        
        # Verify demographic coverage
        ethnicities_found = set()
        for item in batch_data:
            ethnicities_found.add(item['demographics']['ethnicity'])
        
        assert len(batch_analysis['demographic_breakdown']) == len(ethnicities_found)

    # Test 10: Edge Case Handling
    def test_edge_case_handling(self, bias_detection_service):
        """Test bias detection handles edge cases gracefully"""
        # Test with empty image
        empty_image = Image.new('RGB', (1, 1), (0, 0, 0))
        result = bias_detection_service.classify_demographics(empty_image)
        assert result['confidence'] < 0.5  # Low confidence for empty image
        
        # Test with no demographic data
        empty_performance_data = {}
        bias_result = bias_detection_service.detect_performance_bias(empty_performance_data)
        assert bias_result['bias_detected'] is False
        assert bias_result['bias_severity'] == 'none'
        
        # Test with single demographic group
        single_group_data = {"group_a": {"quality": 8.5, "success_rate": 0.96}}
        single_result = bias_detection_service.detect_performance_bias(single_group_data)
        assert single_result['bias_detected'] is False  # Cannot detect bias with single group

    # Test 11: Bias Alert System
    def test_bias_alert_system(self, bias_detection_service):
        """Test bias alert system triggers appropriately"""
        # High bias scenario
        high_bias_data = {
            "group_a": {"quality": 9.5, "success_rate": 0.99},
            "group_b": {"quality": 6.0, "success_rate": 0.80}  # Significant disparity
        }
        
        alert = bias_detection_service.check_bias_alert(high_bias_data)
        
        assert alert['alert_triggered'] is True
        assert alert['severity'] == 'critical'
        assert alert['immediate_action_required'] is True
        assert 'notification_sent' in alert
        assert 'escalation_required' in alert

    # Test 12: Demographic Distribution Validation
    def test_demographic_distribution_validation(self, bias_detection_service):
        """Test validation of demographic representation in datasets"""
        # Test balanced distribution
        balanced_distribution = {
            "asian": 0.18, "caucasian": 0.16, "african": 0.17,
            "hispanic": 0.16, "middle_eastern": 0.16, "indigenous": 0.17
        }
        
        distribution_result = bias_detection_service.validate_demographic_distribution(
            balanced_distribution
        )
        
        assert distribution_result['is_balanced'] is True
        assert distribution_result['diversity_index'] >= 0.8
        
        # Test imbalanced distribution
        imbalanced_distribution = {
            "caucasian": 0.80, "asian": 0.10, "african": 0.05,
            "hispanic": 0.03, "middle_eastern": 0.01, "indigenous": 0.01
        }
        
        imbalanced_result = bias_detection_service.validate_demographic_distribution(
            imbalanced_distribution
        )
        
        assert imbalanced_result['is_balanced'] is False
        assert imbalanced_result['diversity_index'] < 0.5


class TestDemographicClassifier:
    """Test suite for demographic classifier component"""
    
    @pytest.fixture
    def demographic_classifier(self):
        """Create demographic classifier instance"""
        return DemographicClassifier()
    
    def test_age_classification_accuracy(self, demographic_classifier, sample_model_image):
        """Test age classification accuracy and categories"""
        age_result = demographic_classifier.classify_age(sample_model_image)
        
        valid_age_groups = ['child', 'teen', '18-25', '25-35', '35-45', '45-55', '55-65', '65+']
        assert age_result['age_group'] in valid_age_groups
        assert age_result['confidence'] >= 0.7
        
    def test_ethnicity_classification_comprehensive(self, demographic_classifier, sample_model_image):
        """Test comprehensive ethnicity classification"""
        ethnicity_result = demographic_classifier.classify_ethnicity(sample_model_image)
        
        expected_ethnicities = [
            'asian', 'caucasian', 'african', 'hispanic', 
            'middle_eastern', 'indigenous', 'pacific_islander', 'mixed'
        ]
        assert ethnicity_result['ethnicity'] in expected_ethnicities
        assert ethnicity_result['confidence'] >= 0.7
        assert 'sub_ethnicity' in ethnicity_result  # More specific classification
    
    def test_body_type_classification(self, demographic_classifier, sample_model_image):
        """Test body type classification accuracy"""
        body_type_result = demographic_classifier.classify_body_type(sample_model_image)
        
        valid_body_types = [
            'petite', 'average', 'athletic', 'plus_size', 
            'tall', 'curvy', 'lean', 'broad', 'muscular'
        ]
        assert body_type_result['body_type'] in valid_body_types
        assert body_type_result['confidence'] >= 0.7


class TestFairnessMetrics:
    """Test suite for fairness metrics calculations"""
    
    @pytest.fixture
    def fairness_metrics(self):
        """Create fairness metrics calculator"""
        return FairnessMetrics()
    
    def test_demographic_parity_calculation(self, fairness_metrics):
        """Test demographic parity metric calculation"""
        group_outcomes = {
            "group_a": {"positive_outcomes": 85, "total": 100},
            "group_b": {"positive_outcomes": 83, "total": 100},
            "group_c": {"positive_outcomes": 87, "total": 100}
        }
        
        parity_result = fairness_metrics.calculate_demographic_parity(group_outcomes)
        
        assert 'parity_achieved' in parity_result
        assert 'max_variance' in parity_result
        assert 'group_rates' in parity_result
        
        # Verify calculation accuracy
        expected_variance = 0.04  # 87% - 83% = 4%
        assert abs(parity_result['max_variance'] - expected_variance) < 0.01
        
    def test_equalized_odds_calculation(self, fairness_metrics):
        """Test equalized odds metric calculation"""
        classification_data = {
            "group_a": {"tp": 80, "fp": 5, "tn": 90, "fn": 25},
            "group_b": {"tp": 78, "fp": 7, "tn": 88, "fn": 27}
        }
        
        odds_result = fairness_metrics.calculate_equalized_odds(classification_data)
        
        assert 'equalized_odds_achieved' in odds_result
        assert 'tpr_variance' in odds_result  # True Positive Rate variance
        assert 'fpr_variance' in odds_result  # False Positive Rate variance
        assert 'group_metrics' in odds_result
        
    def test_individual_fairness_score(self, fairness_metrics, sample_model_image):
        """Test individual fairness scoring"""
        # Create similar individuals
        image1 = sample_model_image
        image2 = sample_model_image.copy()  # Identical for testing
        
        fairness_score = fairness_metrics.calculate_individual_fairness(
            image1, image2,
            outcome1=8.5, outcome2=8.6  # Similar outcomes for similar individuals
        )
        
        assert 0.0 <= fairness_score <= 1.0
        assert fairness_score >= 0.95  # High fairness for similar individuals


# Integration Tests
class TestBiasDetectionIntegration:
    """Integration tests for bias detection with existing AI services"""
    
    @pytest.mark.integration
    def test_model_enhancement_bias_integration(self, bias_detection_service, sample_model_image):
        """Test bias detection integration with model enhancement service"""
        from app.services.ai.model_enhancement_service import ModelEnhancementService
        
        enhancement_service = ModelEnhancementService()
        
        # Enhance image with bias monitoring
        enhancement_result = enhancement_service.enhance_model(sample_model_image)
        bias_result = bias_detection_service.monitor_bias_realtime(
            image=sample_model_image,
            transformation_metadata=enhancement_result
        )
        
        # Verify integration doesn't break existing functionality
        assert enhancement_result['enhanced_image'] is not None
        assert enhancement_result['quality_score'] > 0
        
        # Verify bias monitoring provides results
        assert bias_result['bias_score'] is not None
        assert bias_result['demographic_classification'] is not None
    
    @pytest.mark.integration  
    def test_garment_optimization_bias_integration(self, bias_detection_service, sample_model_image):
        """Test bias detection integration with garment optimization"""
        from app.services.ai.garment_optimization_service import GarmentOptimizationService
        
        garment_service = GarmentOptimizationService()
        
        # Optimize garment with bias monitoring
        optimization_result = garment_service.optimize_garment(sample_model_image)
        bias_result = bias_detection_service.monitor_bias_realtime(
            image=sample_model_image,
            transformation_metadata=optimization_result
        )
        
        # Verify no bias introduced in garment optimization
        assert bias_result['bias_score'] < 0.3  # Low bias threshold
        assert 'garment_bias_analysis' in bias_result


# Performance Tests
class TestBiasDetectionPerformance:
    """Performance tests for bias detection system"""
    
    @pytest.mark.performance
    def test_bias_detection_speed_requirements(self, bias_detection_service, sample_model_image):
        """Test bias detection meets speed requirements (<2s)"""
        import time
        
        start_time = time.time()
        
        for _ in range(5):  # Test multiple iterations
            bias_detection_service.classify_demographics(sample_model_image)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 5
        
        assert avg_time < 2.0  # Must be under 2 seconds per classification
        
    @pytest.mark.performance
    def test_concurrent_bias_detection(self, bias_detection_service, sample_model_image):
        """Test concurrent bias detection performance"""
        import asyncio
        
        async def bias_detection_task():
            return await bias_detection_service.monitor_bias_realtime(
                image=sample_model_image,
                transformation_metadata={"quality": 8.5}
            )
        
        # Test 5 concurrent bias detections
        tasks = [bias_detection_task() for _ in range(5)]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        # Verify all completed successfully
        assert len(results) == 5
        assert all('bias_score' in result for result in results)
        
        # Verify reasonable concurrent performance
        total_time = end_time - start_time
        assert total_time < 10.0  # 5 concurrent tasks in under 10 seconds