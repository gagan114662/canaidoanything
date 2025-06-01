import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch
import asyncio
from typing import Dict, List, Any

from app.services.ethics.content_safety_service import ContentSafetyService
from app.services.ethics.content_moderator import ContentModerator
from app.services.ethics.ethics_engine import EthicsEngine


class TestContentSafetyService:
    """Test suite for content safety service using TDD methodology"""
    
    @pytest.fixture
    def content_safety_service(self):
        """Create content safety service instance for testing"""
        return ContentSafetyService()
    
    @pytest.fixture
    def sample_safe_content(self):
        """Sample safe content for testing"""
        return {
            "image_content": "professional_fashion_photo",
            "style_description": "elegant evening wear photography",
            "target_audience": "adult_fashion_consumers",
            "commercial_use": True,
            "brand_context": "luxury_fashion"
        }
    
    @pytest.fixture
    def sample_inappropriate_content(self):
        """Sample inappropriate content for testing"""
        return {
            "image_content": "revealing_clothing",
            "style_description": "inappropriate suggestive styling",
            "target_audience": "general_public",
            "commercial_use": True,
            "brand_context": "unknown"
        }

    # Test 1: Service Initialization
    def test_content_safety_service_initialization(self, content_safety_service):
        """Test content safety service initializes correctly"""
        assert content_safety_service is not None
        assert hasattr(content_safety_service, 'content_moderator')
        assert hasattr(content_safety_service, 'ethics_engine')
        assert hasattr(content_safety_service, 'safety_thresholds')
        
        # Verify default safety thresholds
        assert content_safety_service.safety_thresholds['appropriateness_min'] == 0.95
        assert content_safety_service.safety_thresholds['professional_standard_min'] == 0.90
        assert content_safety_service.safety_thresholds['age_appropriate_min'] == 0.95

    # Test 2: Content Appropriateness Validation
    def test_content_appropriateness_validation(self, content_safety_service, sample_model_image):
        """Test content appropriateness validation"""
        validation_result = content_safety_service.validate_content_appropriateness(
            image=sample_model_image,
            metadata={"style": "professional", "context": "business"}
        )
        
        # Verify validation structure
        assert 'is_appropriate' in validation_result
        assert 'appropriateness_score' in validation_result
        assert 'violation_categories' in validation_result
        assert 'recommendations' in validation_result
        assert 'confidence' in validation_result
        
        # Verify appropriateness score range
        assert 0.0 <= validation_result['appropriateness_score'] <= 1.0
        
        # Verify confidence threshold
        assert validation_result['confidence'] >= 0.7

    # Test 3: Age-Appropriate Content Verification
    def test_age_appropriate_content_verification(self, content_safety_service, sample_model_image):
        """Test age-appropriate content verification"""
        # Test appropriate content
        appropriate_result = content_safety_service.verify_age_appropriate_content(
            image=sample_model_image,
            target_age_group="all_ages",
            content_type="fashion_photography"
        )
        
        assert 'age_appropriate' in appropriate_result
        assert 'age_rating' in appropriate_result
        assert 'restriction_reasons' in appropriate_result
        assert 'recommended_age_group' in appropriate_result
        
        # Verify age rating categories
        valid_age_ratings = ['G', 'PG', 'PG-13', 'R', 'Adult']
        assert appropriate_result['age_rating'] in valid_age_ratings

    # Test 4: Professional Standard Compliance
    def test_professional_standard_compliance(self, content_safety_service, sample_safe_content):
        """Test professional standard compliance validation"""
        compliance_result = content_safety_service.validate_professional_standards(sample_safe_content)
        
        # Verify compliance structure
        assert 'meets_standards' in compliance_result
        assert 'compliance_score' in compliance_result
        assert 'standard_violations' in compliance_result
        assert 'improvement_suggestions' in compliance_result
        
        # Verify compliance score
        assert 0.0 <= compliance_result['compliance_score'] <= 1.0
        
        # For safe content, should meet standards
        if compliance_result['compliance_score'] >= 0.9:
            assert compliance_result['meets_standards'] is True

    # Test 5: Inappropriate Content Detection
    def test_inappropriate_content_detection(self, content_safety_service, sample_inappropriate_content):
        """Test detection of inappropriate content"""
        detection_result = content_safety_service.detect_inappropriate_content(sample_inappropriate_content)
        
        # Verify detection structure
        assert 'inappropriate_detected' in detection_result
        assert 'inappropriate_categories' in detection_result
        assert 'severity_level' in detection_result
        assert 'blocking_required' in detection_result
        assert 'moderation_actions' in detection_result
        
        # Verify severity levels
        valid_severity_levels = ['low', 'medium', 'high', 'critical']
        assert detection_result['severity_level'] in valid_severity_levels
        
        # For inappropriate content, should be detected
        if "inappropriate" in str(sample_inappropriate_content).lower():
            assert detection_result['inappropriate_detected'] is True

    # Test 6: Ethics Guideline Enforcement
    def test_ethics_guideline_enforcement(self, content_safety_service):
        """Test ethics guideline enforcement"""
        ethics_scenario = {
            "content_type": "fashion_photography",
            "representation": "diverse_models",
            "message": "body_positive",
            "commercial_intent": "fashion_brand_promotion"
        }
        
        ethics_result = content_safety_service.enforce_ethics_guidelines(ethics_scenario)
        
        # Verify ethics enforcement structure
        assert 'ethics_compliant' in ethics_result
        assert 'ethics_score' in ethics_result
        assert 'guideline_violations' in ethics_result
        assert 'ethical_recommendations' in ethics_result
        assert 'approval_status' in ethics_result
        
        # Verify ethics score
        assert 0.0 <= ethics_result['ethics_score'] <= 1.0

    # Test 7: Multi-Modal Content Analysis
    def test_multimodal_content_analysis(self, content_safety_service, sample_model_image):
        """Test multi-modal content analysis (image + metadata)"""
        multimodal_data = {
            "image": sample_model_image,
            "text_description": "Professional business attire photography",
            "style_tags": ["professional", "business", "formal"],
            "intended_use": "corporate_website"
        }
        
        analysis_result = content_safety_service.analyze_multimodal_content(multimodal_data)
        
        # Verify analysis structure
        assert 'overall_safety_score' in analysis_result
        assert 'image_analysis' in analysis_result
        assert 'text_analysis' in analysis_result
        assert 'contextual_analysis' in analysis_result
        assert 'consistency_check' in analysis_result
        
        # Verify overall safety score
        assert 0.0 <= analysis_result['overall_safety_score'] <= 1.0

    # Test 8: Real-time Content Moderation
    @pytest.mark.asyncio
    async def test_realtime_content_moderation(self, content_safety_service, sample_model_image):
        """Test real-time content moderation with performance requirements"""
        start_time = asyncio.get_event_loop().time()
        
        moderation_result = await content_safety_service.moderate_content_realtime(
            image=sample_model_image,
            content_metadata={"style": "fashion", "urgency": "high"}
        )
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Verify performance requirement (<1s for real-time moderation)
        assert processing_time < 1.0
        
        # Verify moderation result structure
        assert 'moderation_decision' in moderation_result
        assert 'safety_score' in moderation_result
        assert 'automated_actions' in moderation_result
        assert 'human_review_required' in moderation_result
        assert 'timestamp' in moderation_result

    # Test 9: Content Safety Scoring
    def test_content_safety_scoring(self, content_safety_service):
        """Test comprehensive content safety scoring"""
        content_examples = [
            {"type": "professional_headshot", "expected_score": 0.95},
            {"type": "artistic_fashion", "expected_score": 0.85},
            {"type": "casual_wear", "expected_score": 0.90},
            {"type": "inappropriate_content", "expected_score": 0.20}
        ]
        
        for example in content_examples:
            safety_score = content_safety_service.calculate_safety_score(example)
            
            # Verify score is in valid range
            assert 0.0 <= safety_score <= 1.0
            
            # Verify scores align with content type expectations
            expected = example["expected_score"]
            if example["type"] == "inappropriate_content":
                assert safety_score < 0.5
            elif example["type"] == "professional_headshot":
                assert safety_score > 0.8

    # Test 10: Safety Threshold Enforcement
    def test_safety_threshold_enforcement(self, content_safety_service):
        """Test safety threshold enforcement and blocking"""
        test_scenarios = [
            {"safety_score": 0.98, "should_block": False},
            {"safety_score": 0.85, "should_block": False},
            {"safety_score": 0.50, "should_block": True},
            {"safety_score": 0.20, "should_block": True}
        ]
        
        for scenario in test_scenarios:
            enforcement_result = content_safety_service.enforce_safety_thresholds(
                safety_score=scenario["safety_score"]
            )
            
            assert 'content_blocked' in enforcement_result
            assert 'threshold_violations' in enforcement_result
            assert 'enforcement_actions' in enforcement_result
            
            # Verify blocking behavior matches expectations
            if scenario["should_block"]:
                assert enforcement_result['content_blocked'] is True
            else:
                assert enforcement_result['content_blocked'] is False

    # Test 11: Human Review Queue Management
    def test_human_review_queue_management(self, content_safety_service):
        """Test human review queue for edge cases"""
        edge_case_content = {
            "content_type": "artistic_nudity",
            "context": "fine_art_photography",
            "safety_score": 0.65,  # Borderline score
            "cultural_sensitivity": 0.70
        }
        
        queue_result = content_safety_service.queue_for_human_review(edge_case_content)
        
        # Verify queue structure
        assert 'queued_for_review' in queue_result
        assert 'review_priority' in queue_result
        assert 'estimated_review_time' in queue_result
        assert 'interim_action' in queue_result
        assert 'reviewer_categories' in queue_result
        
        # Verify priority levels
        valid_priorities = ['low', 'medium', 'high', 'critical']
        assert queue_result['review_priority'] in valid_priorities

    # Test 12: Content Safety Reporting
    def test_content_safety_reporting(self, content_safety_service):
        """Test content safety reporting and analytics"""
        # Simulate processed content history
        content_history = [
            {"safety_score": 0.95, "blocked": False, "timestamp": "2024-01-01T10:00:00Z"},
            {"safety_score": 0.30, "blocked": True, "timestamp": "2024-01-01T10:05:00Z"},
            {"safety_score": 0.88, "blocked": False, "timestamp": "2024-01-01T10:10:00Z"}
        ]
        
        report = content_safety_service.generate_safety_report(content_history)
        
        # Verify report structure
        assert 'total_content_processed' in report
        assert 'safety_score_distribution' in report
        assert 'blocking_rate' in report
        assert 'violation_categories' in report
        assert 'trends_analysis' in report
        assert 'recommendations' in report
        
        # Verify calculations
        assert report['total_content_processed'] == len(content_history)
        assert 0.0 <= report['blocking_rate'] <= 1.0

    # Test 13: Cultural Content Safety Integration
    def test_cultural_content_safety_integration(self, content_safety_service):
        """Test integration with cultural sensitivity for content safety"""
        cultural_content = {
            "cultural_elements": ["traditional_garment", "cultural_ceremony"],
            "cultural_appropriation_risk": 0.7,
            "cultural_context": "fashion_photography",
            "cultural_consultation": False
        }
        
        cultural_safety_result = content_safety_service.assess_cultural_content_safety(cultural_content)
        
        # Verify cultural safety assessment
        assert 'culturally_safe' in cultural_safety_result
        assert 'cultural_safety_score' in cultural_safety_result
        assert 'cultural_violations' in cultural_safety_result
        assert 'cultural_recommendations' in cultural_safety_result
        
        # High appropriation risk should affect safety
        if cultural_content['cultural_appropriation_risk'] > 0.6:
            assert cultural_safety_result['cultural_safety_score'] < 0.8

    # Test 14: Content Safety Edge Cases
    def test_content_safety_edge_cases(self, content_safety_service):
        """Test content safety handling of edge cases"""
        edge_cases = [
            {"case": "empty_content", "data": {}},
            {"case": "minimal_data", "data": {"type": "unknown"}},
            {"case": "corrupted_metadata", "data": {"style": None, "context": ""}},
            {"case": "extreme_values", "data": {"safety_score": 1.5, "inappropriate": -0.1}}
        ]
        
        for edge_case in edge_cases:
            result = content_safety_service.handle_edge_case_content(edge_case["data"])
            
            # Verify graceful handling
            assert 'handled' in result
            assert 'fallback_used' in result
            assert 'safety_decision' in result
            
            # Edge cases should default to safe/conservative handling
            assert result['handled'] is True

    # Test 15: Performance and Scalability
    def test_content_safety_performance(self, content_safety_service, sample_model_image):
        """Test content safety service performance under load"""
        import time
        
        # Test batch processing performance
        batch_content = [
            {"image": sample_model_image, "metadata": {"style": f"test_{i}"}}
            for i in range(10)
        ]
        
        start_time = time.time()
        batch_results = content_safety_service.process_content_batch(batch_content)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify performance (should process 10 items in reasonable time)
        assert processing_time < 5.0  # 5 seconds for 10 items
        
        # Verify all items processed
        assert len(batch_results) == len(batch_content)
        assert all('safety_score' in result for result in batch_results)

    # Test 16: Safety Configuration Management
    def test_safety_configuration_management(self, content_safety_service):
        """Test safety configuration and threshold management"""
        # Get current configuration
        current_config = content_safety_service.get_safety_configuration()
        
        # Verify configuration structure
        assert 'safety_thresholds' in current_config
        assert 'moderation_rules' in current_config
        assert 'review_criteria' in current_config
        
        # Test configuration updates
        new_thresholds = {
            'appropriateness_min': 0.98,
            'professional_standard_min': 0.92
        }
        
        update_result = content_safety_service.update_safety_configuration(
            {'safety_thresholds': new_thresholds}
        )
        
        assert update_result['success'] is True
        
        # Verify thresholds were updated
        updated_config = content_safety_service.get_safety_configuration()
        assert updated_config['safety_thresholds']['appropriateness_min'] == 0.98


class TestContentModerator:
    """Test suite for content moderator component"""
    
    @pytest.fixture
    def content_moderator(self):
        """Create content moderator instance"""
        return ContentModerator()
    
    def test_content_moderator_initialization(self, content_moderator):
        """Test content moderator initializes correctly"""
        assert content_moderator is not None
        assert hasattr(content_moderator, 'moderation_rules')
        assert hasattr(content_moderator, 'violation_detectors')
    
    def test_inappropriate_content_detection(self, content_moderator, sample_model_image):
        """Test inappropriate content detection algorithms"""
        detection_result = content_moderator.detect_inappropriate_content(sample_model_image)
        
        assert 'inappropriate_detected' in detection_result
        assert 'confidence' in detection_result
        assert 'violation_types' in detection_result
    
    def test_professional_context_validation(self, content_moderator):
        """Test professional context validation"""
        professional_content = {
            "setting": "office_environment",
            "attire": "business_professional",
            "pose": "formal_portrait"
        }
        
        validation_result = content_moderator.validate_professional_context(professional_content)
        
        assert 'is_professional' in validation_result
        assert 'professionalism_score' in validation_result
        assert validation_result['professionalism_score'] >= 0.8


class TestEthicsEngine:
    """Test suite for ethics engine component"""
    
    @pytest.fixture
    def ethics_engine(self):
        """Create ethics engine instance"""
        return EthicsEngine()
    
    def test_ethics_engine_initialization(self, ethics_engine):
        """Test ethics engine initializes with guidelines"""
        assert ethics_engine is not None
        assert hasattr(ethics_engine, 'ethics_guidelines')
        assert hasattr(ethics_engine, 'ethical_principles')
    
    def test_ethical_guideline_enforcement(self, ethics_engine):
        """Test ethical guideline enforcement"""
        ethical_scenario = {
            "representation": "inclusive_diverse",
            "message": "body_positive",
            "exploitation": "none",
            "consent": "verified"
        }
        
        enforcement_result = ethics_engine.enforce_guidelines(ethical_scenario)
        
        assert 'compliant' in enforcement_result
        assert 'ethics_score' in enforcement_result
        assert 'violations' in enforcement_result
    
    def test_bias_detection_in_content(self, ethics_engine):
        """Test bias detection in content representation"""
        content_representation = {
            "demographic_diversity": 0.8,
            "inclusive_messaging": True,
            "stereotypical_elements": False,
            "body_type_diversity": 0.7
        }
        
        bias_result = ethics_engine.detect_representation_bias(content_representation)
        
        assert 'bias_detected' in bias_result
        assert 'bias_score' in bias_result
        assert 'bias_categories' in bias_result


# Integration Tests
class TestContentSafetyIntegration:
    """Integration tests for content safety with other services"""
    
    @pytest.mark.integration
    def test_content_safety_bias_detection_integration(self, content_safety_service, sample_model_image):
        """Test content safety integration with bias detection"""
        from app.services.ethics.bias_detection_service import BiasDetectionService
        
        bias_service = BiasDetectionService()
        
        # Run content safety with bias detection
        safety_result = content_safety_service.validate_content_appropriateness(
            image=sample_model_image,
            metadata={"style": "professional", "demographics_check": True}
        )
        
        bias_result = bias_service.classify_demographics(sample_model_image)
        
        # Verify integration doesn't break functionality
        assert safety_result['is_appropriate'] is not None
        assert bias_result['confidence'] is not None
    
    @pytest.mark.integration
    def test_content_safety_cultural_sensitivity_integration(self, content_safety_service, sample_model_image):
        """Test content safety integration with cultural sensitivity"""
        from app.services.ethics.cultural_sensitivity_service import CulturalSensitivityService
        
        cultural_service = CulturalSensitivityService()
        
        # Run content safety with cultural sensitivity
        safety_result = content_safety_service.validate_content_appropriateness(
            image=sample_model_image,
            metadata={"style": "traditional", "cultural_check": True}
        )
        
        cultural_result = cultural_service.recognize_cultural_garments(sample_model_image)
        
        # Verify integration provides comprehensive safety assessment
        assert safety_result['appropriateness_score'] is not None
        assert cultural_result['appropriation_risk'] is not None


# Performance Tests
class TestContentSafetyPerformance:
    """Performance tests for content safety system"""
    
    @pytest.mark.performance
    def test_content_moderation_speed(self, content_safety_service, sample_model_image):
        """Test content moderation speed requirements"""
        import time
        
        start_time = time.time()
        
        for _ in range(10):  # Test multiple moderations
            content_safety_service.validate_content_appropriateness(
                image=sample_model_image,
                metadata={"style": "test"}
            )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        assert avg_time < 1.0  # Must be under 1 second per moderation
        
    @pytest.mark.performance
    def test_concurrent_content_safety_processing(self, content_safety_service, sample_model_image):
        """Test concurrent content safety processing"""
        import asyncio
        
        async def safety_check_task():
            return await content_safety_service.moderate_content_realtime(
                image=sample_model_image,
                content_metadata={"style": "test"}
            )
        
        # Test 5 concurrent safety checks
        tasks = [safety_check_task() for _ in range(5)]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        # Verify all completed successfully
        assert len(results) == 5
        assert all('safety_score' in result for result in results)
        
        # Verify reasonable concurrent performance
        total_time = end_time - start_time
        assert total_time < 3.0  # 5 concurrent checks in under 3 seconds