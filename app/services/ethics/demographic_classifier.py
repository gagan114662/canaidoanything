"""
Demographic Classifier for Bias Detection

This module provides comprehensive demographic classification capabilities
for age, gender, ethnicity, and body type to support bias detection.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import cv2
import torch
from dataclasses import dataclass

try:
    import mediapipe as mp
    import insightface
    from facenet_pytorch import MTCNN, InceptionResnetV1
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False


@dataclass
class DemographicClassification:
    """Structured demographic classification result"""
    age: str
    gender: str
    ethnicity: str
    body_type: str
    skin_tone: str
    confidence: float
    detailed_scores: Dict[str, float]
    requires_review: bool = False


class DemographicClassifier:
    """
    Advanced demographic classifier for bias detection
    
    Provides comprehensive demographic classification including age, gender,
    ethnicity, body type, and skin tone analysis with confidence scoring.
    """
    
    def __init__(self):
        """Initialize demographic classifier with models"""
        self.logger = logging.getLogger(__name__)
        self.models_loaded = False
        self.use_advanced_models = ADVANCED_MODELS_AVAILABLE
        
        # Classification categories
        self.age_groups = ['child', 'teen', '18-25', '25-35', '35-45', '45-55', '55-65', '65+']
        self.ethnicities = [
            'asian', 'caucasian', 'african', 'hispanic', 
            'middle_eastern', 'indigenous', 'pacific_islander', 'mixed'
        ]
        self.body_types = [
            'petite', 'average', 'athletic', 'plus_size', 
            'tall', 'curvy', 'lean', 'broad', 'muscular'
        ]
        self.skin_tones = ['very_light', 'light', 'medium_light', 'medium', 'medium_dark', 'dark', 'very_dark']
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'age': 0.7,
            'gender': 0.8,
            'ethnicity': 0.6,  # Lower threshold due to complexity
            'body_type': 0.7,
            'overall_min': 0.7
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize demographic classification models"""
        try:
            if self.use_advanced_models:
                self._initialize_advanced_models()
            else:
                self._initialize_fallback_models()
            
            self.models_loaded = True
            self.logger.info("Demographic classification models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load demographic models: {str(e)}")
            self._initialize_fallback_models()
    
    def _initialize_advanced_models(self):
        """Initialize advanced AI models for demographic classification"""
        # Initialize MediaPipe for pose and face detection
        self.mp_pose = mp.solutions.pose
        self.mp_face_detection = mp.solutions.face_detection
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.7
        )
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        # Initialize InsightFace for detailed face analysis
        try:
            self.face_analyzer = insightface.app.FaceAnalysis()
            self.face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            self.logger.warning(f"InsightFace initialization failed: {str(e)}")
            self.face_analyzer = None
        
        # Initialize MTCNN for face detection fallback
        try:
            self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            self.logger.warning(f"MTCNN initialization failed: {str(e)}")
            self.mtcnn = None
        
        self.logger.info("Advanced demographic models initialized")
    
    def _initialize_fallback_models(self):
        """Initialize fallback models using OpenCV"""
        self.logger.warning("Using fallback demographic classification models")
        self.use_advanced_models = False
        
        # Load OpenCV classifiers
        try:
            # Face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Eye detection for additional verification
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load OpenCV classifiers: {str(e)}")
    
    def load_models(self):
        """Public method to load demographic classification models"""
        if not self.models_loaded:
            self._initialize_models()
    
    def use_fallback_models(self):
        """Switch to fallback models"""
        self.use_advanced_models = False
        self._initialize_fallback_models()
    
    def classify_comprehensive(self, image: Image.Image) -> Dict[str, Any]:
        """
        Comprehensive demographic classification
        
        Args:
            image: PIL Image containing person to classify
            
        Returns:
            Complete demographic classification with confidence scores
        """
        try:
            # Convert PIL to CV2 format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rgb_image = np.array(image)
            
            # Perform individual classifications
            age_result = self.classify_age(image)
            gender_result = self.classify_gender(image)
            ethnicity_result = self.classify_ethnicity(image)
            body_type_result = self.classify_body_type(image)
            skin_tone_result = self.classify_skin_tone(image)
            
            # Calculate overall confidence
            individual_confidences = [
                age_result.get('confidence', 0.0),
                gender_result.get('confidence', 0.0),
                ethnicity_result.get('confidence', 0.0),
                body_type_result.get('confidence', 0.0),
                skin_tone_result.get('confidence', 0.0)
            ]
            
            overall_confidence = np.mean(individual_confidences)
            
            # Compile comprehensive result
            comprehensive_result = {
                'age': age_result.get('age_group', 'unknown'),
                'gender': gender_result.get('gender', 'unknown'),
                'ethnicity': ethnicity_result.get('ethnicity', 'unknown'),
                'body_type': body_type_result.get('body_type', 'unknown'),
                'skin_tone': skin_tone_result.get('skin_tone', 'unknown'),
                'confidence': overall_confidence,
                'detailed_scores': {
                    'age_confidence': age_result.get('confidence', 0.0),
                    'gender_confidence': gender_result.get('confidence', 0.0),
                    'ethnicity_confidence': ethnicity_result.get('confidence', 0.0),
                    'body_type_confidence': body_type_result.get('confidence', 0.0),
                    'skin_tone_confidence': skin_tone_result.get('confidence', 0.0)
                },
                'requires_review': overall_confidence < self.confidence_thresholds['overall_min'],
                'classification_method': 'advanced' if self.use_advanced_models else 'fallback'
            }
            
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive classification failed: {str(e)}")
            return self._create_fallback_classification()
    
    def classify_age(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify age group from facial features
        
        Args:
            image: PIL Image containing face
            
        Returns:
            Age classification with confidence
        """
        try:
            if self.use_advanced_models and self.face_analyzer:
                return self._classify_age_advanced(image)
            else:
                return self._classify_age_fallback(image)
                
        except Exception as e:
            self.logger.error(f"Age classification failed: {str(e)}")
            return {'age_group': 'unknown', 'confidence': 0.0}
    
    def _classify_age_advanced(self, image: Image.Image) -> Dict[str, Any]:
        """Advanced age classification using InsightFace"""
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            faces = self.face_analyzer.get(cv_image)
            
            if faces:
                face = faces[0]  # Use first detected face
                age = face.age
                
                # Map numerical age to age groups
                if age < 13:
                    age_group = 'child'
                elif age < 18:
                    age_group = 'teen'
                elif age < 25:
                    age_group = '18-25'
                elif age < 35:
                    age_group = '25-35'
                elif age < 45:
                    age_group = '35-45'
                elif age < 55:
                    age_group = '45-55'
                elif age < 65:
                    age_group = '55-65'
                else:
                    age_group = '65+'
                
                # Confidence based on face detection quality
                confidence = min(float(face.det_score), 1.0)
                
                return {
                    'age_group': age_group,
                    'estimated_age': int(age),
                    'confidence': confidence
                }
            else:
                return {'age_group': 'unknown', 'confidence': 0.0}
                
        except Exception as e:
            self.logger.error(f"Advanced age classification failed: {str(e)}")
            return self._classify_age_fallback(image)
    
    def _classify_age_fallback(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback age classification using basic features"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use basic heuristics for age estimation
                # This is a simplified approach - in production, use trained models
                face_area = faces[0][2] * faces[0][3]  # width * height
                relative_face_size = face_area / (gray.shape[0] * gray.shape[1])
                
                # Basic age estimation based on relative face size and features
                if relative_face_size > 0.3:
                    age_group = 'child'  # Large face relative to image
                else:
                    age_group = '25-35'  # Default adult classification
                
                return {
                    'age_group': age_group,
                    'confidence': 0.6,  # Lower confidence for fallback
                    'method': 'fallback'
                }
            else:
                return {'age_group': 'unknown', 'confidence': 0.0}
                
        except Exception as e:
            self.logger.error(f"Fallback age classification failed: {str(e)}")
            return {'age_group': 'unknown', 'confidence': 0.0}
    
    def classify_gender(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify gender from facial and body features
        
        Args:
            image: PIL Image containing person
            
        Returns:
            Gender classification with confidence
        """
        try:
            if self.use_advanced_models and self.face_analyzer:
                return self._classify_gender_advanced(image)
            else:
                return self._classify_gender_fallback(image)
                
        except Exception as e:
            self.logger.error(f"Gender classification failed: {str(e)}")
            return {'gender': 'unknown', 'confidence': 0.0}
    
    def _classify_gender_advanced(self, image: Image.Image) -> Dict[str, Any]:
        """Advanced gender classification using InsightFace"""
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            faces = self.face_analyzer.get(cv_image)
            
            if faces:
                face = faces[0]
                gender = 'female' if face.gender == 0 else 'male'
                confidence = min(float(face.det_score), 1.0)
                
                return {
                    'gender': gender,
                    'confidence': confidence,
                    'method': 'advanced'
                }
            else:
                return {'gender': 'unknown', 'confidence': 0.0}
                
        except Exception as e:
            self.logger.error(f"Advanced gender classification failed: {str(e)}")
            return self._classify_gender_fallback(image)
    
    def _classify_gender_fallback(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback gender classification using basic features"""
        # Simplified fallback - in production, use trained gender classifiers
        return {
            'gender': 'unknown',
            'confidence': 0.3,
            'method': 'fallback'
        }
    
    def classify_ethnicity(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify ethnicity with cultural sensitivity
        
        Args:
            image: PIL Image containing person
            
        Returns:
            Ethnicity classification with confidence and sub-categories
        """
        try:
            if self.use_advanced_models:
                return self._classify_ethnicity_advanced(image)
            else:
                return self._classify_ethnicity_fallback(image)
                
        except Exception as e:
            self.logger.error(f"Ethnicity classification failed: {str(e)}")
            return {'ethnicity': 'unknown', 'confidence': 0.0}
    
    def _classify_ethnicity_advanced(self, image: Image.Image) -> Dict[str, Any]:
        """Advanced ethnicity classification with sensitivity"""
        try:
            # This is a sensitive area - using general facial feature analysis
            # In production, ensure diverse training data and bias testing
            
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Use face analysis for general features
            if self.face_analyzer:
                faces = self.face_analyzer.get(cv_image)
                
                if faces:
                    face = faces[0]
                    
                    # Use embedding similarity to reference ethnicities
                    # This is a placeholder - implement proper ethnicity classification
                    ethnicity = self._analyze_facial_features_for_ethnicity(face)
                    
                    return {
                        'ethnicity': ethnicity,
                        'confidence': 0.7,  # Conservative confidence
                        'sub_ethnicity': 'general',
                        'method': 'advanced'
                    }
            
            return {'ethnicity': 'unknown', 'confidence': 0.0}
            
        except Exception as e:
            self.logger.error(f"Advanced ethnicity classification failed: {str(e)}")
            return self._classify_ethnicity_fallback(image)
    
    def _analyze_facial_features_for_ethnicity(self, face) -> str:
        """Analyze facial features for ethnicity (placeholder implementation)"""
        # This is a sensitive implementation area
        # In production, ensure:
        # 1. Diverse training data
        # 2. Regular bias testing
        # 3. Cultural sensitivity review
        # 4. Explicit consent for ethnicity classification
        
        # Placeholder implementation returning general classification
        return 'general'  # Conservative default
    
    def _classify_ethnicity_fallback(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback ethnicity classification"""
        # Conservative fallback approach
        return {
            'ethnicity': 'general',
            'confidence': 0.3,
            'method': 'fallback',
            'note': 'ethnicity_classification_requires_advanced_models'
        }
    
    def classify_body_type(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify body type from pose and proportions
        
        Args:
            image: PIL Image containing full or partial body
            
        Returns:
            Body type classification with confidence
        """
        try:
            if self.use_advanced_models:
                return self._classify_body_type_advanced(image)
            else:
                return self._classify_body_type_fallback(image)
                
        except Exception as e:
            self.logger.error(f"Body type classification failed: {str(e)}")
            return {'body_type': 'unknown', 'confidence': 0.0}
    
    def _classify_body_type_advanced(self, image: Image.Image) -> Dict[str, Any]:
        """Advanced body type classification using pose detection"""
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Use MediaPipe pose detection
            results = self.pose_detector.process(rgb_image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Calculate body proportions
                body_analysis = self._analyze_body_proportions(landmarks, image.size)
                
                return {
                    'body_type': body_analysis['body_type'],
                    'confidence': body_analysis['confidence'],
                    'proportions': body_analysis['proportions'],
                    'method': 'advanced'
                }
            else:
                return self._classify_body_type_fallback(image)
                
        except Exception as e:
            self.logger.error(f"Advanced body type classification failed: {str(e)}")
            return self._classify_body_type_fallback(image)
    
    def _analyze_body_proportions(self, landmarks, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze body proportions from pose landmarks"""
        try:
            # Get key body points
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Calculate proportions
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            hip_width = abs(left_hip.x - right_hip.x)
            torso_height = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2)
            
            # Basic body type classification based on ratios
            if shoulder_width > hip_width * 1.1:
                body_type = 'athletic'
            elif hip_width > shoulder_width * 1.1:
                body_type = 'curvy'
            elif torso_height > 0.4:  # Tall relative proportions
                body_type = 'tall'
            else:
                body_type = 'average'
            
            confidence = 0.8 if torso_height > 0.2 else 0.6  # Lower confidence for partial body
            
            return {
                'body_type': body_type,
                'confidence': confidence,
                'proportions': {
                    'shoulder_width': shoulder_width,
                    'hip_width': hip_width,
                    'torso_height': torso_height
                }
            }
            
        except Exception as e:
            self.logger.error(f"Body proportion analysis failed: {str(e)}")
            return {'body_type': 'average', 'confidence': 0.5, 'proportions': {}}
    
    def _classify_body_type_fallback(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback body type classification"""
        return {
            'body_type': 'average',
            'confidence': 0.4,
            'method': 'fallback'
        }
    
    def classify_skin_tone(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify skin tone for bias detection
        
        Args:
            image: PIL Image containing person
            
        Returns:
            Skin tone classification
        """
        try:
            # Convert to appropriate color space for skin analysis
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Detect face region for skin tone analysis
            faces = self.face_cascade.detectMultiScale(
                cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY), 1.1, 4
            )
            
            if len(faces) > 0:
                # Extract face region
                x, y, w, h = faces[0]
                face_region = cv_image[y:y+h, x:x+w]
                
                # Calculate average skin tone
                skin_tone_analysis = self._analyze_skin_tone(face_region)
                
                return {
                    'skin_tone': skin_tone_analysis['skin_tone'],
                    'confidence': skin_tone_analysis['confidence'],
                    'rgb_values': skin_tone_analysis['rgb_values']
                }
            else:
                return {'skin_tone': 'unknown', 'confidence': 0.0}
                
        except Exception as e:
            self.logger.error(f"Skin tone classification failed: {str(e)}")
            return {'skin_tone': 'unknown', 'confidence': 0.0}
    
    def _analyze_skin_tone(self, face_region) -> Dict[str, Any]:
        """Analyze skin tone from face region"""
        try:
            # Calculate average color in face region
            # Focus on center region to avoid hair and background
            h, w = face_region.shape[:2]
            center_region = face_region[h//4:3*h//4, w//4:3*w//4]
            
            # Calculate mean RGB values
            mean_color = np.mean(center_region.reshape(-1, 3), axis=0)
            b, g, r = mean_color  # OpenCV uses BGR
            
            # Map RGB to skin tone categories using L*a*b* color space
            lab_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)
            l_value = lab_color[0, 0, 0]  # Lightness component
            
            # Classify skin tone based on lightness
            if l_value > 200:
                skin_tone = 'very_light'
            elif l_value > 170:
                skin_tone = 'light'
            elif l_value > 140:
                skin_tone = 'medium_light'
            elif l_value > 110:
                skin_tone = 'medium'
            elif l_value > 80:
                skin_tone = 'medium_dark'
            elif l_value > 50:
                skin_tone = 'dark'
            else:
                skin_tone = 'very_dark'
            
            # Calculate confidence based on color consistency
            color_std = np.std(center_region.reshape(-1, 3), axis=0)
            consistency = 1.0 - (np.mean(color_std) / 255.0)
            confidence = min(consistency * 1.2, 1.0)  # Boost slightly
            
            return {
                'skin_tone': skin_tone,
                'confidence': confidence,
                'rgb_values': {'r': int(r), 'g': int(g), 'b': int(b)},
                'l_value': int(l_value)
            }
            
        except Exception as e:
            self.logger.error(f"Skin tone analysis failed: {str(e)}")
            return {
                'skin_tone': 'medium',
                'confidence': 0.3,
                'rgb_values': {'r': 0, 'g': 0, 'b': 0}
            }
    
    def _create_fallback_classification(self) -> Dict[str, Any]:
        """Create fallback classification result on error"""
        return {
            'age': 'unknown',
            'gender': 'unknown',
            'ethnicity': 'unknown',
            'body_type': 'unknown',
            'skin_tone': 'unknown',
            'confidence': 0.0,
            'detailed_scores': {
                'age_confidence': 0.0,
                'gender_confidence': 0.0,
                'ethnicity_confidence': 0.0,
                'body_type_confidence': 0.0,
                'skin_tone_confidence': 0.0
            },
            'requires_review': True,
            'error': True,
            'classification_method': 'fallback'
        }
    
    def validate_classification_result(self, result: Dict[str, Any]) -> bool:
        """Validate demographic classification result quality"""
        try:
            # Check required fields
            required_fields = ['age', 'gender', 'ethnicity', 'body_type', 'confidence']
            if not all(field in result for field in required_fields):
                return False
            
            # Check confidence threshold
            if result['confidence'] < self.confidence_thresholds['overall_min']:
                result['requires_review'] = True
            
            # Check individual confidences
            detailed_scores = result.get('detailed_scores', {})
            low_confidence_areas = []
            
            for category, threshold in self.confidence_thresholds.items():
                if category == 'overall_min':
                    continue
                    
                confidence_key = f"{category}_confidence"
                if detailed_scores.get(confidence_key, 0.0) < threshold:
                    low_confidence_areas.append(category)
            
            if low_confidence_areas:
                result['low_confidence_areas'] = low_confidence_areas
                result['requires_review'] = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Classification validation failed: {str(e)}")
            return False
    
    def get_classification_categories(self) -> Dict[str, List[str]]:
        """Get available classification categories"""
        return {
            'age_groups': self.age_groups,
            'ethnicities': self.ethnicities,
            'body_types': self.body_types,
            'skin_tones': self.skin_tones
        }
    
    def get_classifier_status(self) -> Dict[str, Any]:
        """Get demographic classifier status"""
        return {
            'models_loaded': self.models_loaded,
            'advanced_models_available': self.use_advanced_models,
            'confidence_thresholds': self.confidence_thresholds,
            'supported_categories': list(self.get_classification_categories().keys())
        }