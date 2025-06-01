import torch
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from app.core.config import settings

# Model enhancement imports
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False
    logging.warning("GFPGAN not available. Install gfpgan package for face enhancement.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Install mediapipe for pose detection.")

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Install insightface for face analysis.")

logger = logging.getLogger(__name__)

class ModelEnhancementService:
    """Service for enhancing model appearance (face, pose, body)"""
    
    def __init__(self):
        self.gfpgan_enhancer = None
        self.pose_detector = None
        self.face_analyzer = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_models(self):
        """Load all model enhancement models"""
        try:
            logger.info("Loading model enhancement models...")
            
            # Load GFPGAN for face enhancement
            if GFPGAN_AVAILABLE:
                self._load_gfpgan()
            
            # Load MediaPipe for pose detection
            if MEDIAPIPE_AVAILABLE:
                self._load_mediapipe()
            
            # Load InsightFace for face analysis
            if INSIGHTFACE_AVAILABLE:
                self._load_insightface()
            
            self.model_loaded = True
            logger.info("Model enhancement models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model enhancement models: {str(e)}")
            # Set fallback mode
            self.model_loaded = True
            logger.info("Using fallback model enhancement methods")
    
    def _load_gfpgan(self):
        """Load GFPGAN for face enhancement"""
        try:
            self.gfpgan_enhancer = GFPGANer(
                model_path='GFPGANv1.4.pth',
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            logger.info("GFPGAN loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load GFPGAN: {e}")
            self.gfpgan_enhancer = None
    
    def _load_mediapipe(self):
        """Load MediaPipe for pose detection"""
        try:
            mp_pose = mp.solutions.pose
            self.pose_detector = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.7
            )
            logger.info("MediaPipe pose detector loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load MediaPipe: {e}")
            self.pose_detector = None
    
    def _load_insightface(self):
        """Load InsightFace for face analysis"""
        try:
            self.face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l')
            self.face_analyzer.prepare(ctx_id=0 if self.device == "cuda" else -1)
            logger.info("InsightFace analyzer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load InsightFace: {e}")
            self.face_analyzer = None
    
    def is_model_loaded(self) -> bool:
        """Check if models are loaded"""
        return self.model_loaded
    
    def detect_faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect faces in the image
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of face detection results
        """
        try:
            if self.face_analyzer:
                # Convert PIL to cv2 format
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                faces = self.face_analyzer.get(img_cv)
                
                results = []
                for face in faces:
                    results.append({
                        "bbox": face.bbox.tolist(),
                        "confidence": float(face.det_score),
                        "landmarks": face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else [],
                        "age": getattr(face, 'age', None),
                        "gender": getattr(face, 'gender', None)
                    })
                
                return results
            else:
                # Fallback face detection using OpenCV
                return self._fallback_face_detection(image)
                
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _fallback_face_detection(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Fallback face detection using OpenCV Haar cascades"""
        try:
            # Convert to cv2 format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Load Haar cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            results = []
            for (x, y, w, h) in faces:
                results.append({
                    "bbox": [x, y, x + w, y + h],
                    "confidence": 0.8,  # Default confidence for Haar cascade
                    "landmarks": [],
                    "age": None,
                    "gender": None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback face detection failed: {e}")
            return []
    
    def enhance_face(self, image: Image.Image) -> Image.Image:
        """
        Enhance face quality using GFPGAN
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        try:
            if self.gfpgan_enhancer:
                # Convert PIL to cv2 format
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Enhance with GFPGAN
                _, _, enhanced_img = self.gfpgan_enhancer.enhance(
                    img_cv, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
                
                # Convert back to PIL
                enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
                return enhanced_pil
            else:
                # Fallback enhancement
                return self._fallback_face_enhancement(image)
                
        except Exception as e:
            logger.error(f"Face enhancement failed: {e}")
            return image
    
    def _fallback_face_enhancement(self, image: Image.Image) -> Image.Image:
        """Fallback face enhancement using basic image processing"""
        try:
            # Apply sharpening
            enhancer = ImageEnhance.Sharpness(image)
            enhanced = enhancer.enhance(1.2)
            
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Apply color enhancement
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.05)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Fallback face enhancement failed: {e}")
            return image
    
    def detect_pose(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect pose keypoints in the image
        
        Args:
            image: Input PIL Image
            
        Returns:
            Pose detection results with keypoints and confidence
        """
        try:
            if self.pose_detector:
                # Convert PIL to cv2 format
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                results = self.pose_detector.process(img_cv)
                
                if results.pose_landmarks:
                    keypoints = []
                    for landmark in results.pose_landmarks.landmark:
                        keypoints.append([
                            landmark.x * image.width,
                            landmark.y * image.height,
                            landmark.visibility
                        ])
                    
                    # Calculate overall confidence
                    confidences = [kp[2] for kp in keypoints]
                    avg_confidence = np.mean(confidences)
                    
                    return {
                        "confidence": avg_confidence,
                        "keypoints": keypoints,
                        "pose_detected": True
                    }
                else:
                    return {
                        "confidence": 0.0,
                        "keypoints": [],
                        "pose_detected": False
                    }
            else:
                # Fallback pose detection
                return self._fallback_pose_detection(image)
                
        except Exception as e:
            logger.error(f"Pose detection failed: {e}")
            return {"confidence": 0.0, "keypoints": [], "pose_detected": False}
    
    def _fallback_pose_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback pose detection (simplified)"""
        # For fallback, return a basic pose structure
        return {
            "confidence": 0.7,  # Default confidence
            "keypoints": [],
            "pose_detected": True,
            "fallback_used": True
        }
    
    def optimize_body_proportions(self, image: Image.Image) -> Image.Image:
        """
        Optimize body proportions and posture
        
        Args:
            image: Input PIL Image
            
        Returns:
            Optimized PIL Image
        """
        try:
            # Get pose information
            pose_info = self.detect_pose(image)
            
            if pose_info["pose_detected"] and pose_info["keypoints"]:
                # Apply pose-based corrections
                return self._apply_proportion_corrections(image, pose_info["keypoints"])
            else:
                # Apply general body optimization
                return self._general_body_optimization(image)
                
        except Exception as e:
            logger.error(f"Body optimization failed: {e}")
            return image
    
    def _apply_proportion_corrections(self, image: Image.Image, keypoints: List) -> Image.Image:
        """Apply specific proportion corrections based on pose keypoints"""
        try:
            # Convert to cv2 for processing
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply subtle corrections (this is simplified - real implementation would be more complex)
            # For now, apply slight vertical stretch to improve proportions
            height, width = img_cv.shape[:2]
            
            # Create transformation matrix for subtle proportion adjustment
            stretch_factor = 1.02  # 2% vertical stretch
            M = np.float32([[1, 0, 0], [0, stretch_factor, 0]])
            
            # Apply transformation
            corrected = cv2.warpAffine(img_cv, M, (width, int(height * stretch_factor)))
            
            # Resize back to original size
            corrected = cv2.resize(corrected, (width, height))
            
            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.error(f"Proportion correction failed: {e}")
            return image
    
    def _general_body_optimization(self, image: Image.Image) -> Image.Image:
        """Apply general body optimization without specific pose data"""
        try:
            # Apply subtle enhancements
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Slight sharpening for better definition
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(img_cv, -1, kernel)
            
            # Blend with original
            result = cv2.addWeighted(img_cv, 0.7, sharpened, 0.3, 0)
            
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.error(f"General body optimization failed: {e}")
            return image
    
    def _assess_face_quality(self, image: Image.Image) -> float:
        """Assess face quality in image (0-1 score)"""
        try:
            faces = self.detect_faces(image)
            if not faces:
                return 0.0
            
            # Use highest confidence face
            best_face = max(faces, key=lambda x: x["confidence"])
            
            # Calculate quality metrics
            face_confidence = best_face["confidence"]
            
            # Additional quality checks (simplified)
            img_array = np.array(image)
            
            # Check image sharpness (variance of Laplacian)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 1000, 1.0)  # Normalize
            
            # Check brightness
            brightness = np.mean(img_array)
            brightness_score = 1.0 - abs(brightness - 128) / 128  # Optimal around 128
            
            # Combine scores
            quality_score = (face_confidence * 0.5 + sharpness_score * 0.3 + brightness_score * 0.2)
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Face quality assessment failed: {e}")
            return 0.5  # Default middle score
    
    def enhance_model(self, image: Image.Image) -> Dict[str, Any]:
        """
        Complete model enhancement pipeline
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhancement results with enhanced image and metadata
        """
        try:
            start_time = time.time()
            
            # Detect faces first
            faces = self.detect_faces(image)
            face_count = len(faces)
            
            # Enhance face if detected
            enhanced_image = image
            if face_count > 0:
                enhanced_image = self.enhance_face(enhanced_image)
            
            # Detect and optimize pose/body
            pose_info = self.detect_pose(enhanced_image)
            if pose_info["pose_detected"]:
                enhanced_image = self.optimize_body_proportions(enhanced_image)
            
            # Assess final quality
            quality_score = self._assess_face_quality(enhanced_image) * 10  # Scale to 0-10
            
            processing_time = time.time() - start_time
            
            # Determine if fallback was used
            fallback_used = (
                not GFPGAN_AVAILABLE or 
                not MEDIAPIPE_AVAILABLE or 
                self.gfpgan_enhancer is None or 
                self.pose_detector is None
            )
            
            return {
                "enhanced_image": enhanced_image,
                "quality_score": quality_score,
                "faces_detected": face_count,
                "pose_detected": pose_info["pose_detected"],
                "processing_time": processing_time,
                "fallback_used": fallback_used,
                "metadata": {
                    "original_size": image.size,
                    "enhanced_size": enhanced_image.size,
                    "face_confidence": faces[0]["confidence"] if faces else 0.0,
                    "pose_confidence": pose_info["confidence"]
                }
            }
            
        except Exception as e:
            logger.error(f"Model enhancement pipeline failed: {e}")
            
            # Return original image with error info
            return {
                "enhanced_image": image,
                "quality_score": 3.0,  # Low score for failed enhancement
                "faces_detected": 0,
                "pose_detected": False,
                "processing_time": 0.0,
                "fallback_used": True,
                "error": str(e),
                "metadata": {
                    "original_size": image.size,
                    "enhanced_size": image.size,
                    "face_confidence": 0.0,
                    "pose_confidence": 0.0
                }
            }
    
    def enhance_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Enhance multiple model images in batch
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of enhancement results
        """
        results = []
        for image in images:
            result = self.enhance_model(image)
            results.append(result)
        
        return results
    
    def clear_cache(self):
        """Clear model cache and free memory"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model enhancement cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def unload_models(self):
        """Unload models to free memory"""
        try:
            if self.gfpgan_enhancer:
                del self.gfpgan_enhancer
                self.gfpgan_enhancer = None
            
            if self.pose_detector:
                self.pose_detector.close()
                self.pose_detector = None
            
            if self.face_analyzer:
                del self.face_analyzer
                self.face_analyzer = None
            
            self.model_loaded = False
            self.clear_cache()
            logger.info("Model enhancement models unloaded")
            
        except Exception as e:
            logger.error(f"Failed to unload models: {e}")

    def apply_ai_sharpening(self, image: Image.Image, intensity: float = 0.5) -> Dict[str, Any]:
        logger.info("Applying AI smart sharpening to model.")
        try:
            sharpened_image = ai_smart_sharpen(image.copy(), intensity=intensity)
            applied_successfully = image.tobytes() != sharpened_image.tobytes() if sharpened_image else False
            return {
                "processed_image": sharpened_image,
                "operation": "ai_sharpen",
                "intensity": intensity,
                "success": applied_successfully
            }
        except Exception as e:
            logger.error(f"Error during AI smart sharpening for model: {str(e)}")
            return {"processed_image": image, "operation": "ai_sharpen", "success": False, "error": str(e)}

    def apply_ai_denoising(self, image: Image.Image, strength: float = 0.5) -> Dict[str, Any]:
        logger.info("Applying AI smart denoising to model.")
        try:
            denoised_image = ai_smart_denoise(image.copy(), strength=strength)
            applied_successfully = image.tobytes() != denoised_image.tobytes() if denoised_image else False
            return {
                "processed_image": denoised_image,
                "operation": "ai_denoise",
                "strength": strength,
                "success": applied_successfully
            }
        except Exception as e:
            logger.error(f"Error during AI smart denoising for model: {str(e)}")
            return {"processed_image": image, "operation": "ai_denoise", "success": False, "error": str(e)}

# Add missing import
import time
from app.utils.image_utils import ai_smart_sharpen, ai_smart_denoise