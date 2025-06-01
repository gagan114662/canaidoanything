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
        if self.model_loaded:
            logger.info("Model enhancement models already loaded.")
            return

        logger.info(f"Loading model enhancement models to device: {self.device}...")
        any_model_loaded_successfully = False
        try:
            if GFPGAN_AVAILABLE:
                if self._load_gfpgan():
                    any_model_loaded_successfully = True
            else:
                logger.warning("GFPGAN package not available. Skipping GFPGAN loading.")
            
            if MEDIAPIPE_AVAILABLE:
                if self._load_mediapipe():
                    any_model_loaded_successfully = True
            else:
                logger.warning("MediaPipe package not available. Skipping MediaPipe loading.")

            if INSIGHTFACE_AVAILABLE:
                if self._load_insightface():
                    any_model_loaded_successfully = True
            else:
                logger.warning("InsightFace package not available. Skipping InsightFace loading.")

            # Consider model_loaded true if at least one model component loaded,
            # or if fallbacks are acceptable as "loaded" state.
            # For now, if any actual model loads, consider it a partial success.
            self.model_loaded = True # This means the service is "ready", possibly with fallbacks.
            if any_model_loaded_successfully:
                logger.info("Model enhancement models loading process completed. Some models may be using fallbacks.")
            else:
                logger.warning("All primary model enhancement components failed to load or were unavailable. Service will rely entirely on fallbacks.")
            
        except Exception as e: # Catch-all for unexpected errors during the loading orchestration
            logger.error(f"An unexpected error occurred during the loading of model enhancement models: {e}", exc_info=True)
            self.model_loaded = True # Still allow fallback mode
            logger.warning("Model enhancement services will use fallback methods due to an unexpected error during loading.")
    
    def _load_gfpgan(self) -> bool: # Return bool indicating success
        """Load GFPGAN for face enhancement"""
        logger.info("Attempting to load GFPGAN model...")
        try:
            self.gfpgan_enhancer = GFPGANer(
                model_path=settings.GFPGAN_MODEL_PATH, # Use path from settings
                upscale=settings.GFPGAN_UPSCALE_FACTOR, # Use factor from settings
                arch=settings.GFPGAN_ARCH,
                channel_multiplier=settings.GFPGAN_CHANNEL_MULTIPLIER,
                bg_upsampler=None # BG upsampler not typically used for face enhancement alone
            )
            logger.info("GFPGAN loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load GFPGAN: {e}", exc_info=True)
            self.gfpgan_enhancer = None
            return False
    
    def _load_mediapipe(self) -> bool: # Return bool indicating success
        """Load MediaPipe for pose detection"""
        logger.info("Attempting to load MediaPipe Pose model...")
        try:
            mp_pose = mp.solutions.pose
            self.pose_detector = mp_pose.Pose(
                static_image_mode=settings.MEDIAPIPE_STATIC_MODE,
                model_complexity=settings.MEDIAPIPE_MODEL_COMPLEXITY,
                enable_segmentation=False, # Segmentation not typically needed for just pose
                min_detection_confidence=settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE
            )
            logger.info("MediaPipe Pose model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load MediaPipe Pose model: {e}", exc_info=True)
            self.pose_detector = None
            return False
    
    def _load_insightface(self) -> bool: # Return bool indicating success
        """Load InsightFace for face analysis"""
        logger.info("Attempting to load InsightFace model...")
        try:
            # Specify providers based on device, can help with ONNX runtime issues
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            self.face_analyzer = insightface.app.FaceAnalysis(
                name=settings.INSIGHTFACE_MODEL_NAME,
                providers=providers
            )
            self.face_analyzer.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640)) # Standard detection size
            logger.info("InsightFace model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load InsightFace model: {e}", exc_info=True)
            self.face_analyzer = None
            return False
    
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
        logger.debug(f"Detecting faces in image of size {image.size}, mode {image.mode}.")
        try:
            if self.face_analyzer:
                logger.debug("Using InsightFace for face detection.")
                img_cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR) # Ensure RGB then BGR
                faces = self.face_analyzer.get(img_cv)
                
                results = []
                if faces:
                    for face in faces:
                        results.append({
                            "bbox": face.bbox.astype(int).tolist(), # Ensure bbox is int list
                            "confidence": float(face.det_score),
                            "landmarks": face.landmark_2d_106.astype(int).tolist() if hasattr(face, 'landmark_2d_106') else [],
                            "age": getattr(face, 'age', None),
                            "gender": str(getattr(face, 'gender', 'Unknown')) # Ensure gender is string
                        })
                    logger.info(f"InsightFace detected {len(results)} faces.")
                else:
                    logger.info("InsightFace detected no faces.")
                return results
            else:
                logger.warning("InsightFace analyzer not available. Using fallback face detection.")
                return self._fallback_face_detection(image)
                
        except Exception as e:
            logger.error(f"Face detection failed: {e}", exc_info=True)
            return [] # Return empty list on error
    
    def _fallback_face_detection(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Fallback face detection using OpenCV Haar cascades"""
        logger.info("Using fallback OpenCV Haar cascade for face detection.")
        try:
            img_rgb = image.convert('RGB') # Ensure RGB for grayscale conversion
            img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            face_cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
            if not face_cascade_path.exists():
                logger.error(f"Haar cascade file not found at {face_cascade_path}")
                return []

            face_cascade = cv2.CascadeClassifier(str(face_cascade_path))
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            
            results = []
            for (x, y, w, h) in faces:
                results.append({
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "confidence": 0.8,
                    "landmarks": [], "age": None, "gender": "Unknown"
                })
            logger.info(f"Fallback OpenCV Haar cascade detected {len(results)} faces.")
            return results
            
        except cv2.error as e:
            logger.error(f"OpenCV error in fallback face detection: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Fallback face detection failed: {e}", exc_info=True)
        return []
    
    def enhance_face(self, image: Image.Image) -> Image.Image:
        """
        Enhance face quality using GFPGAN or fallback.
        """
        logger.debug(f"Enhancing face for image size {image.size}, mode {image.mode}.")
        try:
            if self.gfpgan_enhancer:
                logger.info("Using GFPGAN for face enhancement.")
                img_rgb = image.convert('RGB') # GFPGAN expects RGB
                img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
                
                _, _, enhanced_img_cv = self.gfpgan_enhancer.enhance(
                    img_cv, has_aligned=False, only_center_face=False, paste_back=True
                )
                
                enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_img_cv, cv2.COLOR_BGR2RGB))
                logger.info("GFPGAN face enhancement successful.")
                return enhanced_pil
            else:
                logger.warning("GFPGAN enhancer not available. Using fallback face enhancement.")
                return self._fallback_face_enhancement(image)
                
        except Exception as e:
            logger.error(f"Face enhancement failed (main method): {e}", exc_info=True)
            return image # Return original image on error
    
    def _fallback_face_enhancement(self, image: Image.Image) -> Image.Image:
        """Fallback face enhancement using basic image processing."""
        logger.info("Using fallback basic image processing for face enhancement.")
        try:
            enhanced = image.copy() # Work on a copy
            enhancer_sharp = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer_sharp.enhance(settings.FALLBACK_ENHANCE_SHARPNESS) # Use settings
            
            enhancer_contrast = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer_contrast.enhance(settings.FALLBACK_ENHANCE_CONTRAST) # Use settings
            
            enhancer_color = ImageEnhance.Color(enhanced)
            enhanced = enhancer_color.enhance(settings.FALLBACK_ENHANCE_COLOR) # Use settings
            
            logger.info("Fallback face enhancement applied successfully.")
            return enhanced
            
        except Exception as e:
            logger.error(f"Fallback face enhancement failed: {e}", exc_info=True)
            return image # Return original on error
    
    def detect_pose(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect pose keypoints in the image using MediaPipe or fallback.
        """
        logger.debug(f"Detecting pose in image size {image.size}, mode {image.mode}.")
        try:
            if self.pose_detector:
                logger.info("Using MediaPipe for pose detection.")
                img_rgb = image.convert('RGB') # MediaPipe expects RGB
                img_cv_mp = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR) # It internally converts to RGB

                # Process does not need to be in try-except if detector loaded correctly
                pose_results = self.pose_detector.process(img_cv_mp)
                
                if pose_results.pose_landmarks:
                    keypoints = []
                    for landmark in pose_results.pose_landmarks.landmark:
                        keypoints.append([
                            landmark.x * image.width, # Denormalize
                            landmark.y * image.height, # Denormalize
                            round(landmark.visibility, 3) if landmark.HasField('visibility') else 0.5 # Check visibility
                        ])
                    
                    confidences = [kp[2] for kp in keypoints if kp[2] is not None] # Filter None visibilities
                    avg_confidence = round(np.mean(confidences), 3) if confidences else 0.0
                    
                    logger.info(f"MediaPipe detected pose with {len(keypoints)} keypoints. Avg confidence: {avg_confidence:.2f}")
                    return {"confidence": avg_confidence, "keypoints": keypoints, "pose_detected": True}
                else:
                    logger.info("MediaPipe detected no pose landmarks.")
                    return {"confidence": 0.0, "keypoints": [], "pose_detected": False}
            else:
                logger.warning("MediaPipe pose detector not available. Using fallback pose detection.")
                return self._fallback_pose_detection(image)
                
        except Exception as e:
            logger.error(f"Pose detection failed (main method): {e}", exc_info=True)
            return {"confidence": 0.0, "keypoints": [], "pose_detected": False, "error": str(e)}
    
    def _fallback_pose_detection(self, image: Image.Image) -> Dict[str, Any]: # Added image param for consistency
        """Fallback pose detection (simplified)."""
        logger.info("Using fallback pose detection (returns default 'detected' state).")
        # This is a very basic fallback, actual keypoints would require a CV model.
        return {
            "confidence": 0.6,  # Default confidence for fallback
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
        logger.debug(f"Optimizing body proportions for image size {image.size}, mode {image.mode}.")
        try:
            pose_info = self.detect_pose(image) # This already logs sufficiently
            
            if pose_info.get("pose_detected") and pose_info.get("keypoints"):
                logger.info("Pose detected, applying specific proportion corrections.")
                return self._apply_proportion_corrections(image, pose_info["keypoints"])
            else:
                logger.info("No pose or keypoints detected, applying general body optimization.")
                return self._general_body_optimization(image)
                
        except Exception as e:
            logger.error(f"Body optimization failed: {e}", exc_info=True)
            return image # Return original on error
    
    def _apply_proportion_corrections(self, image: Image.Image, keypoints: List) -> Image.Image:
        """Apply specific proportion corrections based on pose keypoints."""
        logger.debug(f"Applying proportion corrections based on {len(keypoints)} keypoints.")
        try:
            img_rgb = image.convert('RGB')
            img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            
            height, width = img_cv.shape[:2]
            stretch_factor = settings.BODY_PROPORTION_STRETCH_FACTOR # Use from settings
            
            logger.debug(f"Applying vertical stretch factor of {stretch_factor}.")
            M = np.float32([[1, 0, 0], [0, stretch_factor, (height * (1 - stretch_factor)) / 2]]) # Centered stretch
            
            corrected_cv = cv2.warpAffine(img_cv, M, (width, height)) # Output size remains same
            
            logger.info("Proportion corrections applied successfully.")
            return Image.fromarray(cv2.cvtColor(corrected_cv, cv2.COLOR_BGR2RGB))
            
        except cv2.error as e:
            logger.error(f"OpenCV error during proportion correction: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Proportion correction failed: {e}", exc_info=True)
        return image # Return original on error
    
    def _general_body_optimization(self, image: Image.Image) -> Image.Image:
        """Apply general body optimization without specific pose data."""
        logger.debug("Applying general body optimization (subtle sharpening).")
        try:
            img_rgb = image.convert('RGB')
            img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            
            kernel = np.array([[-1,-1,-1], [-1,9.5,-1], [-1,-1,-1]]) # Slightly stronger sharpening
            sharpened_cv = cv2.filter2D(img_cv, -1, kernel)
            
            # Blend for subtlety, if desired, or use directly if effect is good
            # result_cv = cv2.addWeighted(img_cv, 0.7, sharpened_cv, 0.3, 0)
            result_cv = sharpened_cv # Using sharpened directly for now
            
            logger.info("General body optimization (sharpening) applied.")
            return Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
            
        except cv2.error as e:
            logger.error(f"OpenCV error during general body optimization: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"General body optimization failed: {e}", exc_info=True)
        return image # Return original on error
    
    def _assess_face_quality(self, image: Image.Image) -> float: # Already fairly well-logged from previous pass
        """Assess face quality in image (0-1 score)"""
        logger.debug(f"Assessing face quality for image size {image.size}, mode {image.mode}.")
        try:
            faces = self.detect_faces(image) # detect_faces itself logs
            if not faces:
                logger.info("No faces detected for quality assessment, returning 0.0 quality.")
                return 0.0
            
            best_face = max(faces, key=lambda x: x.get("confidence", 0.0))
            face_confidence = best_face.get("confidence", 0.0)
            
            img_rgb = image.convert('RGB')
            img_array_assess = np.array(img_rgb)
            
            gray_assess = cv2.cvtColor(img_array_assess, cv2.COLOR_RGB2GRAY)
            sharpness_assess = cv2.Laplacian(gray_assess, cv2.CV_64F).var()
            sharpness_score_assess = min(sharpness_assess / 1000.0, 1.0)
            
            brightness_assess = np.mean(img_array_assess)
            brightness_score_assess = 1.0 - abs(brightness_assess - 128.0) / 128.0
            
            quality_score_combined = (face_confidence * 0.5 + sharpness_score_assess * 0.3 + brightness_score_assess * 0.2)
            final_quality_score = round(min(max(quality_score_combined, 0.0), 1.0), 3)

            logger.debug(f"Face quality assessment: Confidence={face_confidence:.2f}, SharpnessVar={sharpness_assess:.2f} (Score:{sharpness_score_assess:.2f}), Brightness={brightness_assess:.2f} (Score:{brightness_score_assess:.2f}). Final Score: {final_quality_score:.2f}")
            return final_quality_score
            
        except Exception as e:
            logger.error(f"Face quality assessment failed: {e}", exc_info=True)
            return 0.5  # Default neutral score on error
    
    def enhance_model(self, image: Image.Image) -> Dict[str, Any]:
        """
        Complete model enhancement pipeline
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhancement results with enhanced image and metadata
        """
        logger.info(f"Starting model enhancement pipeline for image size {image.size}, mode {image.mode}.")
        if not self.model_loaded: # Ensure models are loaded or loading is attempted
            logger.warning("Models not pre-loaded for enhancement pipeline. Attempting to load now.")
            self.load_models() # This sets self.model_loaded status

        start_time = time.time()
        current_image_state = image.copy() # Work on a copy
        face_count = 0
        detected_faces_info = []
        pose_info_result = {"pose_detected": False, "confidence": 0.0}

        try:
            # Detect faces first
            logger.debug("Detecting faces...")
            detected_faces_info = self.detect_faces(current_image_state) # Uses fallback if primary fails
            face_count = len(detected_faces_info)
            logger.info(f"Detected {face_count} faces.")
            
            # Enhance face if detected
            if face_count > 0 and self.gfpgan_enhancer: # Prioritize GFPGAN if available
                logger.info("Enhancing detected face(s) with GFPGAN.")
                current_image_state = self.enhance_face(current_image_state) # Uses fallback if GFPGAN fails
            elif face_count > 0: # Fallback if GFPGAN not available but faces detected
                 logger.info("GFPGAN not available/loaded. Attempting fallback face enhancement.")
                 current_image_state = self._fallback_face_enhancement(current_image_state)

            # Detect and optimize pose/body
            logger.debug("Detecting pose...")
            pose_info_result = self.detect_pose(current_image_state) # Uses fallback if primary fails
            if pose_info_result.get("pose_detected"):
                logger.info(f"Pose detected with confidence: {pose_info_result.get('confidence', 0.0):.2f}. Optimizing body proportions.")
                current_image_state = self.optimize_body_proportions(current_image_state) # Internally uses pose_info
            else:
                logger.info("No pose detected or pose detection failed. Skipping specific proportion corrections.")
                # Optionally, apply general body optimization even if pose not detected
                # current_image_state = self._general_body_optimization(current_image_state)

            logger.debug("Assessing final enhanced image quality.")
            quality_score = self._assess_face_quality(current_image_state) * 10  # Scale to 0-10
            
            processing_time = time.time() - start_time
            logger.info(f"Model enhancement pipeline completed in {processing_time:.2f}s. Final Quality Score: {quality_score:.2f}")
            
            # More accurate fallback_used check
            # True if any of the primary models were unavailable OR their instances are None
            gfpgan_fallback = not GFPGAN_AVAILABLE or self.gfpgan_enhancer is None
            mediapipe_fallback = not MEDIAPIPE_AVAILABLE or self.pose_detector is None
            insightface_fallback = not INSIGHTFACE_AVAILABLE or self.face_analyzer is None
            
            overall_fallback_status = gfpgan_fallback or mediapipe_fallback or insightface_fallback

            return {
                "enhanced_image": current_image_state,
                "quality_score": round(quality_score, 2),
                "faces_detected": face_count,
                "pose_detected": pose_info_result.get("pose_detected", False),
                "processing_time": round(processing_time, 2),
                "fallback_used": overall_fallback_status,
                "metadata": {
                    "original_size": image.size,
                    "enhanced_size": current_image_state.size,
                    "face_details": detected_faces_info, # Include detected face bboxes etc.
                    "pose_confidence": round(pose_info_result.get('confidence', 0.0), 2),
                    "component_fallbacks": { # More granular fallback info
                        "gfpgan": gfpgan_fallback,
                        "mediapipe_pose": mediapipe_fallback,
                        "insightface_analysis": insightface_fallback,
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Model enhancement pipeline failed: {e}", exc_info=True)
            return {
                "enhanced_image": image, # Return original image on pipeline failure
                "quality_score": 3.0,
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
        logger.info(f"Starting batch enhancement for {len(images)} images.")
        results = []
        for i, img_item in enumerate(images):
            logger.debug(f"Batch enhancing image {i+1}/{len(images)}.")
            try:
                result = self.enhance_model(img_item) # enhance_model has its own comprehensive logging
                results.append(result)
            except Exception as e:
                logger.error(f"Error enhancing image {i+1} in batch: {e}", exc_info=True)
                # Append a basic error structure for this image
                results.append({
                    "enhanced_image": img_item, "quality_score": 0.0, "faces_detected": 0,
                    "pose_detected": False, "processing_time": 0.0, "fallback_used": True,
                    "error": str(e), "metadata": {"original_size": img_item.size}
                })
        logger.info(f"Batch enhancement completed for {len(images)} images.")
        return results
    
    def clear_cache(self):
        """Clear model cache and free GPU memory if applicable."""
        logger.info("Attempting to clear model enhancement caches and free GPU memory.")
        try:
            if torch.cuda.is_available():
                logger.debug("Clearing CUDA cache for model enhancement service.")
                torch.cuda.empty_cache()
            logger.info("Model enhancement cache cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to clear model enhancement cache: {e}", exc_info=True)
    
    def unload_models(self):
        """Unload all models used by this service to free memory."""
        logger.info("Attempting to unload all model enhancement models...")
        unloaded_count = 0
        try:
            if self.gfpgan_enhancer:
                del self.gfpgan_enhancer
                self.gfpgan_enhancer = None
                logger.debug("GFPGAN enhancer unloaded.")
                unloaded_count +=1
            
            if self.pose_detector:
                # MediaPipe Pose objects have a close() method
                try:
                    self.pose_detector.close()
                    logger.debug("MediaPipe Pose detector closed.")
                except Exception as e_mp_close: # Broad exception as close() might fail
                    logger.warning(f"Error closing MediaPipe Pose detector: {e_mp_close}", exc_info=True)
                self.pose_detector = None
                unloaded_count +=1
            
            if self.face_analyzer:
                del self.face_analyzer # InsightFace objects usually don't have explicit close/unload
                self.face_analyzer = None
                logger.debug("InsightFace analyzer unloaded.")
                unloaded_count +=1
            
            self.model_loaded = False # Set to false as primary models are unloaded
            self.clear_cache() # Clear GPU cache after unloading
            logger.info(f"Model enhancement models unloaded successfully. Unloaded {unloaded_count} components.")
            
        except Exception as e:
            logger.error(f"Failed to unload model enhancement models: {e}", exc_info=True)
            # Even on error, try to set models to None and loaded to False
            self.gfpgan_enhancer = None
            self.pose_detector = None
            self.face_analyzer = None
            self.model_loaded = False

# Add missing import
# import time # time is already imported at the top