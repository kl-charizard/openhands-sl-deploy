#!/usr/bin/env python3
"""
🖐️ Pose Extraction Module for ASL Recognition
MediaPipe-based hand and body pose extraction optimized for OpenHands
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PoseExtractor:
    """Extract pose landmarks for ASL recognition using MediaPipe"""
    
    def __init__(self):
        """Initialize MediaPipe pose and hand detection"""
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize detectors
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("✅ MediaPipe pose extractors initialized")
    
    def extract_pose(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract pose landmarks from frame
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            Dictionary containing pose data or None if no pose detected
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb_frame.shape[:2]
            
            # Detect hands
            hand_results = self.hands.process(rgb_frame)
            
            # Detect body pose
            pose_results = self.pose.process(rgb_frame)
            
            # Process results
            pose_data = {
                'hands': [],
                'body': None,
                'frame_shape': (height, width),
                'has_data': False
            }
            
            # Extract hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness
                ):
                    # Convert to normalized coordinates
                    hand_coords = []
                    for landmark in hand_landmarks.landmark:
                        hand_coords.extend([
                            landmark.x,  # Normalized x (0-1)
                            landmark.y,  # Normalized y (0-1)
                            landmark.z   # Relative depth
                        ])
                    
                    pose_data['hands'].append({
                        'landmarks': hand_coords,
                        'handedness': handedness.classification[0].label,
                        'confidence': handedness.classification[0].score
                    })
                
                pose_data['has_data'] = True
            
            # Extract body pose landmarks (focus on upper body for ASL)
            if pose_results.pose_landmarks:
                body_coords = []
                
                # Key body landmarks for ASL (shoulders, arms, torso)
                key_landmarks = [
                    # Shoulders
                    11, 12,  # Left/Right shoulder
                    # Arms  
                    13, 14, 15, 16,  # Left/Right elbow, wrist
                    # Torso
                    23, 24   # Left/Right hip (for body orientation)
                ]
                
                for idx in key_landmarks:
                    if idx < len(pose_results.pose_landmarks.landmark):
                        landmark = pose_results.pose_landmarks.landmark[idx]
                        body_coords.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility
                        ])
                
                pose_data['body'] = body_coords
                pose_data['has_data'] = True
            
            # Convert to OpenHands format
            if pose_data['has_data']:
                return self._format_for_openhands(pose_data)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Pose extraction error: {e}")
            return None
    
    def _format_for_openhands(self, pose_data: Dict[str, Any]) -> np.ndarray:
        """
        Format pose data for OpenHands model input
        
        Args:
            pose_data: Raw pose detection results
            
        Returns:
            Formatted numpy array for OpenHands
        """
        # OpenHands expects concatenated pose features
        features = []
        
        # Hand landmarks (21 points × 3 coords × 2 hands max)
        max_hands = 2
        hand_feature_size = 21 * 3  # 21 landmarks, 3 coordinates each
        
        for i in range(max_hands):
            if i < len(pose_data['hands']):
                # Use actual hand data
                hand_landmarks = pose_data['hands'][i]['landmarks']
                features.extend(hand_landmarks)
            else:
                # Pad with zeros for missing hands
                features.extend([0.0] * hand_feature_size)
        
        # Body landmarks (8 key points × 4 values)
        if pose_data['body']:
            features.extend(pose_data['body'])
        else:
            # Pad with zeros if no body detected
            features.extend([0.0] * (8 * 4))
        
        # Convert to numpy array
        feature_array = np.array(features, dtype=np.float32)
        
        # Reshape for model input (add batch dimension)
        return feature_array.reshape(1, -1)
    
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw pose landmarks on frame for visualization
        
        Args:
            frame: OpenCV frame
            
        Returns:
            Annotated frame
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame = frame.copy()
            
            # Detect and draw hands
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Detect and draw body pose
            pose_results = self.pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Landmark drawing error: {e}")
            return frame
    
    def get_pose_confidence(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Get confidence scores for pose detection
        
        Args:
            frame: OpenCV frame
            
        Returns:
            Dictionary with confidence scores
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            hand_results = self.hands.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)
            
            confidences = {
                'hands': 0.0,
                'body': 0.0,
                'overall': 0.0
            }
            
            # Hand confidence
            if hand_results.multi_handedness:
                hand_scores = [
                    classification.score 
                    for handedness in hand_results.multi_handedness
                    for classification in handedness.classification
                ]
                confidences['hands'] = np.mean(hand_scores) if hand_scores else 0.0
            
            # Body pose confidence (estimated from landmark visibility)
            if pose_results.pose_landmarks:
                visibilities = [
                    landmark.visibility 
                    for landmark in pose_results.pose_landmarks.landmark
                    if hasattr(landmark, 'visibility')
                ]
                confidences['body'] = np.mean(visibilities) if visibilities else 0.0
            
            # Overall confidence
            confidences['overall'] = (confidences['hands'] + confidences['body']) / 2
            
            return confidences
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return {'hands': 0.0, 'body': 0.0, 'overall': 0.0}
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        try:
            if hasattr(self, 'hands'):
                self.hands.close()
            if hasattr(self, 'pose'):
                self.pose.close()
        except:
            pass
