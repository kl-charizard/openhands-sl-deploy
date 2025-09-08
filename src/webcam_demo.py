#!/usr/bin/env python3
"""
🎥 Real-time ASL Recognition using OpenHands Pretrained Models
Tesla P40 optimized webcam demo with mobile-ready inference pipeline
"""

import cv2
import numpy as np
import time
import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    import mediapipe as mp
    from openhands import OpengHands  # Note: may need to handle import errors
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("📦 Install missing dependencies:")
    print("   pip install tensorflow opencv-python mediapipe")
    print("   pip install git+https://github.com/AI4Bharat/OpenHands.git")
    sys.exit(1)

from src.pose_extractor import PoseExtractor
from src.gpu_optimizer import GPUOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASLWebcamRecognizer:
    """Real-time ASL recognition using OpenHands pretrained models"""
    
    def __init__(self, model_path=None, confidence_threshold=0.7):
        """
        Initialize the ASL recognizer
        
        Args:
            model_path: Path to OpenHands model (None for default)
            confidence_threshold: Minimum confidence for predictions
        """
        self.confidence_threshold = confidence_threshold
        self.pose_extractor = PoseExtractor()
        self.gpu_optimizer = GPUOptimizer()
        
        # Initialize GPU optimization
        self._setup_gpu()
        
        # Load OpenHands pretrained model
        self._load_model(model_path)
        
        # Recognition state
        self.recognition_history = []
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def _setup_gpu(self):
        """Configure Tesla P40 for optimal performance"""
        logger.info("🚀 Setting up Tesla P40 optimization...")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for Tesla P40
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                logger.info(f"✅ Tesla P40 configured: {gpus[0]}")
                logger.info(f"✅ Available VRAM: 24GB")
                
                # Optimize for inference
                tf.config.optimizer.set_jit(True)  # Enable XLA
                
            except RuntimeError as e:
                logger.error(f"❌ GPU setup failed: {e}")
        else:
            logger.warning("⚠️ No GPU detected - using CPU (will be slower)")
    
    def _load_model(self, model_path):
        """Load OpenHands pretrained ASL model"""
        logger.info("📦 Loading OpenHands pretrained ASL model...")
        
        try:
            # Try to load OpenHands model
            if model_path and Path(model_path).exists():
                # Load custom model path
                self.model = OpengHands.load_model(model_path)
                logger.info(f"✅ Custom model loaded: {model_path}")
            else:
                # Load default pretrained ASL model
                self.model = OpengHands.load_pretrained('asl')
                logger.info("✅ Default OpenHands ASL model loaded")
                
            # Get model info
            vocab_size = len(self.model.vocabulary) if hasattr(self.model, 'vocabulary') else 'Unknown'
            logger.info(f"📚 ASL Vocabulary size: {vocab_size}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load OpenHands model: {e}")
            logger.info("🔧 Fallback solutions:")
            logger.info("   1. Check OpenHands installation: pip install OpenHands")  
            logger.info("   2. Try manual install: pip install git+https://github.com/AI4Bharat/OpenHands.git")
            logger.info("   3. Use compatibility mode (see docs)")
            sys.exit(1)
    
    def recognize_frame(self, frame):
        """
        Recognize ASL signs in a single frame
        
        Args:
            frame: OpenCV frame (BGR)
            
        Returns:
            dict: Recognition results with confidence scores
        """
        # Extract pose landmarks
        pose_data = self.pose_extractor.extract_pose(frame)
        
        if pose_data is None:
            return {
                'prediction': None,
                'confidence': 0.0,
                'message': 'No hands detected'
            }
        
        try:
            # OpenHands inference
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                prediction = self.model.predict(pose_data)
                
            # Process results
            if hasattr(prediction, 'confidence') and prediction.confidence > self.confidence_threshold:
                return {
                    'prediction': prediction.sign,
                    'confidence': prediction.confidence,
                    'message': f"Recognized: {prediction.sign}"
                }
            else:
                return {
                    'prediction': None,
                    'confidence': prediction.confidence if hasattr(prediction, 'confidence') else 0.0,
                    'message': f"Low confidence: {prediction.confidence:.2f}" if hasattr(prediction, 'confidence') else 'Low confidence'
                }
                
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return {
                'prediction': None,
                'confidence': 0.0,
                'message': f'Error: {str(e)}'
            }
    
    def update_recognition_history(self, result):
        """Update recognition history for sentence building"""
        if result['prediction']:
            self.recognition_history.append({
                'sign': result['prediction'],
                'confidence': result['confidence'],
                'timestamp': time.time()
            })
            
            # Keep only recent predictions (last 10 seconds)
            cutoff_time = time.time() - 10.0
            self.recognition_history = [
                r for r in self.recognition_history 
                if r['timestamp'] > cutoff_time
            ]
    
    def get_sentence(self):
        """Build sentence from recent recognitions"""
        if not self.recognition_history:
            return ""
            
        # Group consecutive similar signs
        sentence_parts = []
        current_sign = None
        
        for recognition in self.recognition_history[-10:]:  # Last 10 recognitions
            sign = recognition['sign']
            if sign != current_sign:
                sentence_parts.append(sign)
                current_sign = sign
        
        return " ".join(sentence_parts)
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            return fps
        
        return None
    
    def draw_results(self, frame, result, fps=None):
        """Draw recognition results on frame"""
        height, width = frame.shape[:2]
        
        # Draw pose landmarks
        annotated_frame = self.pose_extractor.draw_landmarks(frame)
        
        # Recognition info box
        info_box_height = 120
        cv2.rectangle(annotated_frame, (10, 10), (width - 10, info_box_height), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (width - 10, info_box_height), (0, 255, 0), 2)
        
        # Current prediction
        prediction_text = result['message']
        cv2.putText(annotated_frame, prediction_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Confidence bar
        if result['confidence'] > 0:
            bar_width = int((width - 40) * result['confidence'])
            cv2.rectangle(annotated_frame, (20, 50), (20 + bar_width, 65), (0, 255, 0), -1)
            cv2.putText(annotated_frame, f"Confidence: {result['confidence']:.2f}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Sentence
        sentence = self.get_sentence()
        if sentence:
            cv2.putText(annotated_frame, f"Sentence: {sentence}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # FPS counter  
        if fps:
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (width - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # GPU status
        gpu_status = "Tesla P40" if tf.config.list_physical_devices('GPU') else "CPU"
        cv2.putText(annotated_frame, f"Compute: {gpu_status}", (width - 150, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return annotated_frame
    
    def run_webcam_demo(self, camera_id=0):
        """Run real-time webcam ASL recognition"""
        logger.info("🎥 Starting webcam ASL recognition...")
        logger.info("📋 Controls:")
        logger.info("   'q' - Quit")
        logger.info("   'r' - Reset sentence")
        logger.info("   'c' - Clear recognition history")
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            logger.error(f"❌ Cannot open camera {camera_id}")
            return
        
        logger.info("✅ Webcam initialized - starting recognition...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Failed to read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Recognize ASL signs
                result = self.recognize_frame(frame)
                
                # Update history
                self.update_recognition_history(result)
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Draw results
                annotated_frame = self.draw_results(frame, result, fps)
                
                # Display frame
                cv2.imshow('OpenHands ASL Recognition - Tesla P40 Optimized', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("🛑 Quitting...")
                    break
                elif key == ord('r'):
                    self.recognition_history.clear()
                    logger.info("🔄 Recognition history cleared")
                elif key == ord('c'):
                    self.recognition_history.clear()
                    logger.info("🧹 Recognition history cleared")
                    
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Demo error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("✅ Webcam demo finished")

def main():
    parser = argparse.ArgumentParser(description='Real-time ASL recognition with OpenHands')
    parser.add_argument('--model', type=str, help='Path to custom OpenHands model')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold (default: 0.7)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create recognizer
    recognizer = ASLWebcamRecognizer(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Run demo
    recognizer.run_webcam_demo(camera_id=args.camera)

if __name__ == "__main__":
    main()
