#!/usr/bin/env python3
"""
🎬 Video-based ASL Recognition using OpenHands
Process video files for batch ASL recognition with Tesla P40 optimization
"""

import cv2
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.webcam_demo import ASLWebcamRecognizer
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("📦 Install missing dependencies:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASLVideoProcessor:
    """Process video files for ASL recognition"""
    
    def __init__(self, model_path=None, confidence_threshold=0.7):
        """
        Initialize video processor
        
        Args:
            model_path: Path to OpenHands model
            confidence_threshold: Minimum confidence for predictions
        """
        self.recognizer = ASLWebcamRecognizer(model_path, confidence_threshold)
        
    def process_video(self, video_path, output_path=None, frame_skip=1):
        """
        Process video file for ASL recognition
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            frame_skip: Process every Nth frame (for speed)
            
        Returns:
            Dictionary with recognition results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"❌ Video file not found: {video_path}")
            return None
        
        logger.info(f"🎬 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video: {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"📊 Video info: {width}x{height}, {fps} FPS, {duration:.2f}s, {total_frames} frames")
        
        # Setup video writer if output requested
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            logger.info(f"📹 Will save annotated video to: {output_path}")
        
        # Process video
        results = {
            'video_path': str(video_path),
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration_seconds': duration
            },
            'recognition_results': [],
            'processing_info': {
                'frame_skip': frame_skip,
                'frames_processed': 0,
                'processing_time': 0,
                'average_fps': 0
            },
            'sentence_timeline': []
        }
        
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        logger.info(f"🎯 Processing every {frame_skip} frame(s)...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if requested
                if frame_count % frame_skip != 0:
                    if writer:
                        writer.write(frame)  # Write original frame
                    continue
                
                processed_count += 1
                current_time = frame_count / fps if fps > 0 else 0
                
                # Recognize ASL in frame
                result = self.recognizer.recognize_frame(frame)
                result['timestamp'] = current_time
                result['frame_number'] = frame_count
                
                results['recognition_results'].append(result)
                
                # Update recognition history for sentence building
                self.recognizer.update_recognition_history(result)
                
                # Draw results on frame
                annotated_frame = self.recognizer.draw_results(frame, result)
                
                # Add timeline info
                cv2.putText(annotated_frame, f"Time: {current_time:.1f}s Frame: {frame_count}", 
                           (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write annotated frame
                if writer:
                    writer.write(annotated_frame)
                
                # Progress update
                if processed_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    current_time_elapsed = time.time() - start_time
                    processing_fps = processed_count / current_time_elapsed if current_time_elapsed > 0 else 0
                    
                    logger.info(f"⏳ Progress: {progress:.1f}% ({processed_count}/{total_frames//frame_skip} frames processed, {processing_fps:.1f} FPS)")
                
                # Build sentence periodically
                if processed_count % 10 == 0:
                    sentence = self.recognizer.get_sentence()
                    if sentence:
                        results['sentence_timeline'].append({
                            'timestamp': current_time,
                            'sentence': sentence
                        })
        
        except KeyboardInterrupt:
            logger.info("🛑 Processing interrupted by user")
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
        finally:
            cap.release()
            if writer:
                writer.release()
        
        # Final processing info
        end_time = time.time()
        processing_time = end_time - start_time
        
        results['processing_info'].update({
            'frames_processed': processed_count,
            'processing_time': processing_time,
            'average_fps': processed_count / processing_time if processing_time > 0 else 0
        })
        
        logger.info(f"✅ Processing completed!")
        logger.info(f"📊 Processed {processed_count} frames in {processing_time:.2f}s ({results['processing_info']['average_fps']:.1f} FPS)")
        
        # Generate final sentence
        final_sentence = self.recognizer.get_sentence()
        results['final_sentence'] = final_sentence
        
        if final_sentence:
            logger.info(f"💬 Recognized sentence: '{final_sentence}'")
        else:
            logger.info("💬 No clear sentence detected")
        
        return results
    
    def save_results(self, results, output_path):
        """Save recognition results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Results saved to: {output_path}")
    
    def generate_summary(self, results):
        """Generate text summary of recognition results"""
        summary = []
        summary.append("🎬 ASL Video Recognition Summary")
        summary.append("=" * 40)
        summary.append("")
        
        # Video info
        video_info = results['video_info']
        summary.append(f"📹 Video: {results['video_path']}")
        summary.append(f"📊 Resolution: {video_info['width']}x{video_info['height']}")
        summary.append(f"⏱️  Duration: {video_info['duration_seconds']:.2f}s")
        summary.append(f"🎞️ Total frames: {video_info['total_frames']}")
        summary.append("")
        
        # Processing info
        proc_info = results['processing_info']
        summary.append(f"⚡ Processing speed: {proc_info['average_fps']:.1f} FPS")
        summary.append(f"🔄 Frames processed: {proc_info['frames_processed']}")
        summary.append(f"⏱️  Processing time: {proc_info['processing_time']:.2f}s")
        summary.append("")
        
        # Recognition results
        recognition_results = results['recognition_results']
        confident_predictions = [r for r in recognition_results if r.get('confidence', 0) > 0.7]
        unique_signs = set(r['prediction'] for r in confident_predictions if r.get('prediction'))
        
        summary.append(f"🎯 Confident predictions: {len(confident_predictions)}")
        summary.append(f"🔤 Unique signs detected: {len(unique_signs)}")
        
        if unique_signs:
            summary.append(f"📝 Signs: {', '.join(sorted(unique_signs))}")
        
        summary.append("")
        
        # Final sentence
        final_sentence = results.get('final_sentence', '')
        if final_sentence:
            summary.append(f"💬 Final sentence: '{final_sentence}'")
        else:
            summary.append("💬 No clear sentence detected")
        
        return "\n".join(summary)

def main():
    parser = argparse.ArgumentParser(description='Process video files for ASL recognition')
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--output-video', help='Output annotated video path')
    parser.add_argument('--output-json', help='Output JSON results path')
    parser.add_argument('--model', help='Path to custom OpenHands model')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ Input video not found: {input_path}")
        sys.exit(1)
    
    # Create processor
    processor = ASLVideoProcessor(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    try:
        # Process video
        results = processor.process_video(
            video_path=args.input,
            output_path=args.output_video,
            frame_skip=args.frame_skip
        )
        
        if not results:
            logger.error("❌ Video processing failed")
            sys.exit(1)
        
        # Save results
        if args.output_json:
            processor.save_results(results, args.output_json)
        
        # Print summary
        summary = processor.generate_summary(results)
        print("\n" + summary)
        
        print(f"\n✅ Video processing completed successfully!")
        
        if args.output_video:
            print(f"📹 Annotated video: {args.output_video}")
        if args.output_json:
            print(f"📄 Results JSON: {args.output_json}")
        
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
