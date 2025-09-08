#!/usr/bin/env python3
"""
⚡ GPU Benchmark for Tesla P40
Test OpenHands ASL model performance on NVIDIA Tesla P40 (24GB VRAM)
"""

import time
import numpy as np
import tensorflow as tf
import argparse
import logging
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.gpu_optimizer import GPUOptimizer
    from src.pose_extractor import PoseExtractor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Make sure you're in the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Tesla40Benchmark:
    """Comprehensive Tesla P40 performance benchmark for ASL recognition"""
    
    def __init__(self):
        self.gpu_optimizer = GPUOptimizer()
        self.pose_extractor = PoseExtractor()
        self.results = {}
        
        # Test configurations
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        self.sequence_lengths = [64, 90, 128, 256]
        self.input_sizes = [158]  # Standard pose feature size
        
    def setup_tesla_p40(self):
        """Initialize Tesla P40 with optimal settings"""
        logger.info("🚀 Setting up Tesla P40 for benchmarking...")
        
        # Apply optimizations
        success = self.gpu_optimizer.optimize_for_inference()
        
        if not success:
            logger.error("❌ Failed to optimize GPU")
            return False
            
        # Get GPU info
        gpu_info = self.gpu_optimizer.gpu_info
        logger.info(f"🔥 GPU: {gpu_info['name']}")
        logger.info(f"🔥 Tesla P40 detected: {gpu_info['tesla_p40']}")
        logger.info(f"🔥 Available VRAM: {gpu_info['memory_mb']}MB")
        
        return True
    
    def create_dummy_model(self, input_size=158, output_size=2000):
        """Create dummy ASL model for benchmarking"""
        logger.info(f"🤖 Creating dummy ASL model ({input_size} → {output_size})...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("✅ Dummy model created")
        return model
    
    def benchmark_inference_speed(self, model, batch_sizes, num_runs=100):
        """Benchmark inference speed across different batch sizes"""
        logger.info("🏃 Benchmarking inference speed...")
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"📊 Testing batch size: {batch_size}")
            
            # Generate test data
            test_input = np.random.rand(batch_size, 158).astype(np.float32)
            
            # Warmup runs
            for _ in range(10):
                _ = model.predict(test_input, verbose=0)
            
            # Benchmark runs
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = model.predict(test_input, verbose=0)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            # Per-sample metrics
            avg_time_per_sample = avg_time / batch_size
            fps = batch_size / avg_time
            
            results[batch_size] = {
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'avg_time_per_sample_ms': avg_time_per_sample * 1000,
                'fps': fps,
                'throughput_samples_per_sec': batch_size / avg_time
            }
            
            logger.info(f"   ⚡ {avg_time*1000:.2f}±{std_time*1000:.2f}ms "
                       f"({fps:.1f} FPS, {avg_time_per_sample*1000:.2f}ms/sample)")
        
        return results
    
    def benchmark_memory_usage(self, model, max_batch_size=128):
        """Test memory usage with increasing batch sizes"""
        logger.info("🧠 Benchmarking memory usage...")
        
        results = {}
        
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            if batch_size > max_batch_size:
                break
                
            try:
                logger.info(f"📊 Testing memory with batch size: {batch_size}")
                
                # Generate test data
                test_input = np.random.rand(batch_size, 158).astype(np.float32)
                
                # Clear any existing memory
                tf.keras.backend.clear_session()
                
                # Get memory before inference
                try:
                    memory_before = tf.config.experimental.get_memory_info('GPU:0')
                    current_before = memory_before['current'] / 1024 / 1024  # MB
                except:
                    current_before = 0
                
                # Run inference
                _ = model.predict(test_input, verbose=0)
                
                # Get memory after inference
                try:
                    memory_after = tf.config.experimental.get_memory_info('GPU:0')
                    current_after = memory_after['current'] / 1024 / 1024  # MB
                    peak_mb = memory_after['peak'] / 1024 / 1024  # MB
                except:
                    current_after = current_before
                    peak_mb = current_before
                
                memory_used = current_after - current_before
                memory_per_sample = memory_used / batch_size if batch_size > 0 else 0
                
                results[batch_size] = {
                    'memory_used_mb': memory_used,
                    'memory_per_sample_mb': memory_per_sample,
                    'peak_memory_mb': peak_mb,
                    'current_memory_mb': current_after
                }
                
                logger.info(f"   🧠 Memory used: {memory_used:.1f}MB "
                           f"({memory_per_sample:.2f}MB/sample, Peak: {peak_mb:.1f}MB)")
                
            except tf.errors.ResourceExhaustedError as e:
                logger.warning(f"⚠️ OOM at batch size {batch_size}: {e}")
                break
            except Exception as e:
                logger.error(f"❌ Memory test failed at batch size {batch_size}: {e}")
                break
        
        return results
    
    def benchmark_precision_modes(self, model):
        """Compare FP32 vs FP16 vs INT8 performance"""
        logger.info("🎯 Benchmarking precision modes...")
        
        batch_size = 32
        num_runs = 50
        test_input = np.random.rand(batch_size, 158).astype(np.float32)
        
        results = {}
        
        # FP32 (default)
        logger.info("📊 Testing FP32 precision...")
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = model.predict(test_input, verbose=0)
            times.append(time.time() - start)
        
        results['fp32'] = {
            'avg_time_ms': np.mean(times) * 1000,
            'fps': batch_size / np.mean(times),
            'precision': 'FP32'
        }
        
        # Mixed Precision (FP16)
        logger.info("📊 Testing Mixed Precision (FP16)...")
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = model.predict(test_input, verbose=0)
                times.append(time.time() - start)
            
            results['fp16'] = {
                'avg_time_ms': np.mean(times) * 1000,
                'fps': batch_size / np.mean(times),
                'precision': 'Mixed FP16'
            }
            
            # Reset to default
            tf.keras.mixed_precision.set_global_policy('float32')
            
        except Exception as e:
            logger.warning(f"⚠️ FP16 test failed: {e}")
            results['fp16'] = {'error': str(e)}
        
        return results
    
    def benchmark_pose_extraction(self, num_frames=100):
        """Benchmark MediaPipe pose extraction performance"""
        logger.info("🖐️ Benchmarking pose extraction...")
        
        # Generate dummy frames (720p)
        frame_width, frame_height = 1280, 720
        frames = [
            np.random.randint(0, 255, (frame_height, frame_width, 3), dtype=np.uint8)
            for _ in range(num_frames)
        ]
        
        # Benchmark pose extraction
        times = []
        successful_extractions = 0
        
        for frame in frames:
            start_time = time.time()
            pose_data = self.pose_extractor.extract_pose(frame)
            end_time = time.time()
            
            times.append(end_time - start_time)
            if pose_data is not None:
                successful_extractions += 1
        
        avg_time = np.mean(times)
        success_rate = successful_extractions / num_frames
        fps = 1.0 / avg_time
        
        results = {
            'avg_extraction_time_ms': avg_time * 1000,
            'extraction_fps': fps,
            'success_rate': success_rate,
            'total_frames': num_frames,
            'successful_extractions': successful_extractions
        }
        
        logger.info(f"🖐️ Pose extraction: {avg_time*1000:.2f}ms/frame ({fps:.1f} FPS)")
        logger.info(f"🎯 Success rate: {success_rate*100:.1f}%")
        
        return results
    
    def run_full_benchmark(self):
        """Run complete Tesla P40 benchmark suite"""
        logger.info("🚀 Starting Tesla P40 benchmark suite...")
        
        # Setup Tesla P40
        if not self.setup_tesla_p40():
            logger.error("❌ GPU setup failed")
            return None
        
        # Create model
        model = self.create_dummy_model()
        
        # Run benchmarks
        benchmark_results = {
            'gpu_info': self.gpu_optimizer.get_performance_info(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tesla_p40_optimized': True
        }
        
        # Inference speed benchmark
        logger.info("\n" + "="*50)
        logger.info("⚡ INFERENCE SPEED BENCHMARK")
        logger.info("="*50)
        inference_results = self.benchmark_inference_speed(model, self.batch_sizes)
        benchmark_results['inference_speed'] = inference_results
        
        # Memory usage benchmark  
        logger.info("\n" + "="*50)
        logger.info("🧠 MEMORY USAGE BENCHMARK")
        logger.info("="*50)
        memory_results = self.benchmark_memory_usage(model)
        benchmark_results['memory_usage'] = memory_results
        
        # Precision modes benchmark
        logger.info("\n" + "="*50)
        logger.info("🎯 PRECISION MODES BENCHMARK")
        logger.info("="*50)
        precision_results = self.benchmark_precision_modes(model)
        benchmark_results['precision_modes'] = precision_results
        
        # Pose extraction benchmark
        logger.info("\n" + "="*50)
        logger.info("🖐️ POSE EXTRACTION BENCHMARK")
        logger.info("="*50)
        pose_results = self.benchmark_pose_extraction()
        benchmark_results['pose_extraction'] = pose_results
        
        return benchmark_results
    
    def generate_report(self, results, output_dir="benchmark_results"):
        """Generate comprehensive benchmark report"""
        logger.info("📊 Generating benchmark report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save JSON results
        json_path = output_path / f"tesla_p40_benchmark_{int(time.time())}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate text report
        report_path = output_path / f"tesla_p40_report_{int(time.time())}.txt"
        with open(report_path, 'w') as f:
            self._write_text_report(f, results)
        
        # Generate plots
        self._generate_plots(results, output_path)
        
        logger.info(f"📊 Benchmark results saved to: {output_path}")
        logger.info(f"📄 JSON: {json_path}")
        logger.info(f"📄 Report: {report_path}")
        
        return output_path
    
    def _write_text_report(self, f, results):
        """Write formatted text report"""
        f.write("🚀 Tesla P40 ASL Recognition Benchmark Report\n")
        f.write("=" * 50 + "\n\n")
        
        # GPU Info
        if 'gpu_info' in results:
            gpu_info = results['gpu_info']
            f.write(f"GPU: {gpu_info.get('gpu_name', 'Unknown')}\n")
            f.write(f"Optimizations: {gpu_info.get('optimizations_applied', False)}\n")
            f.write(f"Timestamp: {results.get('timestamp', 'Unknown')}\n\n")
        
        # Inference Speed Results
        if 'inference_speed' in results:
            f.write("⚡ INFERENCE SPEED RESULTS\n")
            f.write("-" * 30 + "\n")
            for batch_size, metrics in results['inference_speed'].items():
                f.write(f"Batch Size {batch_size}:\n")
                f.write(f"  Average: {metrics['avg_time_ms']:.2f}ms\n")
                f.write(f"  FPS: {metrics['fps']:.1f}\n")
                f.write(f"  Per sample: {metrics['avg_time_per_sample_ms']:.2f}ms\n\n")
        
        # Memory Usage Results
        if 'memory_usage' in results:
            f.write("🧠 MEMORY USAGE RESULTS\n")
            f.write("-" * 30 + "\n")
            for batch_size, metrics in results['memory_usage'].items():
                f.write(f"Batch Size {batch_size}:\n")
                f.write(f"  Memory used: {metrics['memory_used_mb']:.1f}MB\n")
                f.write(f"  Per sample: {metrics['memory_per_sample_mb']:.2f}MB\n")
                f.write(f"  Peak: {metrics['peak_memory_mb']:.1f}MB\n\n")
        
        # Pose Extraction Results
        if 'pose_extraction' in results:
            pose_results = results['pose_extraction']
            f.write("🖐️ POSE EXTRACTION RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average time: {pose_results['avg_extraction_time_ms']:.2f}ms\n")
            f.write(f"FPS: {pose_results['extraction_fps']:.1f}\n")
            f.write(f"Success rate: {pose_results['success_rate']*100:.1f}%\n\n")
    
    def _generate_plots(self, results, output_path):
        """Generate benchmark visualization plots"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        # Inference speed plot
        if 'inference_speed' in results:
            plt.figure(figsize=(12, 8))
            
            batch_sizes = list(results['inference_speed'].keys())
            fps_values = [results['inference_speed'][bs]['fps'] for bs in batch_sizes]
            
            plt.subplot(2, 2, 1)
            plt.plot(batch_sizes, fps_values, 'o-', linewidth=2, markersize=8)
            plt.title('Tesla P40 Inference Speed (FPS)')
            plt.xlabel('Batch Size')
            plt.ylabel('FPS')
            plt.grid(True, alpha=0.3)
            
            # Memory usage plot
            if 'memory_usage' in results:
                memory_batch_sizes = list(results['memory_usage'].keys())
                memory_values = [results['memory_usage'][bs]['memory_used_mb'] for bs in memory_batch_sizes]
                
                plt.subplot(2, 2, 2)
                plt.plot(memory_batch_sizes, memory_values, 's-', color='orange', linewidth=2, markersize=8)
                plt.title('Tesla P40 Memory Usage')
                plt.xlabel('Batch Size')
                plt.ylabel('Memory Used (MB)')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = output_path / "tesla_p40_performance_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Plots saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Plot generation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Tesla P40 ASL Recognition Benchmark')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64],
                       help='Batch sizes to test')
    parser.add_argument('--runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--output', default='benchmark_results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create benchmark instance
    benchmark = Tesla40Benchmark()
    
    if args.quick:
        benchmark.batch_sizes = [1, 8, 32]
        logger.info("🏃 Running quick benchmark...")
    
    try:
        # Run benchmark
        results = benchmark.run_full_benchmark()
        
        if results:
            # Generate report
            output_path = benchmark.generate_report(results, args.output)
            
            print("\n" + "🎉" * 20)
            print("✅ Tesla P40 benchmark completed successfully!")
            print(f"📊 Results saved to: {output_path}")
            print("🎉" * 20)
        else:
            print("❌ Benchmark failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
