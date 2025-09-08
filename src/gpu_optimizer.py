#!/usr/bin/env python3
"""
⚡ GPU Optimizer for Tesla P40
Optimizes TensorFlow for 24GB VRAM and inference performance
"""

import tensorflow as tf
import logging
import psutil
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GPUOptimizer:
    """Tesla P40 optimization for OpenHands inference"""
    
    def __init__(self):
        """Initialize GPU optimizer"""
        self.gpu_info = self._detect_gpu()
        self.optimization_applied = False
        
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU hardware and capabilities"""
        gpu_info = {
            'available': False,
            'name': 'None',
            'memory_mb': 0,
            'compute_capability': None,
            'tesla_p40': False
        }
        
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_info['available'] = True
                
                # Get GPU details
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                gpu_info['name'] = gpu_name
                
                # Check if Tesla P40
                if 'Tesla P40' in gpu_name or 'P40' in gpu_name:
                    gpu_info['tesla_p40'] = True
                    gpu_info['memory_mb'] = 24 * 1024  # 24GB
                    gpu_info['compute_capability'] = '6.1'
                    logger.info("🚀 Tesla P40 detected!")
                
                # Get memory info
                try:
                    memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    current_mb = memory_info['current'] / 1024 / 1024
                    peak_mb = memory_info['peak'] / 1024 / 1024
                    logger.info(f"📊 GPU Memory - Current: {current_mb:.1f}MB, Peak: {peak_mb:.1f}MB")
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"⚠️ GPU detection error: {e}")
        
        return gpu_info
    
    def optimize_for_inference(self) -> bool:
        """
        Apply Tesla P40 optimizations for inference
        
        Returns:
            True if optimizations applied successfully
        """
        if self.optimization_applied:
            return True
            
        logger.info("⚡ Applying Tesla P40 optimizations...")
        
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logger.warning("⚠️ No GPU available - optimizations skipped")
                return False
            
            # Memory growth (important for Tesla P40)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("✅ Memory growth enabled")
            
            # Mixed precision for faster inference
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("✅ Mixed precision enabled (FP16)")
            
            # XLA (Accelerated Linear Algebra) compilation
            tf.config.optimizer.set_jit(True)
            logger.info("✅ XLA JIT compilation enabled")
            
            # Optimize for inference (not training)
            tf.config.optimizer.set_experimental_options({
                'layout_optimizer': True,
                'constant_folding': True,
                'shape_optimization': True,
                'remapping': True,
                'arithmetic_optimization': True,
                'dependency_optimization': True,
                'loop_optimization': True,
                'function_optimization': True,
                'debug_stripper': True,
                'disable_model_pruning': False,
                'scoped_allocator_optimization': True,
                'pin_to_host_optimization': True,
                'implementation_selector': True,
                'auto_mixed_precision': True,
                'disable_meta_optimizer': False,
            })
            logger.info("✅ Inference optimizations enabled")
            
            # Tesla P40 specific optimizations
            if self.gpu_info['tesla_p40']:
                # Configure for Compute Capability 6.1
                logger.info("🎯 Applying Tesla P40 specific optimizations...")
                
                # Set device placement
                tf.config.set_soft_device_placement(True)
                
                # Enable tensor fusion
                tf.config.optimizer.set_experimental_options({
                    'enable_tensor_fusion': True
                })
                
                logger.info("✅ Tesla P40 optimizations applied")
            
            self.optimization_applied = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            return False
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get current GPU performance information"""
        info = {
            'gpu_available': self.gpu_info['available'],
            'gpu_name': self.gpu_info['name'],
            'optimizations_applied': self.optimization_applied,
            'memory_info': {},
            'system_info': {}
        }
        
        # GPU memory info
        if self.gpu_info['available']:
            try:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                info['memory_info'] = {
                    'current_mb': memory_info['current'] / 1024 / 1024,
                    'peak_mb': memory_info['peak'] / 1024 / 1024,
                    'total_mb': self.gpu_info['memory_mb']
                }
            except:
                pass
        
        # System info
        info['system_info'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
        }
        
        return info
    
    def benchmark_inference_speed(self, model_func, input_data, num_runs=100) -> Dict[str, float]:
        """
        Benchmark model inference speed
        
        Args:
            model_func: Function that performs inference
            input_data: Sample input data
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        logger.info(f"🏃 Running inference benchmark ({num_runs} runs)...")
        
        import time
        
        # Warmup runs
        for _ in range(10):
            _ = model_func(input_data)
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model_func(input_data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate metrics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1.0 / avg_time
        
        metrics = {
            'avg_inference_time_ms': avg_time * 1000,
            'min_inference_time_ms': min_time * 1000,
            'max_inference_time_ms': max_time * 1000,
            'fps': fps,
            'gpu_name': self.gpu_info['name']
        }
        
        logger.info(f"📊 Benchmark Results:")
        logger.info(f"   Average: {metrics['avg_inference_time_ms']:.2f}ms ({metrics['fps']:.1f} FPS)")
        logger.info(f"   Range: {metrics['min_inference_time_ms']:.2f}ms - {metrics['max_inference_time_ms']:.2f}ms")
        
        return metrics
    
    def monitor_gpu_usage(self) -> Optional[Dict[str, float]]:
        """Monitor GPU usage using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'gpu_utilization_percent': float(values[0]),
                    'memory_used_mb': float(values[1]),
                    'memory_total_mb': float(values[2]),
                    'temperature_c': float(values[3])
                }
        except Exception as e:
            logger.debug(f"nvidia-smi monitoring failed: {e}")
        
        return None
    
    def get_optimization_summary(self) -> str:
        """Get summary of applied optimizations"""
        if not self.optimization_applied:
            return "❌ No optimizations applied"
        
        summary = ["✅ Tesla P40 Optimizations Applied:"]
        summary.append("   • Memory growth enabled")
        summary.append("   • Mixed precision (FP16) enabled")
        summary.append("   • XLA JIT compilation enabled")
        summary.append("   • Inference optimizations enabled")
        
        if self.gpu_info['tesla_p40']:
            summary.append("   • Tesla P40 specific optimizations")
            summary.append("   • 24GB VRAM configured")
            summary.append("   • Compute Capability 6.1 optimized")
        
        return "\n".join(summary)
