#!/bin/bash

echo "======================================="
echo "🍎 OpenHands ASL - macOS Setup"
echo "Multi-platform ASL Recognition"  
echo "Supports: Intel Mac, Apple Silicon, CPU"
echo "======================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found! Please install Python 3.8+ first.${NC}"
    echo "📥 Install via Homebrew: brew install python"
    echo "📥 Or download from: https://www.python.org/downloads/"
    exit 1
fi

echo -e "${GREEN}✅ Python detected${NC}"
python3 --version

# Detect Mac architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo -e "${BLUE}🖥️  Apple Silicon (M1/M2) detected${NC}"
    MAC_TYPE="apple_silicon"
elif [[ "$ARCH" == "x86_64" ]]; then
    echo -e "${BLUE}🖥️  Intel Mac detected${NC}"
    MAC_TYPE="intel"
else
    echo -e "${YELLOW}⚠️  Unknown architecture: $ARCH${NC}"
    MAC_TYPE="unknown"
fi

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
    
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already active${NC}"
fi

echo ""
echo "📦 Installing OpenHands ASL dependencies for macOS..."
echo ""

# Upgrade pip first
python3 -m pip install --upgrade pip

# Install core dependencies based on Mac type
echo "🔧 Installing TensorFlow for macOS..."

if [[ "$MAC_TYPE" == "apple_silicon" ]]; then
    echo -e "${BLUE}🚀 Installing TensorFlow for Apple Silicon...${NC}"
    pip install tensorflow-macos>=2.12.0
    
    echo -e "${BLUE}🔧 Installing Metal GPU acceleration (optional)...${NC}"
    pip install tensorflow-metal || echo -e "${YELLOW}⚠️  Metal acceleration install failed (will use CPU)${NC}"
    
elif [[ "$MAC_TYPE" == "intel" ]]; then
    echo -e "${BLUE}🔧 Installing TensorFlow for Intel Mac...${NC}"
    pip install tensorflow>=2.12.0
else
    echo -e "${YELLOW}🔧 Installing standard TensorFlow...${NC}"
    pip install tensorflow>=2.12.0
fi

echo "🔧 Installing OpenCV and MediaPipe..."  
pip install opencv-python>=4.8.0
pip install mediapipe>=0.10.0

echo "🔧 Installing other dependencies..."
pip install -r requirements-mac.txt

echo ""
echo "🤖 Installing OpenHands (experimental)..."
echo -e "${YELLOW}⚠️  Note: OpenHands may require manual installation${NC}"

# Try different OpenHands installation methods
echo "📥 Method 1: pip install..."
pip install OpenHands

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Method 1 failed, trying Method 2: git install...${NC}"
    pip install git+https://github.com/AI4Bharat/OpenHands.git
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️  Method 2 failed, trying Method 3: local clone...${NC}"
        
        # Clone and install locally
        if [ ! -d "temp_openhands" ]; then
            git clone https://github.com/AI4Bharat/OpenHands.git temp_openhands
        fi
        
        cd temp_openhands
        pip install -e .
        cd ..
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ All OpenHands installation methods failed${NC}"
            echo -e "${YELLOW}💡 Manual installation may be required${NC}"
            echo "📚 See: https://github.com/AI4Bharat/OpenHands"
        else
            echo -e "${GREEN}✅ OpenHands installed via local clone${NC}"
        fi
    else
        echo -e "${GREEN}✅ OpenHands installed via git${NC}"
    fi
else
    echo -e "${GREEN}✅ OpenHands installed via pip${NC}"
fi

echo ""
echo "🔍 Testing TensorFlow installation..."
echo ""

python3 -c "
import tensorflow as tf
print('🔥 TensorFlow version:', tf.__version__)

# Check for GPU support
gpus = tf.config.list_physical_devices('GPU')
print('🔥 GPUs detected:', len(gpus))

if len(gpus) > 0:
    print('🔥 GPU details:', gpus)
    try:
        # Try to configure GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('✅ GPU memory growth configured')
    except Exception as e:
        print('⚠️  GPU configuration warning:', e)
else:
    print('💻 Running in CPU mode')
    
# Check for Apple Silicon Metal support
if hasattr(tf.config, 'experimental') and hasattr(tf.config.experimental, 'list_logical_devices'):
    try:
        metal_devices = [d for d in tf.config.experimental.list_logical_devices() if 'GPU' in d.name]
        if metal_devices:
            print('🚀 Apple Silicon Metal GPU acceleration available')
    except:
        pass

print('✅ TensorFlow test completed')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ TensorFlow test failed${NC}"
    echo -e "${YELLOW}💡 Troubleshooting for macOS:${NC}"
    echo "   1. Make sure you're using the right TensorFlow version for your Mac"
    echo "   2. For Apple Silicon: tensorflow-macos + tensorflow-metal"
    echo "   3. For Intel Mac: standard tensorflow package"
    echo "   4. CPU-only mode will still work for testing"
else
    echo -e "${GREEN}✅ TensorFlow test passed${NC}"
fi

echo ""
echo "🧪 Testing OpenHands import..."
echo ""

python3 -c "
try:
    import openhands
    print('✅ OpenHands imported successfully')
    try:
        print('📚 OpenHands version:', openhands.__version__)
    except:
        print('📚 OpenHands version: Unknown')
except ImportError as e:
    print('❌ OpenHands import failed:', e)
    print('💡 This is expected if OpenHands installation failed')
    print('💡 You can still run the demo with CPU-based models')
"

echo ""
echo "🎮 Testing camera access..."
echo ""

python3 -c "
import cv2
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print('✅ Camera access successful')
            print(f'✅ Camera resolution: {frame.shape[1]}x{frame.shape[0]}')
        else:
            print('⚠️  Camera detected but cannot read frames')
        cap.release()
    else:
        print('⚠️  No camera detected or access denied')
        print('💡 Check camera permissions in System Preferences > Security & Privacy > Camera')
except Exception as e:
    print('❌ Camera test failed:', e)
    print('💡 Camera access may require additional permissions on macOS')
"

echo ""
echo "======================================="
echo -e "${GREEN}✅ OpenHands ASL Setup Complete!${NC}"
echo "======================================="
echo ""
echo -e "${BLUE}🚀 Quick Start Commands:${NC}"
echo ""
echo "   📹 Real-time webcam demo:"
echo "      python3 src/webcam_demo.py"
echo ""  
echo "   🧪 Test with sample video:"
echo "      python3 src/video_demo.py --input sample_videos/hello.mp4"
echo ""
echo "   📊 Benchmark performance:"
echo "      python3 src/benchmark_gpu.py"
echo ""
echo "   📱 Create mobile models:"
echo "      python3 src/quantize_model.py --model models/openhands_asl.pkl --output mobile/models/"
echo ""
echo "📚 Documentation: See README.md"
echo "🐛 Issues: https://github.com/kl-charizard/openhands-sl-deploy/issues"
echo ""
echo -e "${GREEN}Press any key to exit...${NC}"
read -n 1
