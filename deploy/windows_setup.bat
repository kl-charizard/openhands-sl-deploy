@echo off
echo =====================================
echo 🚀 OpenHands ASL - Windows Setup
echo Tesla P40 Optimized ASL Recognition  
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    echo 📥 Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python detected
python --version

REM Check if we're in a virtual environment  
if not defined VIRTUAL_ENV (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
    
    echo 🔄 Activating virtual environment...
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ✅ Virtual environment already active
)

echo.
echo 📦 Installing OpenHands ASL dependencies...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install core dependencies (Windows compatible versions)
echo 🔧 Installing TensorFlow...
pip install tensorflow==2.12.0

echo 🔧 Installing OpenCV...  
pip install opencv-python==4.8.1.78

echo 🔧 Installing MediaPipe...
pip install mediapipe==0.10.8

echo 🔧 Installing other dependencies...
pip install -r requirements-windows.txt

echo.
echo 🤖 Installing OpenHands (experimental)...
echo ⚠️  Note: OpenHands may require manual installation on Windows

REM Try different OpenHands installation methods
echo 📥 Method 1: pip install...
pip install OpenHands

if %errorlevel% neq 0 (
    echo ⚠️  Method 1 failed, trying Method 2: git install...
    pip install git+https://github.com/AI4Bharat/OpenHands.git
    
    if %errorlevel% neq 0 (
        echo ⚠️  Method 2 failed, trying Method 3: local clone...
        
        REM Clone and install locally
        if not exist "temp_openhands" (
            git clone https://github.com/AI4Bharat/OpenHands.git temp_openhands
        )
        
        cd temp_openhands
        pip install -e .
        cd ..
        
        if %errorlevel% neq 0 (
            echo ❌ All OpenHands installation methods failed
            echo 💡 Manual installation may be required
            echo 📚 See: https://github.com/AI4Bharat/OpenHands
        ) else (
            echo ✅ OpenHands installed via local clone
        )
    ) else (
        echo ✅ OpenHands installed via git
    )
) else (
    echo ✅ OpenHands installed via pip
)

echo.
echo 🔍 Testing GPU detection...
echo.

python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('🔥 TensorFlow version:', tf.__version__)
print('🔥 GPUs detected:', len(gpus))
if gpus:
    print('🔥 GPU details:', gpus[0])
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print('✅ Tesla P40 memory growth configured')
    except Exception as e:
        print('⚠️  GPU configuration warning:', e)
else:
    print('⚠️  No GPU detected - using CPU mode')
print('✅ TensorFlow GPU test completed')
"

if %errorlevel% neq 0 (
    echo ❌ GPU test failed
    echo 💡 Troubleshooting:
    echo    1. Install CUDA 11.8 and cuDNN 8.6
    echo    2. Check NVIDIA drivers
    echo    3. Restart after CUDA installation
) else (
    echo ✅ GPU test passed
)

echo.
echo 🧪 Testing OpenHands import...
echo.

python -c "
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
    print('💡 You can still run the demo with dummy models')
"

echo.
echo 🎮 Testing webcam access...
echo.

python -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print('✅ Webcam access successful')
        print(f'✅ Camera resolution: {frame.shape[1]}x{frame.shape[0]}')
    else:
        print('⚠️  Webcam detected but cannot read frames')
    cap.release()
else:
    print('⚠️  No webcam detected or access denied')
    print('💡 Check camera permissions in Windows Settings')
"

echo.
echo =======================================
echo ✅ OpenHands ASL Setup Complete!
echo =======================================
echo.
echo 🚀 Quick Start Commands:
echo.
echo    📹 Real-time webcam demo:
echo       python src/webcam_demo.py
echo.  
echo    🧪 Test with sample video:
echo       python src/video_demo.py --input sample_videos/hello.mp4
echo.
echo    📊 Benchmark Tesla P40 performance:
echo       python src/benchmark_gpu.py
echo.
echo    📱 Create mobile models:
echo       python src/quantize_model.py --model models/openhands_asl.pkl --output mobile/models/
echo.
echo 📚 Documentation: See README.md
echo 🐛 Issues: https://github.com/your-username/openhands-asl-deploy/issues
echo.
echo Press any key to exit...
pause >nul
