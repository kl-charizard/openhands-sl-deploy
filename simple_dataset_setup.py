#!/usr/bin/env python3
"""
🤟 Simple ASL Dataset Setup (No Authentication Required)
Creates directory structure and downloads publicly available dataset info
"""

import os
import requests
from pathlib import Path

def setup_directories():
    """Create directory structure for ASL training"""
    dirs = [
        'datasets/asl_alphabet',
        'datasets/wlasl',
        'datasets/processed',
        'models',
        'notebooks',
        'src'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def download_wlasl_info():
    """Download WLASL dataset information and scripts"""
    print("📥 Getting WLASL dataset information...")
    
    wlasl_urls = {
        'dataset_info': 'https://raw.githubusercontent.com/dxli94/WLASL/master/data/WLASL_v0.3.json',
        'download_script': 'https://raw.githubusercontent.com/dxli94/WLASL/master/scripts/video_download.py'
    }
    
    for name, url in wlasl_urls.items():
        try:
            response = requests.get(url)
            if response.status_code == 200:
                file_path = f'datasets/wlasl/{name}.{"json" if "json" in url else "py"}'
                with open(file_path, 'w') as f:
                    f.write(response.text)
                print(f"✅ Downloaded {name}")
            else:
                print(f"❌ Failed to download {name}")
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")

def create_download_instructions():
    """Create instructions for manual dataset downloads"""
    instructions = """
# 🤟 ASL Dataset Download Instructions

## Quick Start - Recommended Order:

### 1. Start with ASL Alphabet (Easiest)
**Dataset**: ASL-MNIST or Kaggle ASL Alphabet
**Purpose**: Learn fingerspelling (A-Z letters)
**Size**: 26K-87K images
**Download**: 
- Hugging Face: https://huggingface.co/datasets/Voxel51/American-Sign-Language-MNIST
- Kaggle: https://www.kaggle.com/datasets/grassknoted/asl-alphabet (requires Kaggle account)

### 2. Scale to Word Recognition 
**Dataset**: WLASL (Word-Level ASL)  
**Purpose**: Full ASL word recognition
**Size**: 12K videos, 2K words
**Download**: https://dxli94.github.io/WLASL/

### 3. Production-Ready Dataset
**Dataset**: ASL Citizen
**Purpose**: Real-world robustness
**Size**: 83K videos, 2.7K signs  
**Download**: https://www.microsoft.com/en-us/research/project/asl-citizen/

## Setup Steps:

1. **Choose a dataset** from above based on your goal
2. **Download manually** from the provided links  
3. **Extract to** the appropriate `datasets/` folder
4. **Start training** with the provided notebook templates

## Training Approach:

### Phase 1: Alphabet (1-2 weeks)
- Train CNN on A-Z letters
- Achieve 95%+ accuracy
- Deploy and test with camera

### Phase 2: Words (2-4 weeks)  
- Use WLASL dataset
- Start with 100 most common words
- Add temporal modeling for video sequences

### Phase 3: Production (1-2 months)
- Combine multiple datasets
- Add data augmentation  
- Optimize for mobile deployment

## Resources Created:
- 📚 `ASL_DATASETS_GUIDE.md` - Comprehensive dataset overview
- 📊 `notebooks/ASL_Training_Starter.ipynb` - Training templates
- 🏗️ Directory structure for organized development
"""
    
    with open('DATASET_DOWNLOAD_INSTRUCTIONS.md', 'w') as f:
        f.write(instructions)
    print("✅ Created dataset download instructions")

def main():
    """Main setup function"""
    print("🤟 Setting up ASL Training Environment (Simple Version)...")
    
    # Create directory structure
    setup_directories()
    
    # Download publicly available info
    print("\n📥 Downloading WLASL dataset information...")
    download_wlasl_info()
    
    # Create instructions  
    print("\n📋 Creating download instructions...")
    create_download_instructions()
    
    print("\n🎉 Setup complete! Next steps:")
    print("1. 📖 Read DATASET_DOWNLOAD_INSTRUCTIONS.md")
    print("2. 📖 Read ASL_DATASETS_GUIDE.md for detailed information")
    print("3. 🔍 Choose a dataset and download manually")
    print("4. 📊 Open notebooks/ASL_Training_Starter.ipynb to start training")

if __name__ == "__main__":
    main()
