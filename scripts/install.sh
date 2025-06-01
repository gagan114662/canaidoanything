#!/bin/bash

# Garment Creative AI - Installation Script

echo "🎨 Garment Creative AI - Installation Script"
echo "============================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version by default)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Ask about GPU support
read -p "🎮 Do you have NVIDIA GPU and want GPU support? (y/n): " gpu_support

if [ "$gpu_support" = "y" ] || [ "$gpu_support" = "Y" ]; then
    echo "🚀 Installing PyTorch with CUDA support..."
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install requirements
echo "📋 Installing Python dependencies..."
pip install -r requirements.txt

# Install additional dependencies that might not be in requirements.txt
echo "🔧 Installing additional dependencies..."

# Real-ESRGAN
echo "  - Installing Real-ESRGAN..."
pip install realesrgan

# SAM2 (if available)
echo "  - Attempting to install SAM2..."
pip install git+https://github.com/facebookresearch/segment-anything-2.git || echo "⚠️  SAM2 installation failed, will use fallback methods"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p app/static/uploads
mkdir -p app/static/outputs
mkdir -p logs
mkdir -p models

# Copy environment file
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

# Make scripts executable
echo "🔐 Making scripts executable..."
chmod +x scripts/*.sh

# Check Redis installation
echo "🔍 Checking Redis..."
if command -v redis-server &> /dev/null; then
    echo "✅ Redis is installed"
else
    echo "❌ Redis is not installed. Please install Redis:"
    echo "   Ubuntu/Debian: sudo apt install redis-server"
    echo "   macOS: brew install redis"
    echo "   Or use Docker: docker run -d -p 6379:6379 redis:alpine"
fi

# Download models (optional)
read -p "📥 Do you want to download AI models now? This may take a while (y/n): " download_models

if [ "$download_models" = "y" ] || [ "$download_models" = "Y" ]; then
    echo "📥 Downloading models..."
    python3 -c "
import torch
from transformers import AutoTokenizer
from diffusers import FluxPipeline

print('Downloading FLUX model...')
try:
    # This will download the model to cache
    pipeline = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.float16)
    print('✅ FLUX model downloaded')
except Exception as e:
    print(f'⚠️  FLUX model download failed: {e}')

print('Downloading other models...')
# Add other model downloads here
"
fi

echo ""
echo "🎉 Installation completed!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Start Redis server if not running"
echo "3. Run: ./scripts/start.sh"
echo ""
echo "For Docker deployment:"
echo "1. docker-compose up -d"
echo ""
echo "API will be available at: http://localhost:8000"
echo "Flower (Celery monitoring): http://localhost:5555"