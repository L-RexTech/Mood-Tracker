#!/bin/bash
set -e

echo "Setting up environment for Railway deployment..."

# Install NumPy first with specific version
pip install --no-cache-dir numpy==1.23.5

# Install scikit-learn with specific version
pip install --no-cache-dir scikit-learn==1.0.2

# Install core dependencies separately to avoid version conflicts
pip install --no-cache-dir fastapi==0.95.2 uvicorn==0.22.0 pydantic==1.10.8

# Install NLTK with specific working version
pip install --no-cache-dir nltk==3.8.1

# Install remaining dependencies
pip install --no-cache-dir pandas==1.5.3 matplotlib==3.7.1

# Create NLTK data directory
mkdir -p /app/nltk_data

# Download NLTK data with explicit path
python -c "import nltk; nltk.download('vader_lexicon', download_dir='/app/nltk_data')"

# Create empty __init__.py in nltk_data to make it importable
touch /app/nltk_data/__init__.py

# Train model to make sure it's available
echo "Pre-training model for deployment..."
python -c "from model import train_model_if_needed; train_model_if_needed()"

echo "Setup complete!"
