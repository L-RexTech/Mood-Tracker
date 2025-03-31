#!/bin/bash
set -e

echo "Setting up environment for Railway deployment..."

# Install core packages first, avoiding joblib-related issues
pip install --no-cache-dir fastapi uvicorn pydantic nltk

# Download NLTK data early
python -c "import nltk; nltk.download('vader_lexicon')"

# Install sklearn separately with specific version known to work well in Railway
pip install --no-cache-dir scikit-learn==1.0.2

# Install remaining dependencies
pip install --no-cache-dir pandas numpy matplotlib

# Train model to make sure it's available
echo "Pre-training model for deployment..."
python train_model.py --samples 1000

echo "Setup complete!"
