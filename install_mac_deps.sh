#!/bin/bash

# Script to install macOS-specific dependencies for Mood Tracker

echo "Checking for macOS dependencies..."

# Check if running on macOS
if [[ $(uname) != "Darwin" ]]; then
    echo "This script is for macOS only."
    exit 0
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH if it's not already there
    if [[ -f ~/.zshrc ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -f ~/.bash_profile ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    
    echo "Homebrew installed successfully."
else
    echo "Homebrew is already installed."
fi

# Install libomp for XGBoost
echo "Installing OpenMP library (required for XGBoost)..."
brew install libomp

echo "Dependencies installed successfully!"
echo "You can now run: python3 train_model.py"
