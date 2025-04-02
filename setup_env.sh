#!/bin/bash

# Set up environment for the mood tracker application

# Check for Python installation
echo "Checking for Python installation..."

# Try different Python commands
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v py &> /dev/null; then
    PYTHON_CMD="py"
else
    echo "Error: Python not found. Please install Python 3.6+ and try again."
    echo "Visit https://www.python.org/downloads/ for installation instructions."
    exit 1
fi

echo "Found Python: $($PYTHON_CMD --version)"

# Check for pip
echo "Checking for pip..."
if $PYTHON_CMD -m pip --version &> /dev/null; then
    PIP_CMD="$PYTHON_CMD -m pip"
    echo "Using pip: $($PIP_CMD --version)"
else
    echo "Pip not found. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py --user
    if $PYTHON_CMD -m pip --version &> /dev/null; then
        PIP_CMD="$PYTHON_CMD -m pip"
        echo "Pip installed successfully: $($PIP_CMD --version)"
    else
        echo "Failed to install pip. Please install pip manually."
        exit 1
    fi
fi

# Install dependencies directly if venv creation fails
echo "Installing dependencies..."
$PIP_CMD install -r requirements.txt || {
    echo "Failed to install dependencies. Please check your internet connection and try again."
    exit 1
}

echo "Setup complete!"
echo ""
echo "To train the model, run: $PYTHON_CMD train_model.py"
echo "To start the API server, run: $PYTHON_CMD -m uvicorn app:app --reload"
echo ""
echo "Note: You can also use a virtual environment if you prefer."
