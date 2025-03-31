#!/bin/bash
# Script to set up Mood Tracker on PythonAnywhere

# Print colorful messages
print_green() {
    echo -e "\e[32m$1\e[0m"
}

print_yellow() {
    echo -e "\e[33m$1\e[0m"
}

print_red() {
    echo -e "\e[31m$1\e[0m"
}

print_blue() {
    echo -e "\e[34m$1\e[0m"
}

# Create directory structure
print_blue "=== Setting up Mood Tracker ==="
print_yellow "Creating project directory..."

# Create the main project directory
mkdir -p ~/mood-tracker

# Clone the repository if GitHub URL is provided
if [ "$1" != "" ]; then
    print_yellow "Cloning repository from $1..."
    git clone "$1" ~/mood-tracker
    
    if [ $? -ne 0 ]; then
        print_red "Error cloning repository. Please check the URL and try again."
        print_yellow "Continuing with manual setup..."
    else
        print_green "Repository cloned successfully!"
        cd ~/mood-tracker
    fi
else
    # Navigate to the project directory
    cd ~/mood-tracker
    print_green "Directory created: ~/mood-tracker"
    print_yellow "You'll need to upload your files manually or clone your repository."
fi

# Create virtual environment
print_yellow "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    print_red "Error creating virtual environment. Trying another method..."
    virtualenv -p python3 venv
    
    if [ $? -ne 0 ]; then
        print_red "Failed to create virtual environment. Please check your Python installation."
        exit 1
    fi
fi

print_green "Virtual environment created successfully!"

# Activate virtual environment
source venv/bin/activate
print_green "Virtual environment activated!"

# Create basic directory structure
mkdir -p static
mkdir -p plots

# Create a basic README if it doesn't exist
if [ ! -f README.md ]; then
    echo "# Mood Tracker" > README.md
    echo "A FastAPI application for tracking and predicting mood based on daily habits." >> README.md
fi

# Create a basic requirements.txt if it doesn't exist
if [ ! -f requirements.txt ]; then
    cat > requirements.txt << EOL
fastapi>=0.68.0
pydantic>=1.8.0
uvicorn>=0.15.0
nltk>=3.6.0
scikit-learn==1.0.2
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
python-multipart>=0.0.5
uvicorn-gunicorn-fastapi>=0.1.0
EOL
    print_yellow "Created basic requirements.txt - you may need to customize it."
fi

# Create WSGI adapter if not exists
if [ ! -f wsgi_adapter.py ]; then
    cat > wsgi_adapter.py << EOL
from fastapi.middleware.wsgi import WSGIMiddleware
from app import app

# Create a WSGI application from the FastAPI app
application = WSGIMiddleware(app)
EOL
    print_yellow "Created wsgi_adapter.py"
fi

print_blue "=== Setup Complete ==="
print_yellow "Next steps:"
echo "1. Install requirements: pip install -r requirements.txt"
echo "2. Configure a web app in the PythonAnywhere web tab"
echo "3. Set your WSGI file to use wsgi_adapter.py"
echo "4. Reload your web app"

print_green "Current directory: $(pwd)"
print_green "Project directory: ~/mood-tracker"
