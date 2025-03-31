# Mood Tracker API

This API predicts a mood score (0-10) based on daily habits and provides personalized recommendations for improvement.

## Setup and Running Instructions

### 1. Set Up the Environment

#### Option 1: Using the setup script (Recommended)
```bash
# Make the setup script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

This script will:
- Find the available Python installation (python3, python, or py)
- Create a virtual environment
- Activate the environment
- Install all dependencies

#### Option 2: Manual setup with pip
```bash
# Install dependencies directly if you have pip3
python3 -m pip install -r requirements.txt

# If pip3 isn't available, you can install it
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
python3 -m pip install -r requirements.txt
```

#### Option 3: Manual setup with virtual environment
```bash
# If python command isn't found, try python3
python3 -m venv venv
# Or if using virtualenv
python3 -m pip install virtualenv
python3 -m virtualenv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python3 -m pip install -r requirements.txt
```

### macOS Specific Setup

If you encounter an error about "libomp.dylib not found" when running the model, you need to install OpenMP:

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenMP library
brew install libomp

# Then try running the model again
python3 train_model.py
```

### 2. Generate Data and Train the Model

```bash
# Generate synthetic data and train the regression model
python3 train_model.py

# Options:
# Generate more samples: python3 train_model.py --samples 5000
# Force regenerate data: python3 train_model.py --force
# Only evaluate existing model: python3 train_model.py --evaluate
# Show accuracy metrics: python3 train_model.py --accuracy
```

This process will:
- Generate synthetic training data
- Split data into training and test sets
- Train a Random Forest regression model
- Evaluate model performance
- Create visualizations in the 'plots' directory

### 3. Run the API Server

```bash
# Start the FastAPI server
python3 -m uvicorn app:app --reload
```

The API will automatically use the trained model. If no model exists, it will generate data and train one on startup.

### 4. Use the API

- Access the API documentation: http://localhost:8000/docs
- Test the API directly through the Swagger UI
- Or send POST requests to http://localhost:8000/predict

## Model Performance

The mood prediction model typically achieves:
- R-squared (R²): ~0.85-0.95 (higher is better, 1.0 is perfect)
- Mean Absolute Error (MAE): ~0.3-0.5 points on the 0-10 scale
- 85-95% of predictions within 1 point of the actual mood score

### Understanding Accuracy Metrics

When evaluating the model, you'll see metrics like:

## API Usage

### Endpoint: `/predict`

**Request:**
```json
{
  "day_rating": "Very stressful, too much work",
  "water_intake": 1.5,
  "people_met": 2,
  "exercise": 30,
  "sleep": 5,
  "screen_time": 6,
  "outdoor_time": 1,
  "stress_level": "High",
  "food_quality": "Moderate"
}
```

**Response:**
```json
{
  "mood_score": 4.8,
  "recommendations": [
    "Insufficient sleep can affect mood. Aim for 7-9 hours of quality sleep.",
    "Consider reducing screen time, especially before bed, to improve mood and sleep quality.",
    "Practice deep breathing exercises for 5 minutes when feeling stressed.",
    "Try meditation to manage stress and improve mindfulness.",
    "Incorporate more fruits and vegetables into your diet for better mood and energy."
  ]
}
```

## Project Structure

- `app.py`: FastAPI application with API endpoints
- `model.py`: Contains the mood prediction model logic
- `recommendations.py`: Generates personalized recommendations
- `data_generator.py`: Generates synthetic training data
- `train_model.py`: Script for training and evaluating the model
- `setup_env.sh`: Helper script to set up the Python environment
- `requirements.txt`: Required Python packages
- `mood_model.pkl`: Trained model file (created after training)
- `model_metrics.json`: Performance metrics for the trained model
- `plots/`: Directory containing model performance visualizations

## Troubleshooting

### "No module named X" errors
If you encounter errors like "No module named numpy", ensure dependencies are installed:
```bash
# Install dependencies directly
python3 -m pip install -r requirements.txt

# If pip is not found, install it first
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

### If "pip: command not found" error
Use the Python module form of pip instead:
```bash
python3 -m pip install -r requirements.txt
```

### If Python isn't installed
1. Check your Python installation with `python3 --version`
2. Use the `setup_env.sh` script which tries different Python commands
3. If Python isn't installed, download it from https://www.python.org/downloads/
