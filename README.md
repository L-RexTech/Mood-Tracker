# Mood Tracker API

This API predicts a mood score (0-10) based on daily habits and provides personalized recommendations for improvement.

## Setup and Running Instructions

### 1. Set Up the Environment

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data and Train the Model

```bash
# Generate synthetic data and train the regression model
python train_model.py

# Options:
# Generate more samples: python train_model.py --samples 5000
# Force regenerate data: python train_model.py --force
# Only evaluate existing model: python train_model.py --evaluate
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
uvicorn app:app --reload
```

The API will automatically use the trained model. If no model exists, it will generate data and train one on startup.

### 4. Use the API

- Access the API documentation: http://localhost:8000/docs
- Test the API directly through the Swagger UI
- Or send POST requests to http://localhost:8000/predict

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
- `requirements.txt`: Required Python packages
- `mood_model.pkl`: Trained model file (created after training)
