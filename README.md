# Mood Tracker API

A machine learning-powered API that predicts mood scores (0-10) based on daily habits and lifestyle factors, providing personalized recommendations for mental health improvement. The system uses a Random Forest regression model with ensemble learning to analyze the relationship between various lifestyle factors and mood patterns.

## Features

- **Machine Learning Model**: Uses Random Forest with Gradient Boosting ensemble for accurate mood prediction
- **Sentiment Analysis**: Analyzes text descriptions of your day using NLTK sentiment analysis
- **Personalized Recommendations**: Provides tailored suggestions based on your lifestyle patterns
- **Performance Visualization**: Generates charts showing model performance and feature importance
- **RESTful API**: FastAPI-based API with automatic documentation
- **CORS Support**: Ready for web application integration

## Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation & Setup

#### Option 1: Automated Setup (Recommended)
```bash
# Make the setup script executable (Linux/macOS)
chmod +x setup_env.sh
./setup_env.sh

# For Windows PowerShell
powershell -ExecutionPolicy Bypass -File setup_env.sh
```

#### Option 2: Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Train the Model
```bash
# Generate synthetic training data and train the model
python train_model.py

# Advanced options:
python train_model.py --samples 5000    # Generate more training samples
python train_model.py --force           # Force regenerate training data
python train_model.py --evaluate        # Only evaluate existing model
python train_model.py --accuracy        # Show detailed accuracy metrics
```

This process will:
- Generate realistic synthetic training data (3000+ samples by default)
- Split data into training and test sets (80/20 split)
- Train an ensemble model (Random Forest + Gradient Boosting)
- Evaluate model performance with multiple metrics
- Create performance visualizations in the `plots/` directory
- Save the trained model as `mood_model.pkl`

#### 2. Start the API Server
```bash
# Start the FastAPI server with auto-reload
python -m uvicorn app:app --reload

# Or run directly
python app.py
```

The API will be available at:
- **API Endpoint**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

#### 3. Making Predictions
You can use the API through:
- The interactive Swagger UI at `/docs`
- Direct HTTP POST requests to `/predict`
- Integration with web applications (CORS enabled)

## Model Performance

The current trained model achieves:
- **R-squared (R²)**: 0.809 (81% of mood variance explained)
- **Mean Absolute Error (MAE)**: 0.516 points on the 0-10 scale
- **Root Mean Square Error (RMSE)**: 0.696
- **Accuracy**: 87.6% of predictions within 1 point of actual mood score
- **High Precision**: 58.4% of predictions within 0.5 points

### Model Architecture
- **Primary Model**: Random Forest Regressor with 100 estimators
- **Ensemble Component**: Gradient Boosting Regressor
- **Feature Engineering**: Polynomial features and sentiment analysis
- **Input Processing**: One-hot encoding for categorical variables, standard scaling for numerical features

### Key Features Analyzed
1. **Day Rating Sentiment**: NLP analysis of daily experience descriptions
2. **Sleep Duration**: Hours of sleep (4-11 hours range)
3. **Exercise**: Physical activity duration in minutes
4. **Social Interaction**: Number of people met
5. **Screen Time**: Digital device usage hours
6. **Outdoor Time**: Time spent outside
7. **Water Intake**: Hydration levels (0.5-5 liters)
8. **Stress Level**: Categorical (Low/Medium/High)
9. **Food Quality**: Nutritional quality (Healthy/Moderate/Unhealthy)

## API Usage

### Main Endpoint: `POST /predict`

Predicts mood score based on daily habits and provides personalized recommendations.

**Request Body Example:**
```json
{
  "day_rating": "Had a great day at work, met some interesting people",
  "water_intake": 2.5,
  "people_met": 5,
  "exercise": 45,
  "sleep": 7.5,
  "screen_time": 4,
  "outdoor_time": 2,
  "stress_level": "Low",
  "food_quality": "Healthy"
}
```

**Response Example:**
```json
{
  "mood_score": 7.8,
  "recommendations": [
    "Great job on staying hydrated! Keep up the good water intake.",
    "Your exercise routine is excellent for maintaining good mood.",
    "Consider spending a bit more time outdoors for additional mood benefits.",
    "Your sleep schedule looks healthy - maintain this routine."
  ]
}
```

### Request Parameters

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `day_rating` | string | Text description of your day | Analyzed for sentiment |
| `water_intake` | float | Water consumption in liters | 0-15 liters |
| `people_met` | integer | Number of people interacted with | ≥ 0 |
| `exercise` | integer | Exercise duration in minutes | ≥ 0 |
| `sleep` | float | Sleep duration in hours | 0-24 hours |
| `screen_time` | float | Screen time in hours | 0-24 hours |
| `outdoor_time` | float | Time spent outdoors in hours | 0-24 hours |
| `stress_level` | string | Stress level | "Low", "Medium", "High" |
| `food_quality` | string | Nutritional quality | "Healthy", "Moderate", "Unhealthy" |

### Additional Endpoints

- `GET /`: Root endpoint with basic API information
- `GET /health`: Health check endpoint
- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /redoc`: Alternative API documentation

## Project Structure

```
Mood-Tracker/
├── app.py                      # FastAPI application with API endpoints
├── model.py                    # Mood prediction model logic and training
├── recommendations.py          # Personalized recommendation engine
├── data_generator.py           # Synthetic training data generation
├── train_model.py              # Model training and evaluation script
├── debug_tool.py               # Debugging utilities for model analysis
├── requirements.txt            # Python package dependencies
├── setup_env.sh               # Environment setup script (Linux/macOS)
├── install_mac_deps.sh        # macOS-specific dependency installer
├── data_structure.md           # Documentation of data schema
├── README.md                   # Project documentation
├── mood_model.pkl             # Trained model file (generated)
├── model_metrics.json         # Model performance metrics (generated)
├── training_data.csv          # Full synthetic dataset (generated)
├── training_data_train.csv    # Training subset (generated)
├── training_data_test.csv     # Testing subset (generated)
└── plots/                     # Model performance visualizations
    ├── mood_prediction_performance.png
    ├── feature_importance.png
    └── error_distribution.png
```

### Core Components

- **`app.py`**: FastAPI web server with CORS support for web integration
- **`model.py`**: Machine learning pipeline with ensemble models and feature engineering
- **`recommendations.py`**: Rule-based recommendation system based on lifestyle factors
- **`data_generator.py`**: Creates realistic synthetic data with proper correlations
- **`train_model.py`**: Command-line interface for model training and evaluation

## Development & Customization

### Customizing the Model
- Modify training parameters in `train_model.py`
- Adjust feature engineering in `model.py`  
- Add new lifestyle factors by updating the data schema
- Experiment with different ML algorithms in the ensemble

### Adding New Features
- Update `MoodInput` class in `app.py` for new input parameters
- Modify `data_generator.py` to include new synthetic data patterns
- Update the recommendation logic in `recommendations.py`

### Performance Monitoring
- Model metrics are automatically saved to `model_metrics.json`
- Performance visualizations are generated in `plots/`
- Use `debug_tool.py` for detailed model analysis

## Deployment

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
- `MODEL_PATH`: Custom path for the trained model file
- `PORT`: API server port (default: 8000)
- `LOG_LEVEL`: Logging verbosity (default: info)

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# For specific package issues
pip install --upgrade scikit-learn pandas numpy
```

**Model Training Issues**
```bash
# Clear existing model and retrain
rm mood_model.pkl model_metrics.json
python train_model.py --force
```

**macOS OpenMP Issues**
```bash
# Install OpenMP for macOS (if using XGBoost)
brew install libomp
```

**Windows PowerShell Execution Policy**
```powershell
# If scripts are blocked
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Debugging Tips
- Use `debug_tool.py` to analyze model predictions
- Check `model_metrics.json` for performance indicators
- Examine plots in `plots/` directory for model insights
- Enable FastAPI debug mode with `--reload` flag

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).
