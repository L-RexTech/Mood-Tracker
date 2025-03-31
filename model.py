import re
import os
import pickle
import pandas as pd
import numpy as np
from typing import Literal, Dict, Any, Union
from pathlib import Path

# Try to import machine learning libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# NLTK for sentiment analysis from day_rating
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# Constants
MODEL_PATH = Path(__file__).parent / "mood_model.pkl"

def analyze_day_rating(day_rating: str) -> float:
    """
    Analyze the day rating text and return a sentiment score between 0-1
    """
    if HAS_NLTK:
        # Use NLTK for sentiment analysis
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(day_rating)
        # Convert compound score from [-1, 1] to [0, 1]
        return (sentiment['compound'] + 1) / 2
    else:
        # Fallback simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'happy', 'positive', 'relaxed', 'productive', 
                          'wonderful', 'amazing', 'fun', 'enjoyable', 'peaceful', 'joyful', 'delightful', 
                          'fantastic', 'pleasant', 'content', 'successful', 'satisfied', 'energetic', 
                          'optimistic', 'vibrant', 'calm', 'gratifying', 'cheerful', 'motivated', 
                          'inspired', 'balanced', 'refreshed', 'tranquil', 'thrilled', 'pleased',
                          'accomplished', 'grateful', 'loved', 'content', 'hopeful', 'excited',
                          'fulfilled', 'comfortable', 'lively', 'serene', 'upbeat', 'enthusiastic']
        
        negative_words = ['bad', 'terrible', 'unhappy', 'sad', 'negative', 'stressful', 'hectic', 
                          'tiring', 'exhausting', 'depressing', 'difficult', 'horrible', 'anxious', 
                          'worried', 'overwhelmed', 'frustrated', 'annoyed', 'angry', 'upset', 
                          'disappointed', 'gloomy', 'miserable', 'irritated', 'tense', 'nervous', 
                          'stressed', 'chaotic', 'unpleasant', 'draining', 'tough', 'painful', 
                          'distressed', 'lonely', 'fearful', 'discouraged', 'helpless', 'agitated',
                          'restless', 'uncomfortable', 'dissatisfied', 'troubled', 'bitter', 'bored',
                          'dreadful', 'fatigued', 'hopeless', 'inadequate']
        
        day_rating_lower = day_rating.lower()
        pos_count = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', day_rating_lower))
        neg_count = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', day_rating_lower))
        
        if pos_count == 0 and neg_count == 0:
            return 0.5  # Neutral
        return pos_count / (pos_count + neg_count)

def prepare_features(
    day_rating: str,
    water_intake: float,
    people_met: int,
    exercise: int,
    sleep: float,
    screen_time: float,
    outdoor_time: float,
    stress_level: Literal["Low", "Medium", "High"],
    food_quality: Literal["Healthy", "Moderate", "Unhealthy"]
) -> Dict[str, Any]:
    """
    Prepare features for the regression model
    """
    # Extract sentiment from day_rating
    sentiment_score = analyze_day_rating(day_rating)
    
    # Return features as a dictionary
    return {
        'sentiment': sentiment_score,
        'water_intake': water_intake,
        'people_met': people_met,
        'exercise': exercise,
        'sleep': sleep,
        'screen_time': screen_time,
        'outdoor_time': outdoor_time,
        'stress_level': stress_level,
        'food_quality': food_quality
    }

def rule_based_prediction(features: Dict[str, Any]) -> float:
    """
    The original rule-based model as a fallback
    """
    # Convert stress level to numeric
    stress_numeric = {"Low": 2, "Medium": 1, "High": 0}
    
    # Convert food quality to numeric
    food_numeric = {"Healthy": 2, "Moderate": 1, "Unhealthy": 0}
    
    # Define optimal values for each factor
    optimal_water = 3.0  # liters
    optimal_exercise = 60  # minutes
    optimal_sleep = 8.0  # hours
    optimal_screen_time = 2.0  # hours (max recommended)
    optimal_outdoor_time = 2.0  # hours
    
    # Calculate factor scores (0-1 scale)
    water_score = min(features['water_intake'] / optimal_water, 1.0)
    exercise_score = min(features['exercise'] / optimal_exercise, 1.0)
    sleep_score = 1.0 - abs(features['sleep'] - optimal_sleep) / 8.0  # Penalize both under and over sleeping
    screen_time_score = max(0, 1.0 - (features['screen_time'] - optimal_screen_time) / 10.0)  # Higher screen time reduces score
    outdoor_time_score = min(features['outdoor_time'] / optimal_outdoor_time, 1.0)
    stress_score = stress_numeric[features['stress_level']] / 2.0
    food_score = food_numeric[features['food_quality']] / 2.0
    social_score = min(features['people_met'] / 5.0, 1.0)  # Assuming meeting 5+ people is optimal
    
    # Weight each factor (sum of weights = 1)
    weights = {
        'day_rating': 0.25,
        'water': 0.05,
        'exercise': 0.10,
        'sleep': 0.15,
        'screen_time': 0.10,
        'outdoor_time': 0.10,
        'stress': 0.15,
        'food': 0.05,
        'social': 0.05
    }
    
    # Calculate weighted average
    weighted_score = (
        weights['day_rating'] * features['sentiment'] +
        weights['water'] * water_score +
        weights['exercise'] * exercise_score +
        weights['sleep'] * sleep_score +
        weights['screen_time'] * screen_time_score +
        weights['outdoor_time'] * outdoor_time_score +
        weights['stress'] * stress_score +
        weights['food'] * food_score +
        weights['social'] * social_score
    )
    
    # Convert to 0-10 scale
    return weighted_score * 10

def create_and_train_model(train_data_path: str = 'training_data_train.csv') -> Pipeline:
    """
    Create and train a regression model for mood prediction
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required to train the regression model.")
    
    # Load training data
    df = pd.read_csv(train_data_path)
    
    # Extract features and target
    X = df.drop(['date', 'day_rating', 'mood_score'], axis=1)
    y = df['mood_score']
    
    # Create preprocessing for categorical features
    categorical_features = ['stress_level', 'food_quality']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Create preprocessing for numerical features
    numerical_features = ['water_intake', 'people_met', 'exercise', 'sleep', 
                         'screen_time', 'outdoor_time']
    numerical_transformer = StandardScaler()
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
    
    # Create and train the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Fit the model
    model.fit(X, y)
    
    return model

def save_model(model, path=MODEL_PATH):
    """Save the trained model to disk"""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path=MODEL_PATH):
    """Load the trained model from disk"""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model
    except (FileNotFoundError, pickle.UnpicklingError):
        print(f"No valid model found at {path}")
        return None

def predict_with_model(model, features_dict: Dict[str, Any]) -> float:
    """
    Make a prediction using the trained model
    """
    # Extract only the features the model was trained on
    features_df = pd.DataFrame({
        'water_intake': [features_dict['water_intake']],
        'people_met': [features_dict['people_met']],
        'exercise': [features_dict['exercise']],
        'sleep': [features_dict['sleep']],
        'screen_time': [features_dict['screen_time']],
        'outdoor_time': [features_dict['outdoor_time']],
        'stress_level': [features_dict['stress_level']],
        'food_quality': [features_dict['food_quality']]
    })
    
    # Make prediction
    prediction = model.predict(features_df)[0]
    
    # Ensure the prediction is in the valid range
    return max(min(prediction, 10.0), 0.0)

def predict_mood_score(
    day_rating: str,
    water_intake: float,
    people_met: int,
    exercise: int,
    sleep: float,
    screen_time: float,
    outdoor_time: float,
    stress_level: Literal["Low", "Medium", "High"],
    food_quality: Literal["Healthy", "Moderate", "Unhealthy"]
) -> float:
    """
    Predict mood score (0-10) based on input parameters.
    Uses a regression model if available, otherwise falls back to rule-based approach.
    """
    # Prepare features
    features = prepare_features(
        day_rating=day_rating,
        water_intake=water_intake,
        people_met=people_met,
        exercise=exercise,
        sleep=sleep,
        screen_time=screen_time,
        outdoor_time=outdoor_time,
        stress_level=stress_level,
        food_quality=food_quality
    )
    
    # Try to use the trained model if available
    model = None
    if HAS_SKLEARN:
        model = load_model()
    
    if model is not None:
        try:
            return predict_with_model(model, features)
        except Exception as e:
            print(f"Error using regression model: {e}. Falling back to rule-based approach.")
            return rule_based_prediction(features)
    else:
        # Fall back to rule-based approach
        return rule_based_prediction(features)

def train_model_if_needed():
    """Check if a model exists, if not, generate data and train one"""
    if not os.path.exists(MODEL_PATH) and HAS_SKLEARN:
        try:
            from data_generator import generate_synthetic_data, train_test_split_data
            
            # Generate data if needed
            if not os.path.exists('training_data.csv'):
                generate_synthetic_data()
            
            # Split data if needed
            if not os.path.exists('training_data_train.csv'):
                train_test_split_data()
            
            # Train and save the model
            model = create_and_train_model()
            save_model(model)
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    return True
