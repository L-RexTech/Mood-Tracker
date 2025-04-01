import re
import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Literal, Dict, Any, Union
from pathlib import Path

# Set NLTK data path for Railway deployment
nltk_data_path = '/app/nltk_data'
if os.path.exists(nltk_data_path):
    os.environ['NLTK_DATA'] = nltk_data_path

# Try to import machine learning libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import HuberRegressor
    HAS_SKLEARN = True
    
    # Disable XGBoost due to OpenMP issues on macOS
    HAS_XGBOOST = False
    print("Using GradientBoosting instead of XGBoost for better compatibility")
    
except ImportError:
    HAS_SKLEARN = False
    HAS_XGBOOST = False

# NLTK for sentiment analysis from day_rating
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True, download_dir=os.environ.get('NLTK_DATA', None))
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# Constants
MODEL_PATH = Path(__file__).parent / "mood_model.pkl"

def analyze_day_rating(day_rating: str) -> float:
    """
    Analyze the day rating text and return a sentiment score between 0-1
    Enhanced to better detect negative emotions and negation patterns
    """
    day_rating_lower = day_rating.lower()
    
    # High priority check for explicit bad day statements
    if any(phrase in day_rating_lower for phrase in ["bad day", "terrible day", "awful day", 
                                                    "horrible day", "not a good day"]):
        # Direct match for these specific phrases should override everything
        print(f"Direct negative day description detected: '{day_rating}'")
        return 0.15  # Very low score (1.5 on 0-10 scale)
    
    # Check for specific negation patterns
    negation_patterns = [
        "not good", "not great", "not a good", "not the best", 
        "not happy", "not fun", "not nice", "not well",
        "wasn't good", "wasnt good", "wasn't great", "wasnt great",
        "wasn't nice", "wasnt nice", "not okay", "not ok"
    ]
    
    # Check for repetitive neutral phrases that indicate mediocre experience
    repetitive_neutral = ["okay okay", "ok ok", "fine fine", "alright alright"]
    
    # Check for direct phrases
    direct_negative_phrases = [
        "not a good day", "bad day", "rough day", "tough day", "difficult day",
        "wasn't a good day", "wasnt a good day", "wasn't great", "wasnt great",
        "mediocre day", "below average day", "subpar day", "disappointing day"
    ]
    
    # Check for these specific patterns first
    for pattern in negation_patterns:
        if pattern in day_rating_lower:
            # Negation found - this should be interpreted as negative
            return 0.25  # Force a negative sentiment (0-10 scale: 2.5)
    
    for phrase in repetitive_neutral:
        if phrase in day_rating_lower:
            # Repetitive neutral words often indicate mediocre experience
            return 0.45  # Mediocre sentiment (0-10 scale: 4.5)
    
    for phrase in direct_negative_phrases:
        if phrase in day_rating_lower:
            # Direct negative phrase
            return 0.2  # Strongly negative (0-10 scale: 2.0)
    
    # Add special handling for severe negative keywords
    severe_negative_keywords = [
            'frustrat', 'anger', 'hate', 'terrible', 'horrible', 'miserable', 'awful',
            'depress', 'anxious', 'hopeless', 'worthless', 'exhausted', 'drained',
            'stressed', 'irritated', 'overwhelmed', 'failure', 'regret', 'resent',
            'lonely', 'isolated', 'disgusted', 'panic', 'devastated', 'helpless',
            'furious', 'sad', 'cry', 'nightmare', 'ruined', 'torture', 'unbearable',
            'dreadful', 'disappointed', 'broken', 'meltdown', 'suffering', 'trauma'
        ]

    for word in severe_negative_keywords:
        if word in day_rating_lower:
            # Force low sentiment score when these words are present
            if HAS_NLTK:
                # Still run NLTK but cap the maximum possible score lower
                sia = SentimentIntensityAnalyzer()
                sentiment = sia.polarity_scores(day_rating)
                return max(0, min((sentiment['compound'] + 1) / 5, 0.3))  # Max 0.3 (0-10 scale: max 3.0)
            else:
                # Direct low score for keyword match in fallback mode
                return 0.25  # Force low sentiment (0-10 scale: 2.5)
    
    # Add more robust pattern matching for "bad day" variations
    if "bad" in day_rating_lower and any(word in day_rating_lower for word in ["day", "time", "experience"]):
        return 0.2  # Strongly negative
    
    # Advanced negation detection - look for "not" followed by positive words
    if any(neg in day_rating_lower for neg in ["not ", "n't ", "no ", "never "]):
        # If negation word is present, check for nearby positive words
        if any(pos in day_rating_lower for pos in ["good", "great", "happy", "nice", "fun", "well"]):
            return 0.3  # Likely negating a positive sentiment
    
    # If NLTK is available, use it with negation awareness
    if HAS_NLTK:
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(day_rating)
        
        # Additional checks for potentially misclassified sentiment
        if "not" in day_rating_lower and sentiment['compound'] > 0:
            # NLTK sometimes misses negations, adjust score down
            return max(0.2, (sentiment['compound'] + 1) / 4)
            
        return (sentiment['compound'] + 1) / 2
    else:
        # Fallback simple sentiment analysis with enhanced negative detection
        positive_words = [
    'good', 'great', 'excellent', 'happy', 'positive', 'relaxed', 'productive', 
    'wonderful', 'amazing', 'fun', 'enjoyable', 'peaceful', 'joyful', 'delightful', 
    'fantastic', 'pleasant', 'content', 'successful', 'satisfied', 'energetic', 
    'optimistic', 'vibrant', 'calm', 'gratifying', 'cheerful', 'motivated', 
    'inspired', 'balanced', 'refreshed', 'tranquil', 'thrilled', 'pleased',
    'accomplished', 'grateful', 'loved', 'hopeful', 'excited', 'fulfilled', 
    'comfortable', 'lively', 'serene', 'upbeat', 'enthusiastic', 'jubilant', 
    'ecstatic', 'radiant', 'elated', 'blissful', 'euphoric', 'harmonious', 
    'confident', 'flourishing', 'revitalized', 'cheery', 'glowing', 'buoyant', 
    'sunny', 'exhilarated', 'optimistic', 'heartwarming', 'joyous', 'rejuvenated',
    'brilliant', 'hope-instilling', 'bubbly', 'peace-filled', 'lighthearted'
]

        negative_words = [
    'bad', 'terrible', 'unhappy', 'sad', 'negative', 'stressful', 'hectic', 
    'tiring', 'exhausting', 'depressing', 'difficult', 'horrible', 'anxious', 
    'worried', 'overwhelmed', 'frustrated', 'annoyed', 'angry', 'upset', 
    'disappointed', 'gloomy', 'miserable', 'irritated', 'tense', 'nervous', 
    'stressed', 'chaotic', 'unpleasant', 'draining', 'tough', 'painful', 
    'distressed', 'lonely', 'fearful', 'discouraged', 'helpless', 'agitated',
    'restless', 'uncomfortable', 'dissatisfied', 'troubled', 'bitter', 'bored',
    'dreadful', 'fatigued', 'hopeless', 'inadequate', 'frustration', 'anxiety',
    'panic', 'rage', 'hostility', 'resentment', 'grief', 'despair', 'guilt',
    'isolated', 'melancholy', 'pessimistic', 'unworthy', 'worthless', 'suffocating', 
    'shattered', 'exasperated', 'irate', 'devastated', 'shaken', 'forlorn', 
    'withdrawn', 'resentful', 'desolate', 'despondent', 'apathetic', 'alienated', 
    'oppressed', 'woeful', 'downhearted', 'tormented', 'discouraging', 'lifeless'
]

        critical_negative_words = [
            'frustrat', 'anger', 'hate', 'terrible', 'horrible', 'miserable', 'awful',
            'depress', 'anxious', 'hopeless', 'worthless', 'exhausted', 'drained',
            'stressed', 'irritated', 'overwhelmed', 'failure', 'regret', 'resent',
            'lonely', 'isolated', 'disgusted', 'panic', 'devastated', 'helpless',
            'furious', 'sad', 'cry', 'nightmare', 'ruined', 'torture', 'unbearable',
            'dreadful', 'disappointed', 'broken', 'meltdown', 'suffering', 'trauma'
        ]
        
        pos_count = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', day_rating_lower))
        neg_count = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', day_rating_lower))
        
        for word in critical_negative_words:
            if word in day_rating_lower:
                neg_count *= 2
                break
                
        if pos_count == 0 and neg_count == 0:
            return 0.5
        
        sentiment_score = pos_count / (pos_count + neg_count)
        
        if neg_count > 0 and 'frustrat' in day_rating_lower:
            sentiment_score = min(sentiment_score, 0.3)
            
        return sentiment_score

    # For explicitly negative expressions or neutral with negative markers
    if "not" in day_rating_lower and "good" in day_rating_lower:
        return 0.3  # "not good" is definitely negative
        
    return sentiment_score

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
    sentiment_score = analyze_day_rating(day_rating)
    
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
    The rule-based model with improved balance between parameters
    """
    stress_numeric = {"Low": 2, "Medium": 1, "High": 0}
    food_numeric = {"Healthy": 2, "Moderate": 1, "Unhealthy": 0}
    
    # Define optimal values and ranges
    optimal_water = 3.0  # liters
    min_optimal_water = 2.5  # minimum acceptable for "optimal" range
    
    optimal_exercise = 60  # minutes
    min_optimal_exercise = 30  # minimum acceptable for "optimal" range
    
    optimal_sleep = 8.0  # hours
    sleep_tolerance = 1.0  # ±1 hour is still considered optimal
    
    optimal_screen_time = 2.0  # hours
    max_optimal_screen = 3.0  # maximum acceptable for "optimal" range
    
    optimal_outdoor_time = 2.0  # hours
    max_optimal_outdoor = 4.0  # maximum acceptable for "optimal" range
    
    # Calculate factor scores with enhanced penalties for very low values
    water_score = min(features['water_intake'] / optimal_water, 1.0)
    # Add severe penalty for zero or very low water intake
    if features['water_intake'] < 0.5:
        water_score = -0.5  # Negative score for severe dehydration
    
    exercise_score = min(features['exercise'] / optimal_exercise, 1.0)
    # Add penalty for zero exercise
    if features['exercise'] == 0:
        exercise_score = -0.3  # Negative score for no physical activity
    
    sleep_score = 1.0 - abs(features['sleep'] - optimal_sleep) / 8.0
    screen_time_score = max(0, 1.0 - (features['screen_time'] - optimal_screen_time) / 10.0)
    
    outdoor_time_score = min(features['outdoor_time'] / optimal_outdoor_time, 1.0)
    # Add penalty for zero outdoor time but also for excessive values
    if features['outdoor_time'] == 0:
        outdoor_time_score = -0.2  # Negative score for no outdoor time
    elif features['outdoor_time'] > 6:
        outdoor_time_score = 0.8  # Cap at 0.8 for very high values (likely errors or exaggeration)
    
    stress_score = stress_numeric[features['stress_level']] / 2.0
    
    # Normalize food quality string to handle case sensitivity
    food_quality_normalized = features['food_quality'].capitalize()
    if food_quality_normalized not in food_numeric:
        food_quality_normalized = "Moderate"  # Default if not recognized
    food_score = food_numeric[food_quality_normalized] / 2.0
    
    social_score = min(features['people_met'] / 5.0, 1.0)
    
    sentiment_score = features['sentiment']
    
    weights = {
        'day_rating': 0.20,  # Reduced from 0.25
        'water': 0.10,       # Increased from 0.05
        'exercise': 0.12,    # Increased from 0.10
        'sleep': 0.15,
        'screen_time': 0.10,
        'outdoor_time': 0.12, # Increased from 0.10
        'stress': 0.10,       # Reduced from 0.15
        'food': 0.06,         # Increased from 0.05
        'social': 0.05
    }
    
    # Calculate weighted average
    weighted_score = (
        weights['day_rating'] * sentiment_score +
        weights['water'] * water_score +
        weights['exercise'] * exercise_score +
        weights['sleep'] * sleep_score +
        weights['screen_time'] * screen_time_score +
        weights['outdoor_time'] * outdoor_time_score +
        weights['stress'] * stress_score +
        weights['food'] * food_score +
        weights['social'] * social_score
    )
    
    # Check for non-optimal values to cap perfect scores
    suboptimal_metrics = []
    
    if features['water_intake'] < min_optimal_water:
        suboptimal_metrics.append(f"water intake ({features['water_intake']:.1f}L vs {min_optimal_water}L min)")
        
    if features['exercise'] < min_optimal_exercise:
        suboptimal_metrics.append(f"exercise ({features['exercise']} min vs {min_optimal_exercise} min)")
        
    if abs(features['sleep'] - optimal_sleep) > sleep_tolerance:
        suboptimal_metrics.append(f"sleep ({features['sleep']:.1f} hrs vs {optimal_sleep-sleep_tolerance}-{optimal_sleep+sleep_tolerance} optimal)")
        
    if features['outdoor_time'] < 1.0:
        suboptimal_metrics.append(f"outdoor time ({features['outdoor_time']:.1f} hrs vs 1.0 min)")
    elif features['outdoor_time'] > max_optimal_outdoor:
        suboptimal_metrics.append(f"excessive outdoor time ({features['outdoor_time']:.1f} hrs)")
    
    # Convert to 0-10 scale
    raw_score = weighted_score * 10
    
    # Apply strict caps for high scores based on non-optimal metrics
    if len(suboptimal_metrics) > 0:
        if raw_score > 9.5:  # If we're getting close to a perfect score
            # Cap based on number of suboptimal metrics
            max_possible = 9.7 - (len(suboptimal_metrics) * 0.4)
            capped_score = min(raw_score, max_possible)
            if raw_score > capped_score:
                print(f"Score capped from {raw_score:.1f} to {capped_score:.1f} due to: {', '.join(suboptimal_metrics)}")
            raw_score = capped_score
    
    return max(min(raw_score, 9.9), 0.0)  # Max 9.9 to prevent perfect 10.0 unless all is optimal

def create_interaction_features(df):
    """
    Create interaction features that could improve predictions
    """
    df_new = df.copy()
    
    df_new['sleep_screen_ratio'] = df_new['sleep'] / (df_new['screen_time'] + 0.5)
    df_new['exercise_outdoor'] = df_new['exercise'] * df_new['outdoor_time']
    df_new['hydration_exercise'] = df_new['water_intake'] / (df_new['exercise'] / 60 + 0.5)
    df_new['social_quality'] = df_new['people_met'] * (df_new['stress_level'] == 'Low').astype(int)
    df_new['sleep_quality'] = 1.0 - abs(df_new['sleep'] - 8.0) / 4.0
    df_new['work_life_balance'] = df_new['outdoor_time'] / (df_new['screen_time'] + 1.0)
    df_new['burnout_risk'] = ((df_new['stress_level'] == 'High').astype(int) * 
                              (df_new['screen_time'] > 6).astype(int) * 
                              (df_new['sleep'] < 6).astype(int))
    df_new['wellness_score'] = ((df_new['water_intake'] / 3.0) + 
                               (df_new['exercise'] / 60.0) + 
                               (df_new['food_quality'] == 'Healthy').astype(int)) / 3.0
    
    return df_new

def create_and_train_model(train_data_path: str = 'training_data_train.csv', use_ensemble=True) -> Pipeline:
    """
    Create and train an enhanced regression model for mood prediction with
    optimizations for ±0.5 point accuracy
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required to train the regression model.")
    
    df = pd.read_csv(train_data_path)
    X_orig = df.drop(['date', 'day_rating', 'mood_score'], axis=1)
    y = df['mood_score']
    
    categorical_features = ['stress_level', 'food_quality']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    numerical_features = ['water_intake', 'people_met', 'exercise', 'sleep', 
                         'screen_time', 'outdoor_time']
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
    
    if use_ensemble and HAS_XGBOOST:
        print("Using model ensemble for improved precision")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=1,
            objective='reg:squarederror',
            random_state=42
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.9,
            random_state=42
        )
        
        huber_model = HuberRegressor(
            epsilon=1.2,
            max_iter=200,
            alpha=0.001
        )
        
        regressor = VotingRegressor([
            ('xgb', xgb_model),
            ('gb', gb_model),
            ('huber', huber_model)
        ])
    
    elif HAS_XGBOOST:
        print("Using optimized XGBoost for regression")
        regressor = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=1,
            objective='reg:squarederror',
            random_state=42
        )
    else:
        print("Using optimized GradientBoostingRegressor (XGBoost unavailable)")
        regressor = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.9,
            max_features=0.8,
            random_state=42
        )
    
    X_orig = create_interaction_features(X_orig)
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    print(f"Training model with {len(X_orig)} samples and {X_orig.shape[1]} features")
    model.fit(X_orig, y)
    
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
    
    features_df = create_interaction_features(features_df)
    
    prediction = model.predict(features_df)[0]
    
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
    # Check for explicit negative day statements first - highest priority
    day_rating_lower = day_rating.lower()
    
    # Direct bad day detection - with logging for debugging
    explicit_bad_day = any(phrase in day_rating_lower for phrase in 
                          ["bad day", "terrible day", "awful day", "horrible day", 
                           "not good day", "wasn't good", "wasnt good"])
    
    if explicit_bad_day:
        print(f"Explicit negative day detected: '{day_rating}'")
        # Hard cap on score regardless of other metrics
        max_possible_score = 4.0
    else:
        # Standard processing for ambiguous phrases
        if any(phrase in day_rating_lower for phrase in 
              ["not a good day", "wasn't good", "wasnt good", "okay okay", "ok ok"]):
            max_possible_score = 5.5  # Cap for these ambiguous negative expressions
        else:
            max_possible_score = 10.0
    
    if ("frustrat" in day_rating_lower or "anger" in day_rating_lower or 
        "hate" in day_rating_lower or "terrible" in day_rating_lower) and stress_level == "Low":
        print("Warning: Contradiction detected - negative emotions in day description but stress level is Low")
    
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
    
    model = None
    if HAS_SKLEARN:
        model = load_model()
    
    if model is not None:
        try:
            prediction = predict_with_model(model, features)
        except Exception as e:
            print(f"Error using regression model: {e}. Falling back to rule-based approach.")
            prediction = rule_based_prediction(features)
    else:
        prediction = rule_based_prediction(features)
    
    # Handle health factor minimums - apply caps for multiple zero values
    zero_health_metrics = sum(1 for value in [water_intake, exercise, outdoor_time] if value == 0)
    
    health_caps = {
        3: 6.0,  # Max 6.0 if all three health metrics are zero
        2: 7.5,  # Max 7.5 if two health metrics are zero
        1: 8.5   # Max 8.5 if one health metric is zero
    }
    
    if zero_health_metrics > 0:
        max_possible_score = health_caps.get(zero_health_metrics, 10.0)
        print(f"Health factor limitation: {zero_health_metrics} zero metrics, max score capped at {max_possible_score}")
    else:
        max_possible_score = 10.0
    
    # Add strict caps for perfect scores
    # Perfect scores (10.0) should only be possible when all parameters are optimal
    perfect_score_possible = (
        analyze_day_rating(day_rating) > 0.95 and
        water_intake >= 2.5 and
        exercise >= 30 and
        (7.0 <= sleep <= 9.0) and
        screen_time <= 3.0 and
        (1.0 <= outdoor_time <= 4.0) and
        stress_level == "Low" and
        food_quality.capitalize() == "Healthy"
    )
    
    # Apply perfect score restriction
    if prediction >= 9.9 and not perfect_score_possible:
        prediction = 9.7  # Cap at 9.7 if not all metrics are ideal
        print("Score capped at 9.7 since not all metrics are in optimal ranges")
    
    return min(prediction, max_possible_score)

def train_model_if_needed():
    """Check if a model exists, if not, generate data and train one"""
    if not os.path.exists(MODEL_PATH) and HAS_SKLEARN:
        try:
            from data_generator import generate_synthetic_data, train_test_split_data
            
            if not os.path.exists('training_data.csv'):
                generate_synthetic_data()
            
            if not os.path.exists('training_data_train.csv'):
                train_test_split_data()
            
            model = create_and_train_model()
            save_model(model)
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    return True
