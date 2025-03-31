import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from pathlib import Path

from data_generator import generate_synthetic_data, train_test_split_data
from model import create_and_train_model, save_model, load_model, MODEL_PATH

# Constants
METRICS_PATH = Path(__file__).parent / "model_metrics.json"

def evaluate_model(model, test_data_path='training_data_test.csv'):
    """Evaluate the trained model on test data"""
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Extract features and target
    X_test = test_df.drop(['date', 'day_rating', 'mood_score'], axis=1)
    y_test = test_df['mood_score']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentage accuracy (within 1 point on 0-10 scale)
    within_one_point = np.mean(np.abs(y_pred - y_test) <= 1.0) * 100
    within_half_point = np.mean(np.abs(y_pred - y_test) <= 0.5) * 100
    
    # Print performance metrics with more visibility
    print("\n" + "="*60)
    print(" "*20 + "MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"R-squared (R²): {r2:.3f}")
    print(f"Predictions within 0.5 points of actual: {within_half_point:.1f}%")
    print(f"Predictions within 1.0 points of actual: {within_one_point:.1f}%")
    print("="*60 + "\n")
    
    # Save metrics to file
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r_squared": float(r2),
        "within_half_point_percentage": float(within_half_point),
        "within_one_point_percentage": float(within_one_point),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {METRICS_PATH}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 10], [0, 10], 'r--')
    plt.xlabel('Actual Mood Score')
    plt.ylabel('Predicted Mood Score')
    plt.title(f'Actual vs. Predicted Mood Scores\nR² = {r2:.3f}, MAE = {mae:.3f}')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True)
    
    # Save the plot
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "mood_prediction_performance.png")
    print(f"Performance plot saved to {plots_dir / 'mood_prediction_performance.png'}")
    
    # Save error distribution histogram
    plt.figure(figsize=(10, 6))
    errors = y_pred - y_test
    plt.hist(errors, bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Prediction Error Distribution')
    plt.grid(True)
    plt.savefig(plots_dir / "error_distribution.png")
    print(f"Error distribution saved to {plots_dir / 'error_distribution.png'}")
    
    # Try to display the plots if running in a graphical environment
    try:
        plt.show()
    except:
        pass
    
    return metrics

def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance with proper feature name generation"""
    try:
        # Get feature importances (works for tree-based models like Random Forest or GradientBoosting)
        importances = model.named_steps['regressor'].feature_importances_
        
        # Generate accurate feature names by extracting from the pipeline if possible
        try:
            # Try to extract feature names from the preprocessor
            preprocessor = model.named_steps['preprocessor']
            transformed_feature_names = []
            
            # Get the transformed feature names for categorical features
            for name, transformer, column in preprocessor.transformers_:
                if name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
                    transformed_feature_names.extend(transformer.get_feature_names_out())
                elif name == 'cat':  # Fallback for older scikit-learn versions
                    for col in column:
                        for category in ['Low', 'Medium', 'High', 'Healthy', 'Moderate', 'Unhealthy']:
                            transformed_feature_names.append(f"{col}_{category}")
            
            # For numerical features with polynomial expansion, generate appropriate names
            numerical_features = ['water_intake', 'people_met', 'exercise', 'sleep', 
                                'screen_time', 'outdoor_time']
            
            # Add base features
            for feature in numerical_features:
                transformed_feature_names.append(feature)
            
            # Add quadratic terms
            for feature in numerical_features:
                transformed_feature_names.append(f"{feature}^2")
            
            # Add interaction terms
            for i, feat1 in enumerate(numerical_features):
                for feat2 in numerical_features[i+1:]:
                    transformed_feature_names.append(f"{feat1} × {feat2}")
                    
            # Add interaction feature names
            interaction_feature_names = [
                'sleep_screen_ratio', 'exercise_outdoor', 'hydration_exercise',
                'social_quality', 'sleep_quality', 'work_life_balance', 
                'burnout_risk', 'wellness_score'
            ]
            transformed_feature_names.extend(interaction_feature_names)
            
            print(f"Generated {len(transformed_feature_names)} feature names for {len(importances)} features")
            
            # If count still doesn't match, use generic feature names
            if len(transformed_feature_names) != len(importances):
                transformed_feature_names = [f"Feature {i}" for i in range(len(importances))]
                
            feature_names = transformed_feature_names
            
        except Exception as e:
            print(f"Could not extract feature names: {e}")
            # Fall back to generic feature names
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Get the top 20 features (or all if less than 20)
        num_to_show = min(20, len(importances))
        indices = np.argsort(importances)[::-1][:num_to_show]
        
        # Create a plot showing only the top features
        plt.figure(figsize=(14, 10))
        plt.title(f'Top {num_to_show} Feature Importances')
        plt.bar(range(num_to_show), importances[indices])
        plt.xticks(range(num_to_show), [str(feature_names[i])[:30] for i in indices], rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "feature_importance.png")
        print(f"Saved feature importance plot to {plots_dir / 'feature_importance.png'}")
        
        # Try to display the plot if running in a graphical environment
        try:
            plt.show()
        except:
            pass
        
    except Exception as e:
        print(f"Could not analyze feature importance: {e}")
        print("This is not critical and won't affect model performance.")

def tune_hyperparameters(X_train, y_train):
    """Tune model hyperparameters using RandomizedSearchCV"""
    from sklearn.ensemble import GradientBoostingRegressor
    
    print("\nPerforming hyperparameter tuning...")
    
    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    }
    
    # Create the model
    gb = GradientBoostingRegressor(random_state=42)
    
    # Set up cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create and run the random search
    random_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_grid,
        n_iter=50,
        scoring='neg_mean_squared_error',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best RMSE: {(-random_search.best_score_)**0.5:.3f}")
    
    return random_search.best_params_

def train_and_evaluate(num_samples=5000, force_retrain=False, tune_params=False):
    """Generate data, train model, and evaluate performance"""
    # Step 1: Generate synthetic data if it doesn't exist
    if not os.path.exists('training_data.csv') or force_retrain:
        print("Generating synthetic data...")
        generate_synthetic_data(num_samples=num_samples)
    
    # Step 2: Split data into training and test sets if they don't exist
    if not os.path.exists('training_data_train.csv') or not os.path.exists('training_data_test.csv') or force_retrain:
        print("Splitting data into train and test sets...")
        train_test_split_data()
    
    # Step 3: Train the model
    print("Training the advanced regression model...")
    model = create_and_train_model()
    
    # Step 4: Save the model
    save_model(model)
    
    # Step 5: Evaluate the model
    metrics = evaluate_model(model)
    
    # Step 6: Analyze feature importance
    analyze_feature_importance(model, [])
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Train a mood prediction model")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--force", action="store_true", help="Force regenerate data and retrain")
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate an existing model")
    parser.add_argument("--accuracy", action="store_true", help="Show accuracy metrics for existing model")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    args = parser.parse_args()
    
    if args.accuracy:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            print("\n" + "="*60)
            print(" "*20 + "SAVED MODEL METRICS")
            print("="*60)
            print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.3f}")
            print(f"Mean Absolute Error (MAE): {metrics['mae']:.3f}")
            print(f"R-squared (R²): {metrics['r_squared']:.3f}")
            print(f"Predictions within 0.5 points of actual: {metrics['within_half_point_percentage']:.1f}%")
            print(f"Predictions within 1.0 points of actual: {metrics['within_one_point_percentage']:.1f}%")
            print(f"Last evaluated: {metrics['timestamp']}")
            print("="*60 + "\n")
        else:
            print("No saved metrics found. Run train_model.py first to generate metrics.")
        return
    
    if args.evaluate and os.path.exists(MODEL_PATH):
        print("Loading existing model for evaluation...")
        model = load_model()
        if model:
            evaluate_model(model)
    else:
        train_and_evaluate(num_samples=args.samples, force_retrain=args.force, tune_params=args.tune)
    
if __name__ == "__main__":
    main()
