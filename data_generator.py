import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_samples=1000, output_file='training_data.csv'):
    """
    Generate synthetic training data for the mood prediction model.
    This creates realistic relationships between daily habits and mood scores.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Base data structure
    data = {
        'date': [],
        'day_rating': [],
        'water_intake': [],
        'people_met': [],
        'exercise': [],
        'sleep': [],
        'screen_time': [],
        'outdoor_time': [],
        'stress_level': [],
        'food_quality': [],
        'mood_score': []  # Target variable
    }
    
    # Sample day ratings with varying sentiment
    positive_phrases = [
        "Had a great day", "Feeling accomplished", "Productive and focused",
        "Wonderful time with family", "Relaxing and peaceful day",
        "Energetic and motivated", "Happy and content", "Successful day at work",
        "Enjoyed every moment", "Feeling blessed and grateful",
        "Feeling refreshed and inspired", "Had a fun and exciting day",
        "Everything went smoothly", "Great conversations and laughter",
        "Full of positive vibes", "Achieved all my goals today",
        "Loved spending time outdoors", "Feeling optimistic about the future",
        "Peaceful and stress-free day", "Made great progress on my tasks"
    ]

    neutral_phrases = [
        "Average day", "Nothing special", "The usual routine",
        "Just another day", "Mixed feelings today", "Some ups and downs",
        "Ordinary day", "Neither good nor bad", "Same as usual",
        "Got through the day", "Just another regular day",
        "Nothing exciting happened", "Routine was the same as always",
        "Not much to say about today", "Neither productive nor lazy",
        "A bit dull but okay", "Went through the motions",
        "Mediocre day, nothing stood out", "Some things went well, some didn’t",
        "A bit slow, but manageable"
    ]

    negative_phrases = [
        "Stressful day", "Feeling overwhelmed", "Too much work",
        "Exhausted and drained", "Anxious all day", "Bad day overall",
        "Nothing went right", "Feeling down", "Frustrated with everything",
        "Didn't sleep well and felt tired", "Felt unproductive and stuck",
        "Struggled with motivation today", "Everything felt off",
        "Dealt with a lot of stress", "Emotionally exhausting day",
        "Felt lonely and disconnected", "Too many problems to deal with",
        "Had no energy to do anything", "Disappointed with how the day went",
        "Dealt with a lot of frustration"
    ]

    # Generate samples
    start_date = datetime.now() - timedelta(days=num_samples)
    
    for i in range(num_samples):
        # Date (sequential)
        current_date = start_date + timedelta(days=i)
        data['date'].append(current_date.strftime('%Y-%m-%d'))
        
        # Water intake (0.5L to 5L)
        water = round(np.random.uniform(0.5, 5.0), 1)
        data['water_intake'].append(water)
        
        # People met (0 to 20)
        people = int(np.random.poisson(5))  # Poisson with mean 5
        data['people_met'].append(min(people, 20))
        
        # Exercise (0 to 120 minutes)
        exercise = int(np.random.exponential(30))  # Exponential with mean 30
        data['exercise'].append(min(exercise, 120))
        
        # Sleep (4 to 11 hours)
        sleep = round(np.random.normal(7, 1.2), 1)  # Normal with mean 7, std 1.2
        data['sleep'].append(min(max(sleep, 4.0), 11.0))
        
        # Screen time (1 to 12 hours)
        screen_time = round(np.random.normal(5, 2), 1)
        data['screen_time'].append(min(max(screen_time, 1.0), 12.0))
        
        # Outdoor time (0 to 6 hours)
        outdoor_time = round(np.random.exponential(1.5), 1)
        data['outdoor_time'].append(min(outdoor_time, 6.0))
        
        # Stress level (categorical: Low, Medium, High)
        stress_probs = [0.3, 0.4, 0.3]  # Probabilities for Low, Medium, High
        stress = np.random.choice(["Low", "Medium", "High"], p=stress_probs)
        data['stress_level'].append(stress)
        
        # Food quality (categorical: Healthy, Moderate, Unhealthy)
        food_probs = [0.33, 0.47, 0.2]  # Probabilities for Healthy, Moderate, Unhealthy
        food = np.random.choice(["Healthy", "Moderate", "Unhealthy"], p=food_probs)
        data['food_quality'].append(food)
        
        # Now calculate a base mood score using realistic weights
        # This creates correlation between habits and mood score
        base_mood = 0
        
        # Water contribution (0 to 0.5)
        base_mood += 0.5 * min(water / 3.0, 1.0)
        
        # People met contribution (0 to 0.5)
        base_mood += 0.5 * min(people / 5.0, 1.0)
        
        # Exercise contribution (0 to 1.0)
        base_mood += 1.0 * min(exercise / 60.0, 1.0)
        
        # Sleep contribution (0 to 1.5, with penalty for too much or too little)
        sleep_score = 1.0 - abs(sleep - 8.0) / 4.0
        base_mood += 1.5 * max(sleep_score, 0.0)
        
        # Screen time contribution (penalty, 0 to -1.0)
        screen_penalty = max(0, (screen_time - 3.0) / 9.0)
        base_mood -= 1.0 * screen_penalty
        
        # Outdoor time contribution (0 to 1.0)
        base_mood += 1.0 * min(outdoor_time / 2.0, 1.0)
        
        # Stress level contribution
        stress_values = {"Low": 1.0, "Medium": 0.0, "High": -1.0}
        base_mood += 1.5 * stress_values[stress]
        
        # Food quality contribution
        food_values = {"Healthy": 0.75, "Moderate": 0.0, "Unhealthy": -0.75}
        base_mood += food_values[food]
        
        # Scale to 0-10 range and add noise
        scaled_mood = 5.0 + base_mood
        noise = np.random.normal(0, 0.7)  # Add some random variation
        final_mood = max(min(scaled_mood + noise, 10.0), 0.0)  # Clamp between 0 and 10
        data['mood_score'].append(round(final_mood, 1))
        
        # Generate day rating text based on mood score
        if final_mood >= 7.0:
            data['day_rating'].append(random.choice(positive_phrases))
        elif final_mood >= 4.0:
            data['day_rating'].append(random.choice(neutral_phrases))
        else:
            data['day_rating'].append(random.choice(negative_phrases))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {num_samples} samples and saved to {output_file}")
    
    return df

def train_test_split_data(data_file='training_data.csv', test_size=0.2):
    """
    Split the data into training and test sets
    """
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Generating new data...")
        df = generate_synthetic_data(output_file=data_file)
    else:
        df = pd.read_csv(data_file)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train and test sets
    test_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:test_idx]
    test_df = df.iloc[test_idx:]
    
    # Save splits
    train_file = data_file.replace('.csv', '_train.csv')
    test_file = data_file.replace('.csv', '_test.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Saved {len(train_df)} training samples to {train_file}")
    print(f"Saved {len(test_df)} test samples to {test_file}")
    
    return train_df, test_df

if __name__ == "__main__":
    # Generate 2000 samples by default
    df = generate_synthetic_data(num_samples=2000)
    
    # Split into train and test sets
    train_test_split_data()
    
    # Show data stats
    print("\nData statistics:")
    print(df.describe())
    
    # Show correlations with mood score
    print("\nFeature correlations with mood score:")
    correlations = df.drop(['date', 'day_rating'], axis=1).corr()['mood_score'].sort_values(ascending=False)
    print(correlations)
