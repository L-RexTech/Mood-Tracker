import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_samples=10000, output_file='training_data.csv'):
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
    "Peaceful and stress-free day", "Made great progress on my tasks",
    "Surrounded by good energy", "Felt deeply appreciated",
    "Had meaningful interactions", "A day full of creativity",
    "Laughs and joy all around", "Made someone's day better",
    "Experienced something new and exciting", "Felt very confident today",
    "Balanced work and relaxation well", "Good food, good mood",
    "Woke up feeling fresh and energized", "Got recognized for my work",
    "Made time for self-care", "Feeling inspired to keep going",
    "Had a fulfilling and rewarding day", "Spent time with supportive people",
    "Everything felt in sync", "A day of personal growth",
    "Found happiness in small moments", "A peaceful and worry-free day"
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
    "A bit slow, but manageable", "Kept myself busy",
    "A normal day without surprises", "Stayed on track but nothing remarkable",
    "Everything felt neutral", "Day passed by without much happening",
    "Neither too good nor too bad", "Did what I needed to, nothing extra",
    "No major wins or losses", "Routine day, no complaints",
    "A few small highlights, but nothing major", "Not a bad day, just uneventful",
    "Some tasks done, some left for tomorrow", "Simply an okay day"
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
    "Dealt with a lot of frustration", "Felt ignored and unappreciated",
    "Struggled with negative thoughts", "Had a rough time at work",
    "Couldn’t focus on anything", "A day full of setbacks",
    "Felt mentally drained", "Felt like giving up today",
    "A lot of miscommunication happened", "Didn’t achieve what I wanted",
    "Everything felt like a challenge", "Too many things went wrong",
    "Feeling burned out", "Had no motivation to do anything",
    "Felt emotionally unstable", "A tough day from start to finish",
    "Nothing seemed to work out", "Struggled to stay positive",
    "Faced a lot of unexpected problems", "Feeling completely exhausted"
]

    # Enhanced phrase collections with special cases for training
    
    # Add negation phrases (like "not good", "wasn't great")
    negation_phrases = [
        "Not a good day at all", "Wasn't what I hoped for", "Not feeling great today",
        "It wasn't a productive day", "Not terrible but not good either",
        "Didn't enjoy most of today", "Not particularly happy with how things went",
        "Things were not ideal today", "Wasn't the best experience",
        "Not satisfied with what I accomplished", "Not as good as yesterday",
        "Didn't feel like myself today", "Not very motivated",
        "Day was not what I expected", "Not much to be excited about"
    ]
    
    # Add phrases with repetitive neutral words
    repetitive_neutral_phrases = [
        "It was okay okay", "Just fine fine", "Alright alright I guess",
        "Meh meh kind of day", "So-so at best", "Kind of okay okay",
        "Neither here nor there", "Just okay I suppose", 
        "Fine but nothing special", "Tolerable I guess",
        "Could be worse could be better", "Just getting by"
    ]
    
    # Add contradictory statements (good things but negative feeling)
    contradiction_phrases = [
        "Everything went right but still felt off", 
        "Good day objectively but I felt down",
        "Should have been happy but wasn't", 
        "Things went well but I'm disappointed",
        "Got a lot done but still unsatisfied",
        "No real problems but felt anxious all day",
        "Nothing bad happened but felt stressed",
        "Everyone said it went well but I disagree",
        "Successful day but emotionally draining",
        "Achieved my goals but didn't feel fulfilled"
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
        
        # Now calculate a base mood score using enhanced factors for more realistic data
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
        
        # Add some interaction effects to make data more realistic and nuanced
        # Sleep quality is worse when screen time is high
        if screen_time > 4 and sleep < 7:
            base_mood -= 0.5
            
        # Exercise and outdoor time together have synergistic positive effects
        if exercise > 30 and outdoor_time > 1:
            base_mood += 0.3
            
        # Low water + high exercise is particularly bad
        if water < 1.5 and exercise > 45:
            base_mood -= 0.4
            
        # Scale to 0-10 range and add controlled noise
        scaled_mood = 5.0 + base_mood
        
        # Use less noise for values in the middle of the scale to improve ±0.5 accuracy
        if 4.0 <= scaled_mood <= 6.0:
            noise = np.random.normal(0, 0.4)  # Less noise in middle range
        else:
            noise = np.random.normal(0, 0.7)  # Regular noise elsewhere
            
        final_mood = max(min(scaled_mood + noise, 10.0), 0.0)  # Clamp between 0 and 10
        data['mood_score'].append(round(final_mood, 1))
        
        # Generate day rating text based on mood score
        if final_mood >= 7.0:
            data['day_rating'].append(random.choice(positive_phrases))
        elif final_mood >= 4.0:
            data['day_rating'].append(random.choice(neutral_phrases))
        else:
            data['day_rating'].append(random.choice(negative_phrases))
    
    # Add special cases for training - negative descriptions with otherwise positive indicators
    # This helps the model learn to handle contradictions properly
    special_cases = 200  # Number of special contradiction cases
    
    for i in range(special_cases):
        # Date
        special_date = start_date + timedelta(days=num_samples + i)
        data['date'].append(special_date.strftime('%Y-%m-%d'))
        
        # Generate mostly positive metrics
        data['water_intake'].append(round(np.random.uniform(2.0, 4.0), 1))  # Good hydration
        data['people_met'].append(np.random.randint(3, 10))  # Decent social interaction
        data['exercise'].append(np.random.randint(20, 60))  # Some exercise
        data['sleep'].append(round(np.random.uniform(7.0, 9.0), 1))  # Good sleep
        data['screen_time'].append(round(np.random.uniform(1.0, 3.0), 1))  # Low screen time
        data['outdoor_time'].append(round(np.random.uniform(1.0, 3.0), 1))  # Good outdoor time
        
        # Random mix of stress and food quality
        data['stress_level'].append(np.random.choice(["Low", "Medium"]))
        data['food_quality'].append(np.random.choice(["Moderate", "Healthy"]))
        
        # But add negative day descriptions with frustration
        negative_frustration_phrases = [
            "Dealt with a lot of frustration today", 
            "Feeling frustrated despite everything",
            "Day was filled with frustrating moments",
            "So frustrated with how things went",
            "Everything was frustrating today"
        ]
        data['day_rating'].append(random.choice(negative_frustration_phrases))
        
        # And ensure lower mood scores despite good metrics
        data['mood_score'].append(round(np.random.uniform(2.0, 5.0), 1))  # Lower mood scores
    
    # Add special training cases for ambiguous phrases
    ambiguous_cases = 150  # Number of ambiguous phrases to add
    
    # Negation cases - negative phrases with potentially positive metrics
    for i in range(ambiguous_cases):
        special_date = start_date + timedelta(days=num_samples + special_cases + i)
        data['date'].append(special_date.strftime('%Y-%m-%d'))
        
        # Mixed metrics - some good, some bad
        data['water_intake'].append(round(np.random.uniform(1.5, 3.5), 1))
        data['people_met'].append(np.random.randint(2, 8))
        data['exercise'].append(np.random.randint(15, 45))
        data['sleep'].append(round(np.random.uniform(6.5, 8.0), 1))
        data['screen_time'].append(round(np.random.uniform(2.0, 5.0), 1))
        data['outdoor_time'].append(round(np.random.uniform(0.5, 2.0), 1))
        
        # Random stress level and food quality
        data['stress_level'].append(np.random.choice(["Low", "Medium", "High"]))
        data['food_quality'].append(np.random.choice(["Healthy", "Moderate", "Unhealthy"]))
        
        # Add negation or repetitive neutral phrases
        phrase_type = i % 3
        if phrase_type == 0:
            data['day_rating'].append(random.choice(negation_phrases))
            data['mood_score'].append(round(np.random.uniform(3.0, 4.5), 1))
        elif phrase_type == 1:
            data['day_rating'].append(random.choice(repetitive_neutral_phrases))
            data['mood_score'].append(round(np.random.uniform(4.0, 6.0), 1))
        else:
            data['day_rating'].append(random.choice(contradiction_phrases))
            data['mood_score'].append(round(np.random.uniform(3.5, 5.5), 1))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {num_samples + special_cases + ambiguous_cases} samples and saved to {output_file}")
    
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
    # Generate more samples by default
    df = generate_synthetic_data(num_samples=10000)
    
    # Split into train and test sets
    train_test_split_data()
    
    # Show data stats
    print("\nData statistics:")
    print(df.describe())
    
    # Show correlations with mood score
    print("\nFeature correlations with mood score:")
    correlations = df.drop(['date', 'day_rating'], axis=1).corr()['mood_score'].sort_values(ascending=False)
    print(correlations)
