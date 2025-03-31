#!/usr/bin/env python3
import sys
from model import analyze_day_rating, predict_mood_score

def test_day_rating(description):
    """Test how a day description is analyzed"""
    sentiment = analyze_day_rating(description)
    
    print(f"\nAnalyzing: '{description}'")
    print(f"Sentiment score: {sentiment:.2f} (0-1 scale)")
    print(f"Approximate mood score: {sentiment * 10:.1f} (0-10 scale)")
    
    # Test full prediction with neutral other metrics
    prediction = predict_mood_score(
        day_rating=description,
        water_intake=2.0,
        people_met=3,
        exercise=30,
        sleep=7.5,
        screen_time=3.0,
        outdoor_time=1.0,
        stress_level="Low",
        food_quality="Moderate"
    )
    
    print(f"Full prediction with neutral metrics: {prediction:.1f}")
    
    # Test contradiction handling (good metrics + bad day)
    contradiction_prediction = predict_mood_score(
        day_rating=description,
        water_intake=3.0,        # Good
        people_met=5,            # Good
        exercise=60,             # Good
        sleep=8.0,               # Good
        screen_time=2.0,         # Good
        outdoor_time=2.0,        # Good
        stress_level="Low",      # Good
        food_quality="Healthy"   # Good
    )
    
    print(f"Prediction with good metrics: {contradiction_prediction:.1f}")
    print(f"Does sentiment override good metrics? {'Yes' if contradiction_prediction < 7.0 else 'No'}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_day_rating(" ".join(sys.argv[1:]))
    else:
        # Test a few examples
        test_phrases = [
            "It was a bad day",
            "It was a good day",
            "Bad day overall",
            "Just okay okay",
            "Not a good day",
            "Had a terrible day",
            "Feeling frustrated despite everything",
            "Everything was fine but I felt sad"
        ]
        
        for phrase in test_phrases:
            test_day_rating(phrase)
            print("-" * 50)
