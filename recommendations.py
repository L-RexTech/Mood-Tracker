from typing import List, Literal

def generate_recommendations(
    water_intake: float,
    people_met: int,
    exercise: int,
    sleep: float,
    screen_time: float,
    outdoor_time: float,
    stress_level: Literal["Low", "Medium", "High"],
    food_quality: Literal["Healthy", "Moderate", "Unhealthy"]
) -> List[str]:
    """
    Generate personalized recommendations based on input parameters
    """
    recommendations = []
    
    # Optimal values for parameters
    optimal_water = 2.5  # liters
    optimal_exercise = 30  # minutes
    optimal_sleep = 8.0  # hours
    max_screen_time = 4.0  # hours
    optimal_outdoor_time = 1.0  # hour
    
    # Enhanced recommendations for critical health factors
    # Water intake - more urgent message for zero intake
    if water_intake == 0:
        recommendations.append("URGENT: Hydration is critical for health and mood. Start drinking water immediately and aim for at least 2.5 liters daily.")
    elif water_intake < optimal_water:
        recommendations.append(f"Try to drink at least {optimal_water} liters of water daily for better energy and mood.")
    
    # Exercise - stronger recommendation for zero exercise
    if exercise == 0:
        recommendations.append("IMPORTANT: Physical activity is essential for mood regulation. Even 15 minutes of light exercise can significantly improve your mood.")
    elif exercise < optimal_exercise:
        recommendations.append(f"Aim for at least {optimal_exercise} minutes of exercise daily to boost endorphins.")
    
    # Sleep recommendations
    if sleep < 7.0:
        recommendations.append("Insufficient sleep can affect mood. Aim for 7-9 hours of quality sleep.")
    elif sleep > 9.0:
        recommendations.append("Too much sleep can also affect mood. Try to maintain a consistent sleep schedule.")
    
    # Screen time recommendations
    if screen_time > max_screen_time:
        recommendations.append("Consider reducing screen time, especially before bed, to improve mood and sleep quality.")
    
    # Outdoor time - stronger recommendation for zero outdoor time
    if outdoor_time == 0:
        recommendations.append("IMPORTANT: Time outdoors is vital for vitamin D production and mental health. Try to get outside for at least 30 minutes daily.")
    elif outdoor_time < optimal_outdoor_time:
        recommendations.append("Spending time outdoors in natural light can improve mood. Try to get outside for at least an hour daily.")
    
    if stress_level in ["Medium", "High"]:
        stress_recommendations = [
            "Practice deep breathing exercises for 5 minutes when feeling stressed.",
            "Try meditation to manage stress and improve mindfulness.",
            "Consider journaling to process thoughts and reduce stress.",
            "Take short breaks throughout the day to reset and refocus."
        ]
        recommendations.extend(stress_recommendations[:2])  # Add 2 stress recommendations
    
    if food_quality in ["Moderate", "Unhealthy"]:
        food_recommendations = [
            "Incorporate more fruits and vegetables into your diet for better mood and energy.",
            "Reduce processed foods and sugar which can cause energy crashes.",
            "Stay hydrated and eat regular meals to stabilize mood.",
            "Consider foods rich in omega-3 fatty acids which can support brain health."
        ]
        recommendations.extend(food_recommendations[:2])  # Add 2 food recommendations
    
    if people_met < 2:
        recommendations.append("Social interaction can improve mood. Try to connect with friends or family regularly.")
    
    # Prioritize recommendations for zero values
    zero_recommendations = [r for r in recommendations if r.startswith("URGENT") or r.startswith("IMPORTANT")]
    other_recommendations = [r for r in recommendations if not (r.startswith("URGENT") or r.startswith("IMPORTANT"))]
    
    # Combine with zero recommendations first
    prioritized_recommendations = zero_recommendations + other_recommendations
    
    # Limit to top 5 recommendations to avoid overwhelming the user
    return prioritized_recommendations[:5]
