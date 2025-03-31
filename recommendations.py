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
    
    # Check each parameter and provide recommendations
    if water_intake < optimal_water:
        recommendations.append(f"Try to drink at least {optimal_water} liters of water daily for better energy and mood.")
    
    if exercise < optimal_exercise:
        recommendations.append(f"Aim for at least {optimal_exercise} minutes of exercise daily to boost endorphins.")
    
    if sleep < 7.0:
        recommendations.append("Insufficient sleep can affect mood. Aim for 7-9 hours of quality sleep.")
    elif sleep > 9.0:
        recommendations.append("Too much sleep can also affect mood. Try to maintain a consistent sleep schedule.")
    
    if screen_time > max_screen_time:
        recommendations.append("Consider reducing screen time, especially before bed, to improve mood and sleep quality.")
    
    if outdoor_time < optimal_outdoor_time:
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
    
    # Limit to top 5 recommendations to avoid overwhelming the user
    return recommendations[:5]
