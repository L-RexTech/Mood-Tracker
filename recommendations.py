from typing import List, Literal, Tuple

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
    # Store recommendations with priority: (recommendation, priority)
    recommendations: List[Tuple[str, str]] = []
    
    # Optimal values for parameters
    optimal_water = 2.5  # liters
    optimal_exercise = 30  # minutes
    optimal_sleep = 8.0  # hours
    max_screen_time = 4.0  # hours
    optimal_outdoor_time = 1.0  # hour
    
    # Enhanced recommendations for critical health factors
    # Water intake - more urgent message for zero intake
    if water_intake == 0:
        recommendations.append(("Hydration is critical for health and mood. Start drinking water immediately and aim for at least 2.5 liters daily.", "High"))
    elif water_intake < optimal_water:
        recommendations.append((f"Try to drink at least {optimal_water} liters of water daily for better energy and mood.", "Medium"))
    
    # Exercise - stronger recommendation for zero exercise
    if exercise == 0:
        recommendations.append(("Physical activity is essential for mood regulation. Even 15 minutes of light exercise can significantly improve your mood.", "High"))
    elif exercise < optimal_exercise:
        recommendations.append((f"Aim for at least {optimal_exercise} minutes of exercise daily to boost endorphins.", "Medium"))
    
    # Sleep recommendations
    if sleep < 7.0:
        if sleep < 5.0:
            recommendations.append(("Sleep is crucial for mental health. Aim for at least 7-9 hours of sleep per night.", "High"))
        else:
            recommendations.append(("Try to get at least 7-9 hours of sleep per night for optimal mood and cognitive function.", "Medium"))
    elif sleep > 9.0:
        recommendations.append(("Too much sleep can also affect mood. Try to maintain a consistent sleep schedule.", "Medium"))
    
    # Screen time recommendations
    if screen_time > max_screen_time:
        recommendations.append(("Consider reducing screen time, especially before bed, to improve mood and sleep quality.", "Medium"))
    
    # Outdoor time - stronger recommendation for zero outdoor time
    if outdoor_time == 0:
        recommendations.append(("Time outdoors is vital for vitamin D production and mental health. Try to get outside for at least 30 minutes daily.", "High"))
    elif outdoor_time < optimal_outdoor_time:
        recommendations.append(("Spending time outdoors in natural light can improve mood. Try to get outside for at least an hour daily.", "Medium"))
    
    if stress_level in ["Medium", "High"]:
        stress_priority = "High" if stress_level == "High" else "Medium"
        stress_recommendations = [
            ("Practice deep breathing exercises for 5 minutes when feeling stressed.", stress_priority),
            ("Try meditation to manage stress and improve mindfulness.", stress_priority),
            ("Consider journaling to process thoughts and reduce stress.", "Medium"),
            ("Take short breaks throughout the day to reset and refocus.", "Medium")
        ]
        recommendations.extend(stress_recommendations[:2])  # Add 2 stress recommendations
    
    if food_quality in ["Moderate", "Unhealthy"]:
        food_priority = "High" if food_quality == "Unhealthy" else "Medium"
        food_recommendations = [
            ("Incorporate more fruits and vegetables into your diet for better mood and energy.", food_priority),
            ("Reduce processed foods and sugar which can cause energy crashes.", food_priority),
            ("Stay hydrated and eat regular meals to stabilize mood.", "Medium"),
            ("Consider foods rich in omega-3 fatty acids which can support brain health.", "Low")
        ]
        recommendations.extend(food_recommendations[:2])  # Add 2 food recommendations
    
    if people_met < 2:
        social_priority = "Medium" if people_met == 1 else "High"
        recommendations.append(("Social interaction can improve mood. Try to connect with friends or family regularly.", social_priority))
    
    # Sort recommendations by priority
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    sorted_recommendations = sorted(recommendations, key=lambda x: priority_order[x[1]])
    
    # Format recommendations with priority
    formatted_recommendations = [f"{priority} Priority: {rec}" for rec, priority in sorted_recommendations]
    
    # Limit to top 5 recommendations to avoid overwhelming the user
    return formatted_recommendations[:5]
