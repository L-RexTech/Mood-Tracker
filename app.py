from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Literal
from model import predict_mood_score, train_model_if_needed
from recommendations import generate_recommendations
import os

# Check if model exists, train if needed
train_model_if_needed()

app = FastAPI(title="Mood Tracker API", 
              description="API to predict mood score based on daily habits and provide recommendations")

class MoodInput(BaseModel):
    day_rating: str = Field(..., description="Text description of your day")
    water_intake: float = Field(..., description="Water intake in liters", ge=0)
    people_met: int = Field(..., description="Number of people met", ge=0)
    exercise: int = Field(..., description="Exercise duration in minutes", ge=0)
    sleep: float = Field(..., description="Sleep duration in hours", ge=0)
    screen_time: float = Field(..., description="Screen time in hours", ge=0)
    outdoor_time: float = Field(..., description="Time spent outdoors in hours", ge=0)
    stress_level: Literal["Low", "Medium", "High"] = Field(..., description="Stress level")
    food_quality: Literal["Healthy", "Moderate", "Unhealthy"] = Field(..., description="Food quality")

    @validator('water_intake', 'sleep', 'screen_time', 'outdoor_time')
    def check_reasonable_values(cls, v, values, field):
        max_values = {
            'water_intake': 10,
            'sleep': 24,
            'screen_time': 24,
            'outdoor_time': 24
        }
        if v > max_values[field.name]:
            raise ValueError(f"{field.name} seems unreasonably high")
        return v

class MoodOutput(BaseModel):
    mood_score: float = Field(..., description="Predicted mood score (0-10)")
    recommendations: List[str] = Field(..., description="Personalized recommendations")

@app.post("/predict", response_model=MoodOutput, summary="Predict mood score")
async def predict_mood(input_data: MoodInput):
    try:
        # Predict mood score
        mood_score = predict_mood_score(
            day_rating=input_data.day_rating,
            water_intake=input_data.water_intake,
            people_met=input_data.people_met,
            exercise=input_data.exercise,
            sleep=input_data.sleep,
            screen_time=input_data.screen_time,
            outdoor_time=input_data.outdoor_time,
            stress_level=input_data.stress_level,
            food_quality=input_data.food_quality
        )
        
        # Generate recommendations
        recommendations = generate_recommendations(
            water_intake=input_data.water_intake,
            people_met=input_data.people_met,
            exercise=input_data.exercise,
            sleep=input_data.sleep,
            screen_time=input_data.screen_time,
            outdoor_time=input_data.outdoor_time,
            stress_level=input_data.stress_level,
            food_quality=input_data.food_quality
        )
        
        return MoodOutput(mood_score=round(mood_score, 1), recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
