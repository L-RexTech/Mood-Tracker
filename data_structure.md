# Mood Tracker Dataset Structure

## Input Parameters

| Parameter     | Type     | Description                                          | Range/Values                            |
|---------------|----------|------------------------------------------------------|----------------------------------------|
| date          | datetime | Date of the mood record                              | YYYY-MM-DD format                       |
| day_rating    | text     | Subjective text description of the day               | Free text (analyzed for sentiment)      |
| water_intake  | float    | Amount of water consumed                             | 0.5-5.0 liters                          |
| people_met    | integer  | Number of people interacted with                     | 0-20 people                             |
| exercise      | integer  | Duration of physical activity                        | 0-120 minutes                           |
| sleep         | float    | Duration of sleep                                    | 4.0-11.0 hours                          |
| screen_time   | float    | Time spent using electronic devices                  | 1.0-12.0 hours                          |
| outdoor_time  | float    | Time spent outside                                   | 0.0-6.0 hours                           |
| stress_level  | category | Perceived stress level                               | "Low", "Medium", "High"                 |
| food_quality  | category | Overall nutritional quality of food consumed         | "Healthy", "Moderate", "Unhealthy"      |
| mood_score    | float    | Target variable - quantitative measure of mood       | 0.0-10.0 scale                          |

## Data Flow

1. **Training Data**: All 11 parameters are used to train the model, with `mood_score` as the target variable.

2. **API Input**: When using the API, users provide 9 parameters (all except `date` and `mood_score`).

3. **API Output**: The model predicts the `mood_score` based on the 9 input parameters.

## Parameter Relationships

The model analyzes complex relationships between these parameters, for example:
- Sleep quality may be affected by screen time
- Exercise benefits may be enhanced by outdoor time
- Stress levels often correlate with sleep and social interaction

These relationships are captured through both the base model and the derived interaction features that improve prediction accuracy.
