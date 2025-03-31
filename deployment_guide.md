# Mood Tracker Deployment Guide

This guide covers multiple free deployment options for your Mood Tracker API.

## Option 1: PythonAnywhere (Simplest)

PythonAnywhere is a Python-specific platform that makes deployment extremely easy.

See the detailed instructions in [pythonanywhere_setup.md](pythonanywhere_setup.md)

## Option 2: Heroku (Reliable)

Heroku offers a simple deployment process with good reliability.

### Setup Instructions:

1. **Install the Heroku CLI**:
   ```bash
   # On macOS
   brew install heroku
   # Or use the installer from heroku.com
   ```

2. **Create a `runtime.txt` file**:
   ```
   python-3.9.16
   ```

3. **Make sure your `Procfile` is correctly set up**:
   ```
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

4. **Login and create a new Heroku app**:
   ```bash
   heroku login
   heroku create mood-tracker-app
   ```

5. **Deploy your app**:
   ```bash
   git push heroku main
   ```

## Option 3: Netlify Functions (Serverless)

Netlify offers serverless functions with a generous free tier.

### Setup Instructions:

1. **Create a `netlify.toml` file**:
   ```toml
   [build]
     command = "pip install -r requirements.txt && python train_model.py --samples 1000"
     publish = "docs"
     functions = "functions"
   ```

2. **Create a serverless function** in `functions/mood-predict.py`:
   ```python
   from http.server import BaseHTTPRequestHandler
   import json
   import sys
   import os
   
   # Add the parent directory to the path
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   
   from model import predict_mood_score 
   from recommendations import generate_recommendations
   
   def handler(event, context):
     # Parse the request body
     try:
         body = json.loads(event['body'])
         
         # Extract parameters
         day_rating = body.get('day_rating', '')
         water_intake = float(body.get('water_intake', 0))
         people_met = int(body.get('people_met', 0))
         exercise = int(body.get('exercise', 0))
         sleep = float(body.get('sleep', 0))
         screen_time = float(body.get('screen_time', 0))
         outdoor_time = float(body.get('outdoor_time', 0))
         stress_level = body.get('stress_level', 'Medium')
         food_quality = body.get('food_quality', 'Moderate')
         
         # Predict mood score
         mood_score = predict_mood_score(
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
         
         # Generate recommendations
         recommendations = generate_recommendations(
             water_intake=water_intake,
             people_met=people_met,
             exercise=exercise,
             sleep=sleep,
             screen_time=screen_time,
             outdoor_time=outdoor_time,
             stress_level=stress_level,
             food_quality=food_quality
         )
         
         # Return the result
         return {
             'statusCode': 200,
             'headers': {'Content-Type': 'application/json'},
             'body': json.dumps({
                 'mood_score': round(mood_score, 1),
                 'recommendations': recommendations
             })
         }
     except Exception as e:
         return {
             'statusCode': 500,
             'headers': {'Content-Type': 'application/json'},
             'body': json.dumps({'error': str(e)})
         }
   ```

3. **Create a simple HTML frontend** in `docs/index.html`

4. **Connect to the Netlify CLI** and deploy:
   ```bash
   npm install -g netlify-cli
   netlify deploy
   ```

## Option 4: Simple Docker Deployment on Any Provider

You can use our Dockerfile to deploy on any platform that supports Docker:

1. **Build the Docker image**:
   ```bash
   docker build -t mood-tracker .
   ```

2. **Run the container locally to test**:
   ```bash
   docker run -p 8000:8000 mood-tracker
   ```

3. **Push to any Docker-based hosting service**:
   - DigitalOcean App Platform
   - Google Cloud Run
   - AWS App Runner

## Comparing Deployment Options

| Service | Ease of Use | Free Tier | Setup Time | Best For |
|---------|------------|-----------|------------|----------|
| PythonAnywhere | ★★★★★ | Yes | 10-15 min | Beginners |
| Heroku | ★★★★☆ | Yes | 15-20 min | Most use cases |
| Netlify | ★★★☆☆ | Yes | 20-30 min | Static site + API |
| Docker | ★★★☆☆ | Depends on provider | 20+ min | Custom requirements |
