# Mood Tracker Deployment Guide

This guide covers multiple free deployment options for your Mood Tracker API.

## Option 1: Render.com (Easiest)

[Render](https://render.com/) offers a free tier for web services with automatic deployment from GitHub.

### Setup Instructions:

1. **Create a GitHub repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/mood-tracker.git
   git push -u origin main
   ```

2. **Create a `render.yaml` file**:

   ```yaml
   services:
     - type: web
       name: mood-tracker-api
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
       envVars:
         - key: PYTHON_VERSION
           value: 3.9
   ```

3. **Sign up at [render.com](https://render.com/)** and connect your GitHub repo.

4. **Select "Blueprint" deployment** and use your GitHub repository.

5. Render will automatically detect the configuration and deploy your API.

## Option 2: Fly.io (More Performance)

[Fly.io](https://fly.io/) offers a generous free tier with global deployment.

### Setup Instructions:

1. **Install Fly CLI**:
   ```bash
   # On macOS
   brew install flyctl
   # Or use the install script
   curl -L https://fly.io/install.sh | sh
   ```

2. **Create a `Dockerfile`**:

   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app
   COPY . .

   RUN pip install --no-cache-dir -r requirements.txt

   # Install libomp for XGBoost
   RUN apt-get update && apt-get install -y libgomp1

   # Initialize the model
   RUN python train_model.py

   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
   ```

3. **Create a `fly.toml` file**:

   ```toml
   app = "mood-tracker-api"
   
   [build]
     dockerfile = "Dockerfile"

   [http_service]
     internal_port = 8080
     force_https = true
   ```

4. **Deploy to Fly.io**:
   ```bash
   fly auth login
   fly launch --no-deploy
   fly deploy
   ```

## Option 3: Railway (Great Developer Experience)

[Railway](https://railway.app/) offers a generous free tier with an excellent developer experience and simple deployment process.

### Setup Instructions:

1. **Create a GitHub repository** for your Mood Tracker API if you haven't already done so.

2. **Create a `railway.json` file** in your project root:

   ```json
   {
     "$schema": "https://railway.app/railway.schema.json",
     "build": {
       "builder": "NIXPACKS",
       "buildCommand": "pip install -r requirements.txt"
     },
     "deploy": {
       "startCommand": "uvicorn app:app --host 0.0.0.0 --port $PORT",
       "restartPolicyType": "ON_FAILURE",
       "restartPolicyMaxRetries": 3
     }
   }
   ```

3. **Create a `.env` file** (optional, for any environment variables you might need):

   ```
   PORT=8000
   ENVIRONMENT=production
   ```

4. **Create a `Procfile`** as a fallback:

   ```
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

5. **Sign up at [railway.app](https://railway.app/)** and connect your GitHub account.

6. **Create a new project** and select "Deploy from GitHub repo."

7. **Select your repository** and Railway will automatically detect your configuration and deploy your app.

8. **Setup Resources**: In the Railway dashboard, you can add a database or other services if needed.

### Railway Advantages:

- Simple deployment process directly from GitHub
- Good free tier (up to $5 of usage or 500 hours per month)
- Automatic HTTPS support
- Real-time logs and metrics
- Built-in database options if you need them
- Quick and responsive deployments

### Testing Your Deployment:

Once deployed, Railway will provide you with a public URL (like `https://mood-tracker-production.up.railway.app`). You can test your API by appending the endpoint.

## Option 4: Hugging Face Spaces (AI-focused)

[Hugging Face Spaces](https://huggingface.co/spaces) is perfect for AI applications.

### Setup Instructions:

1. **Create an account** on [Hugging Face](https://huggingface.co/).

2. **Create a new Space** with FastAPI template.

3. **Create a `requirements.txt`** file with all your dependencies.

4. **Upload your code**, making sure to include the pre-trained model file.

5. **Add a `Dockerfile`**:

   ```dockerfile
   FROM python:3.9

   WORKDIR /code

   COPY ./requirements.txt /code/requirements.txt
   RUN pip install --no-cache-dir -r /code/requirements.txt

   COPY . /code

   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

## Option 5: Streamlit Community Cloud (For UI)

If you want to add a simple UI to your model:

1. **Create a Streamlit app**:

   ```python
   # filepath: /Users/piyush.verma1/Mood Tracker/streamlit_app.py
   import streamlit as st
   import requests
   import json

   st.title("Mood Tracker")

   with st.form("mood_input_form"):
       col1, col2 = st.columns(2)
       
       with col1:
           day_rating = st.text_area("How was your day?", height=100)
           water_intake = st.slider("Water intake (liters)", 0.0, 5.0, 1.5, 0.1)
           people_met = st.slider("People met", 0, 20, 3)
           exercise = st.slider("Exercise (minutes)", 0, 120, 30)
       
       with col2:
           sleep = st.slider("Sleep (hours)", 4.0, 11.0, 7.0, 0.1)
           screen_time = st.slider("Screen time (hours)", 0.0, 12.0, 3.0, 0.1)
           outdoor_time = st.slider("Outdoor time (hours)", 0.0, 6.0, 1.0, 0.1)
           stress_level = st.selectbox("Stress level", ["Low", "Medium", "High"])
           food_quality = st.selectbox("Food quality", ["Healthy", "Moderate", "Unhealthy"])
       
       submit_button = st.form_submit_button(label='Predict Mood')
       
   if submit_button:
       # Use local API if running on same machine
       url = "http://localhost:8000/predict"
       
       # Prepare payload
       payload = {
           "day_rating": day_rating,
           "water_intake": water_intake,
           "people_met": people_met,
           "exercise": exercise,
           "sleep": sleep,
           "screen_time": screen_time,
           "outdoor_time": outdoor_time,
           "stress_level": stress_level,
           "food_quality": food_quality
       }
       
       try:
           # Make API request
           response = requests.post(url, json=payload)
           result = response.json()
           
           # Display results
           st.header(f"Mood Score: {result['mood_score']}/10")
           
           # Create a progress bar for the score
           st.progress(result['mood_score'] / 10.0)
           
           # Display recommendations
           st.subheader("Recommendations:")
           for rec in result['recommendations']:
               st.markdown(f"- {rec}")
               
       except Exception as e:
           st.error(f"Error connecting to API: {e}")
           st.info("Make sure the FastAPI server is running locally on port 8000")
   ```

2. **Create a `packages.txt` file**:
   ```
   libgomp1
   ```

3. **Add your app to a GitHub repository, then deploy on [Streamlit Community Cloud](https://streamlit.io/cloud)**.

## Option 6: Embedding in a Static Website (GitHub Pages)

For a lightweight solution that runs the model in the browser:

1. Use [PyScript](https://pyscript.net/) to run Python code in the browser.
2. Host the website on GitHub Pages.

## Option 7: Cloud Functions

For a serverless approach, convert your API to cloud functions:

- [Google Cloud Functions](https://cloud.google.com/functions) (Free tier)
- [Azure Functions](https://azure.microsoft.com/en-us/services/functions/) (Free tier)
- [AWS Lambda](https://aws.amazon.com/lambda/) (Free tier)

## Tips for Any Deployment

1. **Precompute the model**: Include the trained model file in your repository.
2. **Optimize dependencies**: Remove unnecessary packages to reduce container size.
3. **Set up proper error handling**: Make your API robust to various input conditions.
4. **Add monitoring**: Use free logging services to track performance.
