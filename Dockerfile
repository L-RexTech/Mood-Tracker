FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader vader_lexicon

# Copy application code
COPY . .

# Pre-train model with fewer samples to save deployment time
RUN python train_model.py --samples 1000

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
