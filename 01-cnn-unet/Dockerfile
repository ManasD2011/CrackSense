# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code
COPY src/      ./src/
COPY api/      ./api/
COPY frontend/ ./frontend/
COPY saved_models/ ./saved_models/

# Environment
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "api/main.py"]
