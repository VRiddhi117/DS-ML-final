# Dockerfile (API)
# Place this file at: api/Dockerfile

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy API code + models (important for Render)
COPY api /app/api
COPY models /app/models
# Optional: if you want to include data files inside the image (usually not needed)
# COPY data /app/data

# Expose port Render commonly uses; Render will override via $PORT
EXPOSE 8000

# Make sure app.py lives at api/app.py and FastAPI object is named "app"
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
