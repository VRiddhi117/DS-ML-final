FROM python:3.11-slim

WORKDIR /app

# Install deps first (better caching)
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy API code + model (source -> destination)
COPY app.py /app/app.py
COPY best_model.joblib /app/best_model.joblib

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
