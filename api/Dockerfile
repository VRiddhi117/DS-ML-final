FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy model + code
COPY api/best_model.joblib /app/best_model.joblib
COPY api/app.py /app/app.py
COPY api/housing_pipeline.py /app/housing_pipeline.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
