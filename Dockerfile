FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# port
EXPOSE 8000

# Run unified FastAPI + Dash server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
