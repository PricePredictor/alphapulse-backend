# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH so "app." imports work
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app using the correct import path
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
