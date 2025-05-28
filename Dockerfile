# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory to project root
WORKDIR /

# System dependencies (optional, e.g., for LightGBM)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project code into container
COPY . .

# Set PYTHONPATH so "app." imports work
ENV PYTHONPATH=.

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app (note: app.main because main.py is inside /app)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
