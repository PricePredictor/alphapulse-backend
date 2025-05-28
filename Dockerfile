# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Let Python recognize "app." as importable module
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app correctly from inside app folder
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
