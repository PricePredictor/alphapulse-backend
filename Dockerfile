# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory to project root
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything into container (including app/)
COPY . .

# Set PYTHONPATH so "app." imports work
ENV PYTHONPATH=/code

# Expose FastAPI port
EXPOSE 8000

# Launch app (notice app.main:app now works)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
