# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Set PYTHONPATH so that "app." imports work
ENV PYTHONPATH=/code

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
