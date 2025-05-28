# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH so that app.* imports work correctly
ENV PYTHONPATH=/code

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI app using python module so PYTHONPATH is respected
CMD ["python", "-m", "app.main"]
