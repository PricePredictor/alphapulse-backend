FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Install system-level dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Set PYTHONPATH for correct imports
ENV PYTHONPATH=/code

# Expose FastAPI default port
EXPOSE 8000

# Launch the app correctly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
