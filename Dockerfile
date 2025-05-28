# Set working directory to project root
WORKDIR /

# Copy everything into container
COPY . .

# Ensure Python can see the app module
ENV PYTHONPATH=.

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
