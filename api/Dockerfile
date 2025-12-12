# api/Dockerfile

# Use a lean Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file first to optimize cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This copies src/, api/, data/raw/, etc.
COPY . /app

# CRITICAL FIX: Explicitly set PYTHONPATH to include the src directory.
# We set it directly rather than trying to append to the potentially undefined
# system $PYTHONPATH, which resolves the warning and ensures module discoverability.
ENV PYTHONPATH=/app/src

# Set the command to run the FastAPI application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]