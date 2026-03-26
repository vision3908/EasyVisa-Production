# Start from Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code — api.py at root, src/ package for ML logic
COPY api.py .
COPY src/ ./src/

# Expose port 8000
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 