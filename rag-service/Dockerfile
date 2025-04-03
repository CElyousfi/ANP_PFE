# rag-service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/general data/commercial data/technical data/safety data/regulatory

# Create directories for FAISS indexes
RUN mkdir -p faiss_index/general faiss_index/commercial faiss_index/technical faiss_index/safety faiss_index/regulatory

# Expose the port
EXPOSE 8001

# Command to run the application
CMD ["python", "main.py"]