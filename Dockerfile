FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies needed by PyTorch and basic operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Use headless OpenCV to avoid X11/GUI library requirements
RUN sed -i 's/opencv-python/opencv-python-headless/g' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code (ignoring data, models, environments via .dockerignore)
COPY . .

# Expose the application port
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
