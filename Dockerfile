FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-opencv \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Create non-root user and fix Ultralytics permissions
RUN useradd -m -u 1000 visionflow && \
    mkdir -p /home/visionflow/.config/Ultralytics && \
    chown -R visionflow:visionflow /app /home/visionflow/.config/Ultralytics
USER visionflow

# Expose port for Cloud Run
EXPOSE 8080

# Run web server with Gunicorn
CMD ["gunicorn", "--workers=1", "--threads=2", "--bind=0.0.0.0:8080", "scripts.web:app"]
