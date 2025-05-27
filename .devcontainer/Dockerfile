FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
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
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Install additional development tools
RUN pip3 install jupyter notebook jupyterlab black flake8 pytest pytest-cov

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Create vscode user
RUN useradd -m -s /bin/bash vscode && \
    usermod -aG sudo vscode
USER vscode
