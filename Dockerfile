# Multi-stage build for dependency caching
# Stage 1: Build dependencies
FROM python:3.10-slim as dependencies

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables for pip caching and TensorFlow
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Create cache directory and install dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Production image
FROM python:3.10-slim as production

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy Python packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy the application code
COPY . .

# Create directories for data and models
RUN mkdir -p data models

# Expose the port the app runs on
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/healthcheck || exit 1

# Run the application
CMD ["python", "app.py"]
