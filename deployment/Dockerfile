# Dockerfile for AWS deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including NeMo requirements
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libsox-dev \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional NeMo dependencies for production
RUN pip install --no-cache-dir \
    apex \
    cython \
    soundfile \
    librosa

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=5000
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]