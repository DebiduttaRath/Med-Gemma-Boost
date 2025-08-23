# AWS Deployment Dockerfile for MedGemma Healthcare AI Platform
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir streamlit requests Pillow PyPDF2 accelerate datasets faiss-cpu huggingface-hub nltk numpy pandas pdfplumber peft plotly rouge-score safetensors scikit-learn sentence-transformers torch transformers trl openai

# Copy application files
COPY . .

# Create .streamlit directory and config
RUN mkdir -p .streamlit
RUN echo "\
[server]\n\
headless = true\n\
address = \"0.0.0.0\"\n\
port = 5000\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > .streamlit/config.toml

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:5000/_stcore/health

# Start command
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]