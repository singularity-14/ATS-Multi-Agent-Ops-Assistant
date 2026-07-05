# ATS Agentic Ops Assistant — Production Docker Image
FROM python:3.11-slim

# Metadata
LABEL maintainer="ATS AI Core Team" \
      description="ATS Agentic Ops Assistant — Multi-agent AI for CERN accelerator operations" \
      version="1.0.0"

# Security: run as non-root
RUN useradd -m -u 1000 atsai

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Create data directories
RUN mkdir -p data/eval_reports data/docs && chown -R atsai:atsai /app

# Switch to non-root user
USER atsai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Default: run FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
