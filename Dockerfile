# Base image
FROM python:3.13-slim AS base

# Install system dependencies needed by some ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (package manager)
FROM base AS uv_install
RUN pip install --no-cache-dir uv

# Application image
FROM uv_install AS app

WORKDIR /app

# Copy project metadata
COPY uv.lock pyproject.toml ./

# Install dependencies using uv (fast, reproducible)
RUN uv sync --no-dev --frozen

# Copy application code
COPY src ./src
COPY models ./models

# Expose API port
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
