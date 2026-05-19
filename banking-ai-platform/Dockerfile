FROM python:3.11-slim

LABEL maintainer="shreyashkumar456@gmail.com"
LABEL description="Banking AI Platform — Data Engineering + AI"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "."

# Application code
COPY src/ src/
COPY config/ config/
COPY tests/ tests/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import src; print('healthy')"

# Default: run quality checks (overridden in docker-compose)
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
