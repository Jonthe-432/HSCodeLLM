FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime deps first for better layer caching
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install ".[all]"

# Pre-warm the CN cache as a separate, optional step at runtime.
ENV HSCODE_CACHE_DIR=/data/cache
VOLUME ["/data/cache"]

ENTRYPOINT ["hscode"]
CMD ["--help"]
