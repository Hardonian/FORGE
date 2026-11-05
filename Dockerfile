# =============================================================================
# Stage 1: Build & Dependency Wheel Cache
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY forge ./forge

RUN pip install --upgrade pip && \
    pip install --prefix=/install .

# =============================================================================
# Stage 2: Hardened Production Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORGE_HOST=0.0.0.0 \
    FORGE_PORT=8000 \
    FORGE_WORKERS=4 \
    PATH="/install/bin:${PATH}" \
    PYTHONPATH="/install/lib/python3.11/site-packages:${PYTHONPATH}"

# Install curl for healthcheck and ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group
RUN groupadd -g 10001 forgegroup && \
    useradd -u 10001 -g forgegroup -m -s /bin/bash forgeuser

# Copy installed packages from builder
COPY --from=builder /install /install

# Copy project files
COPY forge ./forge
COPY examples ./examples
COPY README.md LICENSE ./
COPY docker-entrypoint.sh ./docker-entrypoint.sh

# Create persistent output directory and assign permissions
RUN mkdir -p /app/runs && \
    chmod +x /app/docker-entrypoint.sh && \
    chown -R forgeuser:forgegroup /app

USER forgeuser

EXPOSE 8000

VOLUME ["/app/runs"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["serve"]
