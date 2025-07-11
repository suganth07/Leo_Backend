# ================================
# BUILDER STAGE - Heavy dependencies and compilation
# ================================
FROM python:3.11-slim-bookworm AS builder

# Set working directory
WORKDIR /app

# Install minimal build dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install to a specific directory
COPY requirements.txt .

# Create virtual environment and install dependencies with aggressive cleanup
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with size optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=1000 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout=1000 -r requirements.txt && \
    pip cache purge && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + || true && \
    find /opt/venv -type f -name "*.pyx" -delete && \
    find /opt/venv -type f -name "*.pxd" -delete && \
    find /opt/venv -type f -name "*.c" -delete && \
    find /opt/venv -type f -name "*.cpp" -delete && \
    find /opt/venv -type f -name "*.so" -exec strip {} \; || true

# Pre-download and cache InsightFace models in builder stage
RUN mkdir -p /root/.insightface
ENV INSIGHTFACE_HOME=/root/.insightface
RUN python -c "import insightface; import os; print('Starting model download...'); app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640)); print('Models cached successfully')" || echo "Model download attempted"

# ================================
# RUNTIME STAGE - Ultra-minimal runtime dependencies
# ================================
FROM python:3.11-slim-bookworm AS runner

# Install only essential runtime libraries (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    libavcodec59 \
    libavformat59 \
    libswscale6 \
    libavutil57 \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/* /usr/share/doc/* /usr/share/man/*

# Copy virtual environment from builder (contains all Python packages)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pre-downloaded InsightFace models from builder
COPY --from=builder /root/.insightface /home/.insightface

# Set working directory
WORKDIR /app

# Copy only the application file
COPY insight.py .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /home/.insightface

# Switch to non-root user
USER appuser

# Set optimized environment variables for production
ENV INSIGHTFACE_HOME=/home/.insightface \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    PATH="/opt/venv/bin:$PATH"

# Expose port
EXPOSE 10001

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:10001/health', timeout=10)" || exit 1

# Run insight.py
CMD ["python", "insight.py"]
