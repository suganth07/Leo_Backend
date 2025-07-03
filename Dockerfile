FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for dlib-binary
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT:-10000}"]
