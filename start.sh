#!/bin/bash

# Startup script for Render deployment - Zero local storage approach
echo "Starting Leo Backend deployment with zero local storage..."

# Set environment variables for memory optimization
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp

# Optimize memory usage
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

# No need to create any model directories - we're using face_recognition with dlib-binary

# Start the application
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
