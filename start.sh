#!/bin/bash

# Startup script for Render deployment
echo "Starting Leo Backend deployment..."

# Set environment variables for optimization
export PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp
export INSIGHTFACE_HOME=/tmp/.insightface

# Create necessary directories
mkdir -p /tmp/.insightface

# Start the application
python sample.py
