#!/usr/bin/env bash
# Render build script: install dependencies, then train the model
set -e

echo "=== Python version ==="
python --version

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Training model (this runs once at build time) ==="
python model_pipeline.py

echo "=== Build complete ==="
