#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting build process..."

# Navigate to the directory containing manage.py and requirements.txt
# This assumes build.sh is placed in BACKEND/agent_ai_project/
# If your Render Root Directory is set to 'BACKEND/agent_ai_project',
# then this `cd` command might not be strictly necessary, but it's safer.
# cd /app/backend # This path is specific to Dockerfile WORKDIR, adjust if needed.

# Ensure pip is up-to-date (optional, but good practice)
pip install --upgrade pip

# Install Python dependencies from requirements.txt
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Run Django's collectstatic command to gather all static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Build process finished successfully."