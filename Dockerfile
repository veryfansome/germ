# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Copy the application and test files
COPY app /app
COPY tests /tests

# Install the Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt