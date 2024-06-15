# Use the official Python image from the Docker Hub
FROM python:3.12-slim

RUN apt-get update
RUN apt-get install -y postgresql-client
RUN mkdir -p /src/bot

# Copy the requirements file
COPY bot/requirements.txt /src/bot

# Install the Python dependencies
RUN python -m venv germ && . germ/bin/activate
RUN pip install --no-cache-dir -r /src/bot/requirements.txt
RUN opentelemetry-bootstrap -a install