# Use the official Python image from the Docker Hub
FROM python:3.12-slim

RUN apt-get update
RUN apt-get install -y netcat-traditional postgresql-client
RUN mkdir -p /src
RUN mkdir -p /var/lib/germ/models

# Copy the requirements file
COPY requirements.txt /src

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /src/requirements.txt
RUN opentelemetry-bootstrap -a install