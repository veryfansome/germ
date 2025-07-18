# Use the official Python image from the Docker Hub
FROM python:3.12-slim

RUN apt-get update
RUN apt-get install -y \
    curl \
    default-mysql-client \
    less \
    netcat-traditional \
    pigz \
    postgresql-client \
    unzip

# Install Python dependencies
RUN pip install --upgrade pip
