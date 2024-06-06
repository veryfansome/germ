# Use the official Python image from the Docker Hub
FROM python:3.12-slim

RUN mkdir -p /src/bot

# Copy the requirements file
COPY bot/requirements.txt /src/bot

# Install the Python dependencies
RUN pip install --no-cache-dir -r /src/bot/requirements.txt

# Copy the application and test files. Not currently needed, mounted by `docker compose`.
#COPY bot /src/bot
#COPY tests /src/tests