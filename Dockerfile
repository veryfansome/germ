# Use the official Python image from the Docker Hub
FROM python:3.9-slim

RUN mkdir /app

# Copy the requirements file
COPY app/requirements.txt /app

# Install the Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application and test files
COPY app /app
COPY tests /tests

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]