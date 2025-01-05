# Use the official Python image from the Docker Hub
FROM python:3.12-slim

RUN apt-get update
RUN apt-get install -y curl netcat-traditional openjdk-17-jdk postgresql-client
RUN mkdir -p /src
RUN mkdir -p /var/lib/germ/models

# Copy the requirements file
COPY requirements.txt /src

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /src/requirements.txt
RUN opentelemetry-bootstrap -a install

# Install jbang and neo4j-migrations tool
RUN curl -sL https://sh.jbang.dev | bash
RUN /root/.jbang/bin/jbang trust add https://github.com/neo4j/
RUN /root/.jbang/bin/jbang neo4j-migrations@neo4j --version
RUN printf '#!/bin/bash\n/root/.jbang/bin/jbang neo4j-migrations@neo4j $@\n' > /usr/local/bin/neo4j-migrations
RUN chmod 775 /usr/local/bin/neo4j-migrations
