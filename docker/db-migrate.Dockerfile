# Use the official Python image from the Docker Hub
FROM python:3.12-slim

RUN apt-get update
RUN apt-get install -y curl netcat-traditional openjdk-17-jdk postgresql-client

# Install jbang and neo4j-migrations tool
RUN curl -sL https://sh.jbang.dev | bash
RUN /root/.jbang/bin/jbang eu.michael-simons.neo4j:neo4j-migrations-cli:2.17.3 --version
RUN printf '#!/bin/bash\n/root/.jbang/bin/jbang eu.michael-simons.neo4j:neo4j-migrations-cli:2.17.3 $@\n' > /usr/local/bin/neo4j-migrations
RUN chmod 775 /usr/local/bin/neo4j-migrations
