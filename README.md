<img src="./bot/static/logo.webp" alt="logo.webp" title="Germ" style="width:200px; height:200px;">

# germ

[![tests](https://github.com/veryfansome/germ/actions/workflows/tests.yml/badge.svg)](https://github.com/veryfansome/germ/actions/workflows/tests.yml)

Germ is a chatbot project built using FastAPI and Gunicorn.

## Project Premise

The aim of this project is to create a chat-driven ML exploration system that is simple to run, maintain, and expand.

### Core Features To-Date

- Persistent chat history
- Bookmarks that allow saving and recalling entire conversations, therefore, making it possible to iteratively progress conversations on complex technical discussions, similar to how "quick save/load" can be used to incrementally progress challenging video game encounters.

## Setup

Make sure you have Docker and Docker Compose installed. Try [Rancher Desktop](https://rancherdesktop.io/).

### Environment Setup

At the root level of the project, create a `.env` file with the following content.

```shell
OPENAI_API_KEY=sk-proj-xxxxxxxxxx
POSTGRES_USER=germ
POSTGRES_PASSWORD=bacteria4life
```

## Running the Application

```bash
docker-compose up --build
```

## Running tests

```bash
docker-compose run test
```