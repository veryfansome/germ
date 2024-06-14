# germ

[![tests](https://github.com/veryfansome/germ/actions/workflows/tests.yml/badge.svg)](https://github.com/veryfansome/germ/actions/workflows/tests.yml)

Germ is a chatbot project built using FastAPI and Gunicorn.

<img src="./bot/static/logo.webp" alt="logo.webp" title="Germ" style="width:200px; height:200px;">

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

## AI Assisted design and implementation

[Design docs](./docs)