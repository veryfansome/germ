# germ

This is a simple FastAPI project served using Gunicorn and Docker.

## Setup

Make sure you have Docker and Docker Compose installed.

### Environment Setup

At the root level of the project, create a `.env` file with the following content.

```shell
OPENAI_API_KEY=sk-proj-xxxxxxxxxx
CHAT_HISTORY_POSTGRES_USER=germ
CHAT_HISTORY_POSTGRES_PASSWORD=bacteria4life
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

Project structure created by ChatGPT via

```text
Make a simple green unicorn project with a docker-compose
file, unit tests, and GitHub actions setup to run unit tests
on PR and push.
```

[Design docs](./docs)