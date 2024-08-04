# germ

[![tests](https://github.com/veryfansome/germ/actions/workflows/tests.yml/badge.svg)](https://github.com/veryfansome/germ/actions/workflows/tests.yml)

Germ is a chatbot project that aims to create a chat-driven ML exploration system that is simple to run, maintain, and expand. The underlying chat completion capability is powered by OpenAI's GPT (3.5, 4, 4o, etc.). DALL-E can also be used to return image requests. Other OpenAI models can be integrated but have not been integrated yet (PRs plz!).

The chat experience and interface is barebones compared to ChatGPT, but it does a good enough job. The pacing is slightly different because ChatGPT streams responses, which results in an immediate response that can take a long time to complete. Germ doesn't stream, it asks for the whole responses all at once, which means a slower initial response but a faster overall result on longer form responses, which is often the case when you're asking for generated code.

### Core Features
You might be thinking, "Meh, why not just use ChatGPT?". Well, you own all your data and have more control to customize your experience. Here are some features I find appealing.

- Persistent chat history stored in PostgreSQL can be mined for training data
- Chat history enables "bookmarks" that allow saving and recalling entire conversations, therefore, making it possible to iteratively progress conversations on complex technical discussions, similar to how "quick save/load" can be used to incrementally progress challenging video game encounters.
- Event-based design that allows adding new behavior without increasing overall wait times for users
- "Handler" pattern enable rich multi-model responses and make it easy to explore new ideas

### Design
![Design diagram](https://github.com/user-attachments/assets/fdee35ea-c40b-4538-a0c3-df11765e54c2)

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

## Interface

So basic, anyone could use it!

### Chat

On the botton right, we can see previous sessions, which have been bookmarked automatically. Clicking on them will recall the conversation, allowing the discussion to continue in a new session. The chat interface is very barebones.

![Chat with bookmarks](https://github.com/user-attachments/assets/4f6afee0-f740-470b-9010-588ffbac621e)

The chatbox loves Markdown. Responses render Markdown and if you enter Markdown in the textarea, it will be rendered also. You can copy anything you send or get from the bot using the copy button, which you can see on the bottom left. If you wish to re-enter something you typed previously, you can up-arrow like on a terminal.

![Markdown with copy button](https://github.com/user-attachments/assets/4daf76d0-6b91-49b2-8dec-fb651172a3a0)

### PostgreSQL

For convenience, you can directly query the DB, which is very helpful when debugging.

![Postgres](https://github.com/user-attachments/assets/20f12445-6b9d-4285-a494-169079ff6b03)
