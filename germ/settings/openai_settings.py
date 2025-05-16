import os

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o"
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL") or "dall-e-3"
MINI_MODEL = os.getenv("OPENAI_MINI_MODEL") or "gpt-4o-mini"
REASONING_MODEL = os.getenv("OPENAI_REASONING_MODEL") or None
SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL") or "gpt-4o"
ROUTING_MODEL = MINI_MODEL

ENABLED_CHAT_MODELS = (
    *([] if not CHAT_MODEL else [CHAT_MODEL]),
    *([] if not REASONING_MODEL else [REASONING_MODEL]),
)
ENABLED_IMAGE_MODELS = (
    *([] if not IMAGE_MODEL else [IMAGE_MODEL]),
)
HTTPX_TIMEOUT = 30

ASSISTANT_INSTRUCTION = (os.getenv("OPENAI_ASSISTANT_INSTRUCTION")
                         or ("Be a helpful assistant. "
                             "Answer in valid Markdown format only."))
ASSISTANT_NAME = os.getenv("OPENAI_ASSISTANT_NAME") or "_GERM_:Assistant"

SUMMARIZER_INSTRUCTION = (os.getenv("OPENAI_SUMMARIZER_INSTRUCTION")
                          or ("Use tools to write code for reading attached files. "
                              "Answer in valid Markdown format only."))
SUMMARIZER_NAME = os.getenv("OPENAI_ASSISTANT_NAME") or "_GERM_:Summarizer"
