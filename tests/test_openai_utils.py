from api.models import ChatMessage


def test_handle_negative_feedback():
    chat_frame: list[ChatMessage] = [
        ChatMessage(role="user",
                    content="draw a cow"),
        ChatMessage(role="assistant",
                    content="I can generate an image of a cow based on a textual prompt. Let me do that for you."),
        ChatMessage(role="user",
                    content="you used the wrong model to respond, the correct model would have been dall-e-2"),
        ChatMessage(role="assistant",
                    content="Ok. I've updated my model selection behavior based on \"draw a cow\" "
                            + "and your feedback that `dall-e-2` is the correct model"),
        ChatMessage(role="user",
                    content="draw a duck"),
    ]
    pass
