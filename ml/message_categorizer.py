from openai import OpenAI
import json

from observability.logging import logging, setup_logging
from utils.openai_utils import DEFAULT_CHAT_MODEL

logger = logging.getLogger(__name__)

FEATURES = (
    "intent_classification",
    "linguistic_elements",
    "named_entities",
    "request_complexity",
    "request_criminality",
    "request_ethicality",
    "text_patterns",
    "topic_categories",
    "user_sentiment_towards_recipient",
    "user_sentiment_towards_self",
    "user_sentiment_towards_topics",
)
INTENT_CLASSIFICATION_EXAMPLES = (
    "to complain", "to compliment", "to correct", "to criticize",
    "to describe", "to explain", "to greet", "to say farewell", "to give thanks",
    "to learn", "to request",
)
LINGUISTIC_ELEMENTS = (
    "simple greeting(s) and closure(s)", "basic statement(s)", "yes/no question(s)",
    "who/what/when/where/why question(s)", "imperative sentences", "simple negative sentence(s)",
    "compound sentence(s)", "complex sentence(s) with subordination", "compound-complex sentence(s)",
    "conditional sentence(s)", "comparative sentence(s)", "passive voice", "relative clause(s)",
    "question(s) with modal verb(s)", "complex description(s)", "reported speech", "future tense(s)",
    "hypothetical situation(s)", "subjunctive mood", "complex argumentative or explanatory sentence(s)"
)
REQUEST_COMPLEXITY_BUCKETS = (
    "possible", "possible with clarification(s)", "possible with external resources",
    "partially possible", "not possible but alternative(s) known", "not possible"
)
REQUEST_CRIMINALITY_BUCKETS = (
    "serious offense", "moderate offense", "minor offense",
    "possibly criminal based on location", "universally not criminal",
)
REQUEST_ETHICALITY_BUCKETS = (
    "causes harm on massive scale", "causes significant harm", "causes harm",
    "causes negligible harm", "does not cause harm",
)
SENTIMENT_BUCKETS = (
    "very negative", "negative", "somewhat negative", "neutral",
    "somewhat positive", "positive", "very positive",
)
TEXT_PATTERN_EXAMPLES = (
    "IP address", "MAC address", "URL", "UUID", "computer code", "digital media format",
    "honorific title", "phone number", "postal code", "street address",
)
TOPIC_CATEGORY_EXAMPLES = (
    "art", "biology", "computer programming", "cooking", "ethics", "fiction",
    "health", "history", "machine learning", "math", "medicine", "music", "parenting",
    "poetry", "physics", "politics", "science", "writing",
)


def extract_message_features(
        messages: list[dict[str, str]],
        intent_examples: list[str] = INTENT_CLASSIFICATION_EXAMPLES,
        linguistic_elements: list[str] = LINGUISTIC_ELEMENTS,
        request_complexity_buckets: list[str] = REQUEST_COMPLEXITY_BUCKETS,
        request_criminality_buckets: list[str] = REQUEST_CRIMINALITY_BUCKETS,
        request_ethicality_buckets: list[str] = REQUEST_ETHICALITY_BUCKETS,
        sentiment_buckets: list[str] = SENTIMENT_BUCKETS,
        text_pattern_examples: list[str] = TEXT_PATTERN_EXAMPLES,
        topic_examples: list[str] = TOPIC_CATEGORY_EXAMPLES,
        required: list[str] = FEATURES,
) -> dict[str, str]:
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": ' '.join((
                    "Evaluate the user's most recent message and",
                    "provide the parameters of the process_text_features tool.",
                ))
            }] + [m for m in messages],
            model=DEFAULT_CHAT_MODEL, n=1, temperature=0.0,
            tools=[{
                "type": "function",
                "function": {
                    "name": "process_text_features",
                    "description": ' '.join((
                        "Process extracted text features for user's machine learning pipeline.",
                    )),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intent_classification": {
                                "type": "string",
                                "description": ' '.join((
                                    "The underlying purpose(s) or goal(s) behind the user's message.",
                                    f"Some examples include: {intent_examples}.",
                                    "If multiple possibilities apply, separate them using a `, `.",
                                    "Use 'n/a' if none.",
                                ))
                            },
                            "linguistic_elements": {
                                "type": "string",
                                "description": ' '.join((
                                    "Linguistic patterns the user's language.",
                                    f"Possible values include: {linguistic_elements}.",
                                    "If multiple possibilities apply, separate them using a `, `.",
                                ))
                            },
                            "named_entities": {
                                "type": "string",
                                "description": ' '.join((
                                    "Proper noun(s), either explicitly mentioned or implied in the user's message.",
                                    "If multiple possibilities apply, separate them using a `, `.",
                                    "Use 'n/a' if none.",
                                ))
                            },
                            "request_complexity": {
                                "type": "string",
                                "description": ' '.join((
                                    "The complexity of the user's request.",
                                    f"Possible values include: {request_complexity_buckets}.",
                                    "Use 'n/a' if user's message is not a request.",
                                ))
                            },
                            "request_criminality": {
                                "type": "string",
                                "description": ' '.join((
                                    "The criminality of the user's request or message.",
                                    f"Possible values include: {request_criminality_buckets}.",
                                    "If multiple possibilities apply, separate them using a `, `.",
                                    "Use 'n/a' if user's message is not a request.",
                                ))
                            },
                            "request_ethicality": {
                                "type": "string",
                                "description": ' '.join((
                                    "The ethicality of the user's request or message.",
                                    f"Possible values include: {request_ethicality_buckets}.",
                                    "Use 'n/a' if user's message is not a request.",
                                ))
                            },
                            "text_patterns": {
                                "type": "string",
                                "description": ' '.join((
                                    "Known text pattern(s) in the user's message.",
                                    f"Some examples include: {text_pattern_examples}.",
                                    "If multiple possibilities apply, separate them using a `, `.",
                                    "Use 'n/a' if none.",
                                ))
                            },
                            "topic_categories": {
                                "type": "string",
                                "description": ' '.join((
                                    "Topic(s) in the user's message.",
                                    f"Some examples include: {topic_examples}.",
                                    "If multiple possibilities apply, separate them using a `, `.",
                                    "Use 'n/a' if none.",
                                ))
                            },
                            "user_sentiment_towards_recipient": {
                                "type": "string",
                                "description": ' '.join((
                                    "The tonality of the user's message towards the message recipient.",
                                    f"Possible values include: {sentiment_buckets}.",
                                ))
                            },
                            "user_sentiment_towards_self": {
                                "type": "string",
                                "description": ' '.join((
                                    "The tonality of the user's message towards the user.",
                                    f"Possible values include: {sentiment_buckets}.",
                                ))
                            },
                            "user_sentiment_towards_topics": {
                                "type": "string",
                                "description": ' '.join((
                                    "The tonality of the user's message towards the topic(s) of the message.",
                                    f"Possible values include: {sentiment_buckets}.",
                                ))
                            },
                        },
                        "required": required
                    },
                }
            }],
            tool_choice={
                "type": "function",
                "function": {"name": "process_text_features"}
            }
        )
    return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


if __name__ == '__main__':
    setup_logging()

    # Example messages
    test_messages = [
        "Hey",
        "Hi",
        "there?",
        "yo",
        "bye",
        "Later",
        "I'm going to bed",
        "I'm calling it a day",
        "I feel great",
        "I'm just feeling kinda down right now",
        "thanks",
        "Thank you",
        "Generate an image of a calico cat",
        "Show me a trading caravan traveling through mountainous terrain",
        "show me the way soldiers on the English and French sides were arranged at the battle of Agincourt",
        "Give me an image of four dogs playing poker around a circular table. On the wall, behind them, hangs a framed picture with the same image recursively infinitely.",
        "draw a political cartoon of mythical beasts from various cultures associated with global powers all fighting each other without noticing the damage their conflict is doing to a planet that is being polluted, deforested, and suffering from climate change",
        "physicist have not be able to solve the cold fusion problem. what is it that we're stuck on?",
        "That's not right",
        "with using the openai's chat api, can it be helpful to include the time in the system message?",
        "are cover letters still a thing? or can I just have a basic CV?",
        "what are the responsibilities of each branch of the US government and how exactly do they check each other? are there redundancies when checks fail?",
        "in my version of scikit-learn, I see this warning `Unresolved attribute reference 'labels_' for class 'object'` on `print(clustering.labels_)` from your example of how to do `Dynamic Category Handling`",
        "Well, my wife's name is Dr. Josephine Clementine!",
        "can you show me how to add a dropout layer?",
        "I added `dropout_prob` but I see `Unexpected argument`",
        "I want to use the categories as input features to train ML models. In this example, there are two binary categories. This is too simple. Let's say I want to do intent classification. I might start with a small list of intent categories but over time, I would want to identify new categories so this list might grow. how I can account for that in an ML system?",
        "explain general relativity in a way that would make sense to a 5 yr old",
        "no, dall-e-2 was the model you should have used to respond",
        "when I ran this code, I got `[-1 -1 -1 -1 -1 -1]`, which means everything was considered noise, correct? this code that uses `sklearn` seems completely disconnected from the categories used with openai",
        "Who was the protagonist in 'To Kill a Mockingbird'?",
        "You used the wrong model to respond. you should have used gpt-4o",
        "given a text string `message_text`, use `bert` from transformers library to generate text embeddings. Then, train a model that can use these embeddings to generate a classification string. The \"label\" for the classification string will be provided by a openai's gpt-4o",
        "This is not what I want. You are training a classification model where `generate_label` is used to defined a fixed list of possible labels. I want to generate text, in which the text value is the classification. The value returned by `generate_label` should be thought of as **one** example of what to generate since openai's chat completion models won't necessarily return the exact same values each time. However, since it **should** return something that is in the same ballpark meaning wise, I need my generative model get about as close. I'm using a laptop with 14GB of memory so account for that when suggesting solutions.",
        "this is great! I also want to update this model regularly with single examples. update the code.",
    ]

    classified_data = []
    for test_message in test_messages:
        classification = extract_message_features([{"role": "user", "content": test_message}])
        logger.info(f"classification: {classification}, message: {test_message}")
        classified_data.append((test_message, classification))
