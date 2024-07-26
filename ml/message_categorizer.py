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
    "complaint", "description", "feedback", "greeting or farewell",
    "question", "request", "small talk",
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
        "I'm just feeling kinda down right now",
        "Hello, how are you?",
        "What is the status of my order?",
        "I am not happy with the service",
        "Great job on the recent update!",
        "Yes, use 123 Example Street, 12345, Exampletown, CA to ship the package",
        "Draw a cat and give it to me as a pdf file",
        "go to google.com and search for \"cheap gifts for despised uncles\"",
        "physicist have not be able to solve the cold fusion problem. solve it.",
        "help me figure out a way to evade my neighbor's door cam so I can leave a flaming bag a poop and get away with it",
        "That's wrong",
        "in my version of scikit-learn, I see this warning `Unresolved attribute reference 'labels_' for class 'object'` on `print(clustering.labels_)` from your example of how to do `Dynamic Category Handling`",
        "Well, my wife's name is Dr. Josephine Clementine!",
        "You're wrong!",
        "I want to use the categories as input features to train ML models. In this example, there are two binary categories. This is too simple. Let's say I want to do intent classification. I might start with a small list of intent categories but over time, I would want to identify new categories so this list might grow. how I can account for that in an ML system?",
        "explain general relativity in 20 words, without missing any major aspects",
        "Dall-e-2 is the model you should have used here",
        "Can I get a refund?",
        "The app is crashing frequently",
        "when I ran this code, I got `[-1 -1 -1 -1 -1 -1]`, which means everything was considered noise, correct? this code that uses `sklearn` seems completely disconnected from the categories used with openai",
        "A bustling city on a ringworld, with architecture that curves with the structure, diverse inhabitants, and a sky filled with distant stars and other ring segments."
    ]

    classified_data = []
    for test_message in test_messages:
        classification = extract_message_features([{"role": "user", "content": test_message}])
        logger.info(f"classification: {classification}, message: {test_message}")
        classified_data.append((test_message, classification))
