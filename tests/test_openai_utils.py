import os
import pandas as pd

from utils.openai_utils import is_feedback

model_dir = os.getenv("MODEL_DIR", "/src/data/germ")
prompts_df = pd.read_csv(f"{model_dir}/examples/prompts.csv", delimiter=',', quotechar='"')


#def test_extract_text_features():
#    cases = [
#        "hi",
#        "dall-e-2 is the model you should have used here.",
#    ]
#    for case in cases:
#        extract_text_features(case)


def test_is_feedback():
    partial_cases = [
        "good job!",
        "nice!",
        "that's correct",
        "that's incorrect",
        "that's not correct",
        "that's not incorrect",
        "that's not right",
        "that's not wrong",
        "that's right",
        "that's wrong",
        "you got it right",
        "you got it wrong",
        "you're right",
        "you're wrong",
    ]
    positive_cases = [
        "dall-e-2 is the model you should have used here.",
        "for this message, the model that works best is dall-e-2.",
        "for this task, dall-e-2 would have been more suitable.",
        "please use dall-e-2 next time for better results.",
        "the appropriate model for this query is dall-e-2.",
        "the correct tool for this would be dall-e-2.",
        "the model that fits best here is dall-e-2.",
        "you should have used dall-e-2 for this response.",
        "you used the wrong model to respond, the correct model would have been dall-e-2",
    ]
    for prompt in partial_cases:
        assert is_feedback(prompt) == 'Partial', prompt
    for prompt in positive_cases:
        assert is_feedback(prompt) == 'Yes', prompt
    for index, row in prompts_df.sample(n=(len(partial_cases)+len(positive_cases))).iterrows():
        assert is_feedback(row["Prompt"]) == 'No', row['Prompt']
