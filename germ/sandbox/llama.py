import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
#model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the model and pinned to MPS:
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # This will map “layers” to mps:0
    torch_dtype=torch.bfloat16,  # LoRA adapters (if any) will be in bfloat16
    trust_remote_code=True,
).eval()

class StopOnAnswer(StoppingCriteria):
    def __init__(self, stop_str: str, tokenizer):
        self.stop_ids = tokenizer(stop_str, add_special_tokens=False)["input_ids"]
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[-1] < len(self.stop_ids):
            return False
        # Check if last N tokens match stop_ids
        return torch.equal(input_ids[0, -len(self.stop_ids) :], torch.tensor(self.stop_ids).to(input_ids.device))


if __name__ == "__main__":

    tag_pattern = re.compile(r"<\|[a-z_]+\|>")
    system_message = ("You are a helpful reasoning assistant. "
                      "Wrap each reasoning step in <think></think> tags. "
                      "Take as many steps as needed to get to the right answer. "
                      "Put the final answer at the end between <answer></answer> tags.")
    example = ("What is 1 + 1?\n"
               "<think>one and another one is two</think>\n"
               "<answer>2</answer>")
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}
Example:
{example}
<|eot_id|><|start_header_id|>user<|end_header_id|>
What is 234 + 456?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    outputs = model.generate(
        **encoded,
        do_sample=True,
        max_new_tokens=128,
        return_dict_in_generate=True,
        #stopping_criteria=StoppingCriteriaList([StopOnAnswer("</answer>", tokenizer)]),
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    print(outputs)
    decoded_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    len2trim = len(tag_pattern.subn("", prompt)[0])
    print(decoded_text[len2trim:])
    #parts = decoded_text.split(prompt)
    #if len(parts) > 1:
    #    generated_text = parts[1]
    #else:
    #    generated_text = ""
    #print(f"Model output: {generated_text}")

    #messages = [
    #    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    #    {"role": "user", "content": "Who are you?"},
    #]
    #outputs = pipe(
    #    messages,
    #    max_new_tokens=128,
    #)
    #print(outputs[0]["generated_text"][-1])