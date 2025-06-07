import copy
import numpy as np
import random
import re
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from dataclasses import dataclass
from datasets import Dataset
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, get_scheduler, logging as hf_logging
from typing import List, Any

hf_logging.set_verbosity_error()

# Hyper-Parameters
MODEL_ID          = "meta-llama/Llama-3.2-1B-Instruct"
LR                = 2e-5
BATCH_PROMPTS     = 8  # prompts per step
GROUP_SIZE        = 4  # K samples per prompt (32 forward passes / step)
EPOCHS            = 2
#MAX_STEPS         = 10_000  # Stop early for demo
MAX_STEPS         = 50
MAX_NEW_TOKENS    = 128
KL_COEFF          = 0.02
CLIP_EPS          = 0.2
ACCUM_STEPS       = 1
WARMUP_STEPS      = 500
SEED              = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

accelerator = Accelerator(
    gradient_accumulation_steps=ACCUM_STEPS,
    kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
)
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)
old_model = copy.deepcopy(model).eval()  # start with identical weights
old_model.to(device)  # stay on same GPU / dtype
for p in old_model.parameters():  # freeze
    p.requires_grad_(False)

model_ref = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)
model_ref.eval()  # frozen for KL penalty

# Dummy math dataset
def gen_addition_dataset(n: int = 8_000):
    qs, ans = [], []
    for _ in range(n):
        a, b = random.randint(2, 999), random.randint(2, 999)
        qs.append(f"What is {a} + {b}?")
        ans.append(str(a + b))
    return Dataset.from_dict({"question": qs, "answer": ans})
dataset = gen_addition_dataset()

# Simple prompt template
EXAMPLE = ("What is 1 + 1?\n"
           "<think>one and another one is two</think>\n"
           "<answer>2</answer>")
SYS_PROMPT = ("You are a helpful reasoning assistant. "
              "Wrap reasoning steps in <think></think> tags. "
              "Take as many steps as needed to get to the right answer. "
              "Put the answer between <answer></answer> tags.")
SYS_TAG_PATTERN = re.compile(r"<\|[a-z_]+\|>")
def format_prompt(row):
    return {
        # See: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md
        "prompt": ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                   f"{SYS_PROMPT}\n"
                   f"Example:\n{EXAMPLE}\n"
                   "<|eot_id|><|start_header_id|>user<|end_header_id|>"
                   f"{row['question']}\n"
                   "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"),
        "answer": row["answer"],
    }
dataset = dataset.map(format_prompt, remove_columns=["question"])

# Reward Functions
def has_think_tags(text: str) -> bool:
    checks = []
    lines = text.split("\n")
    for line in lines[:-1]:
        checks.append(line.startswith("<think>") and line.endswith("</think>"))
    return lines[-1].startswith("<answer>") and lines[-1].endswith("</answer>") and False not in checks

def extract_answer(text: str) -> str:
    lines = text.split("\n")
    if lines[-1].startswith("<answer>") and lines[-1].endswith("</answer>"):
        return lines[-1].split("<answer>")[-1].split("</answer>")[0].strip()
    # No credit for wrong format
    return ""

def compute_rewards(samples: List[str], gt_answer: str) -> torch.Tensor:
    scores = []
    for s in samples:
        acc   = 1.0 if extract_answer(s) == gt_answer else 0.0
        fmt   = 0.1 if has_think_tags(s) else 0.0
        scores.append(acc + fmt)
    return torch.tensor(scores, device=device)

# DataLoader
@dataclass
class PromptCollator:
    tokenizer: Any
    max_len: int = 512

    def __call__(self, batch):
        answers = [ex["answer"] for ex in batch]
        prompts = [ex["prompt"] for ex in batch]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        encoded["answers"] = answers
        encoded["prompts"] = prompts
        return encoded

loader = DataLoader(
    dataset.shuffle(seed=SEED),
    batch_size=BATCH_PROMPTS,
    collate_fn=PromptCollator(tokenizer),
)

# Optimiser
opt = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = get_scheduler(
    "linear",
    opt,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=MAX_STEPS,
)

model, model_ref, opt, loader, scheduler = accelerator.prepare(
    model, model_ref, opt, loader, scheduler
)

# GRPO Loop
gen_cfg = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    top_p=0.95,
    top_k=50,
    temperature=0.7,
)

def batched_generate(prompts_enc):
    """
    Prompts_enc: tokenizer-encoded batch of size B
    Returns:
        samples_text  : List[B][G]  text strings
        samples_ids   : Tensor [B, G, L_total_max]  (padded on the right with pad_token_id)
    """
    B = prompts_enc["input_ids"].size(0)
    all_ids = []  # to collect raw s_ids per group iteration → each [B, L_total_i]
    all_text = []  # to collect decoded text per group iteration → each is List[str] of length B

    with torch.no_grad():
        for _ in range(GROUP_SIZE):
            # 1) Generate a batch of sequences under the current policy
            outs = model.generate(
                **{k: v for k, v in prompts_enc.items() if k not in {"answers", "prompts"}},
                generation_config=gen_cfg,
                return_dict_in_generate=True,
            )
            s_ids = outs.sequences  # shape: [B, L_total_i]  ← varies per iteration

            # 2) Store everything
            all_ids.append(s_ids)  # [B, L_total_i]
            all_text.append([
                t[len(SYS_TAG_PATTERN.subn("", prompts_enc["prompts"][t_idx])[0]):]
                for t_idx, t in enumerate(
                    tokenizer.batch_decode(s_ids, skip_special_tokens=True)  # List[str] * B
                )
            ])

    # Now: all_ids = [ [B, L_total_0], [B, L_total_1], …, [B, L_total_{G-1}] ]
    # We need to pad each list‐element to a common shape before stacking.

    # 3) Pad `all_ids` (raw sequences) to a common `L_total_max`
    #    so that torch.stack(all_padded_ids, dim=1) → shape [B, G, L_total_max].
    max_total_len = max(tensor.size(1) for tensor in all_ids)  # find largest “[B, L_total_i]” → take max over i
    padded_ids = []
    for s_ids in all_ids:
        L_i = s_ids.size(1)
        if L_i < max_total_len:
            pad_len = max_total_len - L_i
            # Create a pad tensor shaped [B, pad_len], filled with pad_token_id
            pad_tensor = s_ids.new_full((B, pad_len), tokenizer.pad_token_id)
            s_ids = torch.cat([s_ids, pad_tensor], dim=1)  # now [B, max_total_len]
        padded_ids.append(s_ids)
    # Stack into: [B, G, max_total_len]
    samples_ids = torch.stack(padded_ids, dim=1)

    B = len(all_text[0])  # batch size
    G = len(all_text)  # group size (should be GROUP_SIZE)
    samples_text: List[List[str]] = [
        [all_text[g][b] for g in range(G)]
        for b in range(B)
    ]
    return samples_text, samples_ids

global_step = 0
model.train()
for epoch in range(EPOCHS):
    for batch in loader:
        with accelerator.accumulate(model):
            # 1. Sample G outputs per prompt under no_grad, get token IDs & text
            samples_text, samples_ids = batched_generate(batch)
            gt_answers = batch["answers"]  # list length = B

            # 2. Reward
            rewards = []
            for i, row in enumerate(samples_text):
                rewards.append(compute_rewards(row, gt_answers[i]))
            rewards = torch.stack(rewards, dim=0)  # [B, G]

            # 3. GRPO advantage: group-wise z-score
            mu  = rewards.mean(dim=1, keepdim=True)  # [B, 1]
            std = rewards.std(dim=1, keepdim=True) + 1e-8
            advantage = (rewards - mu) / std  # [B, G]

            # 4. Recompute log‐probs of generated tokens under current model (with grad), _including_ prompt context
            B, G, L_total = samples_ids.shape
            prompt_len = batch["input_ids"].size(-1)

            # Flatten full sequences [prompt_ids + generated_ids]
            flat_full = rearrange(samples_ids, "b g l -> (b g) l")  # shape: [(B*G), L_total]
            # Run the model on the entire sequence so that logits are conditioned on the prompt
            out_full = model(input_ids=flat_full)  # shape: [(B*G), L_total, V]
            logits_full = out_full.logits[..., :-1, :]  # shift for next‐token: [(B*G), L_total-1, V]

            # The target IDs for next‐token prediction
            tgt_full = flat_full[..., 1:]  # shape: [(B*G), L_total-1]

            # Compute log‐probabilities for every token (conditional on all previous tokens)
            logp_full = nn.functional.log_softmax(logits_full, dim=-1)  # [(B*G), L_total-1, V]
            # Pick out the log‐prob of the _actual_ token at each position
            flat_logp_all = logp_full.gather(dim=-1, index=tgt_full.unsqueeze(-1)).squeeze(-1)  # [(B*G), L_total-1]

            # Now mask out everything _before_ the prompt. We only want the sum over generated tokens.
            # Create a boolean mask of shape [(B*G), L_total-1], where positions < prompt_len are False.
            # Because for prompt indices 0..prompt_len-1, the next‐token predictions do not count.
            device = flat_logp_all.device
            total_next_tokens = flat_logp_all.size(-1)  # == L_total-1
            # For each flat index i, positions 0..(prompt_len-1 - 1) correspond to prompt tokens for prediction.
            # We want to zero those out.
            mask = torch.zeros_like(flat_logp_all, dtype=torch.bool)  # same shape
            # For each sample in the flattened batch, mark indices ≥ (prompt_len) as True.
            mask[:, prompt_len:] = True

            # Zero out the prompt-part log-probs; sum over only the generated tokens
            flat_logp_gen = flat_logp_all.masked_select(mask)  # this flattens out
            # But masked_select flattens across the entire matrix; easier to reshape and then sum:
            flat_logp_all_reshaped = flat_logp_all.view(B, G, total_next_tokens)
            # Sum from index [prompt_len .. end] along dim=-1
            logp_policy = flat_logp_all_reshaped[:, :, prompt_len:].sum(dim=-1)  # shape: [B, G]

            # 5. KL term (we can still do this under no_grad for model_ref)
            with torch.no_grad():
                # Repeat the same procedure for the reference model (include prompt context)
                out_ref_full = model_ref(input_ids=flat_full, use_cache=False)
                logits_ref_full = out_ref_full.logits[..., :-1, :]
                tgt_ref = flat_full[..., 1:]
                logp_ref_full = nn.functional.log_softmax(logits_ref_full, dim=-1)
                flat_logp_ref_all = logp_ref_full.gather(dim=-1, index=tgt_ref.unsqueeze(-1)).squeeze(-1)
                # Mask out everything before prompt_len
                flat_logp_ref_all_reshaped = flat_logp_ref_all.view(B, G, total_next_tokens)
                ref_logp = flat_logp_ref_all_reshaped[:, :, prompt_len:].sum(dim=-1)  # shape [B, G]

            # 6. PPO‐style clipped policy loss + KL regularization
            with torch.no_grad():
                # Same: compute log‐probs under old_model, including prompt context
                out_old_full = old_model(input_ids=flat_full, use_cache=False)
                logits_old_full = out_old_full.logits[..., :-1, :]
                logp_old_full = nn.functional.log_softmax(logits_old_full, dim=-1)
                flat_logp_old_all = logp_old_full.gather(dim=-1, index=tgt_ref.unsqueeze(-1)).squeeze(-1)
                flat_logp_old_all_reshaped = flat_logp_old_all.view(B, G, total_next_tokens)
                logp_old = flat_logp_old_all_reshaped[:, :, prompt_len:].sum(dim=-1)  # [B, G]

            ratio = (logp_policy - logp_old).exp()  # π_new / π_old
            unclipped = ratio * advantage  # PPO(clip) loss
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage
            policy_loss = -torch.min(unclipped, clipped).mean()

            # 7. E_{x ~ π_new}[logπ_new(x) – logπ_ref(x)] ≈ mean over (logp_policy – ref_logp)
            kl = (logp_policy - ref_logp).mean()
            loss = policy_loss + KL_COEFF * kl

            # 8. Backward, step, zero_grad, scheduler
            accelerator.backward(loss)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # Move current params into old_model so it lags by one update
            with torch.no_grad():
                for p_old, p_new in zip(old_model.parameters(), model.parameters()):
                    p_old.data.copy_(p_new.data)
            scheduler.step()
            opt.zero_grad()
            global_step += 1

        #if accelerator.is_main_process and global_step % 10 == 0:
        if accelerator.is_main_process:
            print(
                f"step {global_step:>6} | "
                f"loss {loss.item():.4f} | "
                f"R_mean {rewards.mean().item():.3f} | "
                f"KL {kl.item():.3f}"
            )
        if global_step >= MAX_STEPS:
            break
    if global_step >= MAX_STEPS:
        break

if accelerator.is_main_process:
    model.save_pretrained("data/llama3b_grpo_rl")
    tokenizer.save_pretrained("data/llama3b_grpo_rl")
    print("\nFinished RL fine-tuning -> saved to data/llama3b_grpo_rl")