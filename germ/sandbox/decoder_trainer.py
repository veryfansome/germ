import math
import time
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, get_cosine_schedule_with_warmup

from germ.sandbox.decoder_v2 import DecoderModel, init_weights
from germ.sandbox.sampler import BucketBatchSampler

# Config
MAX_BLOCK_LEN   = 1024  # ≤ 4096
BATCH_SIZE      = 16  # per step (no grad-accum)
LOSS_SCALE      = 512.0  # static loss-scaling factor, lack of GradScaler support on MPS
LR              = 3e-4
EPOCHS          = 3
WARMUP_FRAC     = 0.1
N_BUCKETS       = 60
SEED            = 42
DEVICE          = torch.device("mps")
DTYPE           = torch.float16  # On M-series
CHECKPOINT_DIR  = "./data/decoder"

# Tokenizer
TOK = GPT2TokenizerFast.from_pretrained("gpt2")
TOK.pad_token = TOK.eos_token  # GPT-2 has no PAD by default


def collate(batch, pad_id=TOK.pad_token_id):
    seqs = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    Lmax = max(len(s) for s in seqs)
    batch_sz = len(seqs)

    inp  = torch.full((batch_sz, Lmax), pad_id, dtype=torch.long)
    tgt  = torch.full_like(inp, pad_id)

    for i, s in enumerate(seqs):
        n = s.numel()
        inp[i, :n] = s
        tgt[i, :n-1] = s[1:]          # next-token prediction
    return inp, tgt


def encode(example):
    ids = TOK(example["text"],
              add_special_tokens=False,
              truncation=True,
              max_length=MAX_BLOCK_LEN)["input_ids"]
    return {"input_ids": ids, "len": len(ids)}


def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    tot_loss, n_tokens = 0., 0
    t0 = time.time()
    for step, (inp, tgt) in enumerate(loader):
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=DTYPE):
            logits = model(inp)
            base_loss = loss_fn(
                logits.view(-1, TOK.vocab_size).float(),  # cast → fp32 for numerical safety
                tgt.view(-1)
            )

        if train:
            # Scale only when we will back-prop
            scaled_loss = base_loss * LOSS_SCALE
            scaled_loss.backward()
            for p in model.parameters():  # Unscale the gradients in-place
                if p.grad is not None:
                    p.grad.div_(LOSS_SCALE)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # optional
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sched.step()

        n_tok = (tgt != TOK.pad_token_id).sum().item()
        tot_loss += base_loss.item() * n_tok
        n_tokens += n_tok

        if train and step % 100 == 0:
            ppl = math.exp(tot_loss / n_tokens)
            print(f"step {step:5d}  loss={base_loss.item():.4f}  ppl={ppl:.2f}  time={time.time() - t0:.1f}s")

    ppl = math.exp(tot_loss / n_tokens)
    print(
        f"\n[{'train' if train else 'valid'}] ppl={ppl:.2f}  tokens={n_tokens / 1e6:.2f}M  time={time.time() - t0:.1f}s")
    return ppl


if __name__ == "__main__":
    import os

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    ds = ds.map(encode, remove_columns=["text"], num_proc=8)

    val_ds = load_dataset("roneneldan/TinyStories", split="validation")
    val_ds = val_ds.map(encode, remove_columns=["text"], num_proc=8)

    lengths = ds["len"]
    sampler = BucketBatchSampler(lengths, batch_size=BATCH_SIZE,
                                 n_buckets=N_BUCKETS, shuffle=True)

    train_loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collate,
        pin_memory=False,
        persistent_workers=True,
        num_workers=1,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        persistent_workers=True,
        num_workers=1,
    )

    model = DecoderModel(
        vocab_size=TOK.vocab_size,
        max_seq_len=MAX_BLOCK_LEN,
        pad_token_id=TOK.pad_token_id,
    ).to(DEVICE, DTYPE)
    model.apply(lambda m: init_weights(m) if isinstance(m, (nn.Linear,)) else None)  # if not already called

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-6,  # 1e-8 under-flows in fp16
        fused=False,
    )
    num_steps = EPOCHS * len(train_loader)
    sched = get_cosine_schedule_with_warmup(optimizer,
                                            int(num_steps * WARMUP_FRAC),
                                            num_steps)

    loss_fn = nn.CrossEntropyLoss(ignore_index=TOK.pad_token_id)

    best_val = float("inf")
    for ep in range(1, EPOCHS+1):
        print(f"\n--- epoch {ep}/{EPOCHS} ---")
        run_epoch(train_loader, train=True)
        val_ppl = run_epoch(val_loader, train=False)

        if val_ppl < best_val:
            best_val = val_ppl
            ckpt = f"decoder_ep{ep}_ppl{val_ppl:.2f}.pt"
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
            torch.save({
                "model":   model.state_dict(),
                "optim":   optimizer.state_dict(),
                "sched":   sched.state_dict(),
                "epoch":   ep,
                "val_ppl": val_ppl,
            }, ckpt_path)
            print(f"✓ saved checkpoint to {ckpt_path}")

    print("done.")
