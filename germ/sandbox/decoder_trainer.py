import math
import numpy as np
import random
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
BATCH_SIZE      = 16  # Per step (no grad-accum)
WINDOW_SIZE     = 200  # Reporting only
LOSS_SCALE      = 64.0  # Static loss-scaling factor, lack of GradScaler support on MPS
ADAM_EPS        = 6.1e-5  # Minimum normal fp16 number
CLIP_NORM       = 1.0
LR              = 3e-4
EPOCHS          = 3
WARMUP_FRAC     = 0.1
N_BUCKETS       = 100
SEED            = 42
DEVICE          = torch.device("mps")
DTYPE           = torch.float16  # On M-series
CHECKPOINT_DIR  = "./data/decoder"

# Tokenizer
TOK = GPT2TokenizerFast.from_pretrained("gpt2")
TOK.pad_token = TOK.eos_token  # GPT-2 has no PAD by default


def collate(batch, pad_id=TOK.pad_token_id):
    seqs = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    #Lmax = max(len(s) for s in seqs)
    Lmax = next_power_of_2(max(len(s) for s in seqs))
    batch_sz = len(seqs)

    inp  = torch.full((batch_sz, Lmax), pad_id, dtype=torch.long)
    tgt  = torch.full_like(inp, pad_id)

    for i, s in enumerate(seqs):
        n = s.numel()
        inp[i, :n] = s
        tgt[i, :n-1] = s[1:]          # next-token prediction
    return inp, tgt


def dataloader_worker_init(worker_id):
    seed = SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def encode(example):
    ids = TOK(example["text"],
              add_special_tokens=False,
              truncation=True,
              max_length=MAX_BLOCK_LEN)["input_ids"]
    return {"input_ids": ids, "len": len(ids)}


def next_power_of_2(n: int) -> int:
    """
    Return the next power-of-2 ≥ n, min 8, max MAX_BLOCK_LEN.

    Needed due to:
    - MPS Memory Leak: https://github.com/pytorch/pytorch/issues/155060
    - MPS SDPA memory leak: https://github.com/pytorch/pytorch/issues/152344

    TL;DR:
    - Each distinct (B, H, L, D) shape that hits F.scaled_dot_product_attention compiles a new Metal graph and keeps
      the compiled objects alive.
    - Bucketed loader sees hundreds of different L values (1 … 1024), so the leak grows until the high-water-mark
      (≈ 80 % of unified memory) is hit and PyTorch aborts.
    - Padding to the nearest power-of-two results in, at most, 10 distinct lengths so leak stays below 1 GB
    """
    n = max(n, 8)
    return 1 << ((n - 1).bit_length())


def prepare_dataset(dataset):
    dataset = dataset.map(encode, remove_columns=["text"], num_proc=8)
    # Avoids the following assertion error when dataset has empty sequences after tokenization:
    #   RuntimeError: [srcBuf length] > 0 INTERNAL ASSERT FAILED at
    #   "/Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":565,
    #   please report a bug to PyTorch. Placeholder tensor is empty!
    dataset = dataset.filter(lambda x: x["len"] > 0, num_proc=8)
    return dataset


class Trainer:
    def __init__(self, loss_scale = LOSS_SCALE):
        self.bad_batches = 0
        self.loss_scale = loss_scale
        self.window_loss = 0.0
        self.window_toks = 0

    def run_epoch(self, loader, train=True):
        self.window_loss = self.window_toks = 0.0  # Reset every epoch
        if train:
            model.train()
        else:
            model.eval()

        tot_loss, n_tokens = 0., 0
        t0 = tw0 = time.time()
        for step, (inp, tgt) in enumerate(loader):
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, dtype=DTYPE):
                logits = model(inp)
                base_loss = loss_fn(
                    logits.view(-1, TOK.vocab_size).to(torch.float32),  # Compute loss in fp32
                    tgt.view(-1)
                )

            if train:
                # Scale only when we will back-prop
                scaled_loss = base_loss * self.loss_scale
                scaled_loss.backward()

                # Check for Inf/NaN in loss or any gradient
                has_inf = not torch.isfinite(scaled_loss).item()
                if not has_inf:
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            has_inf = True
                            break

                # If bad batch, lower scale and skip the update
                if has_inf:
                    self.bad_batches += 1
                    self.loss_scale = max(1.0, self.loss_scale / 2)
                    optimizer.zero_grad(set_to_none=True)
                    sched.step()
                    continue  # skip this mini-batch

                # If good batch, un-scale gradients, clip, and step
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(self.loss_scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                sched.step()

            n_tok = (tgt != TOK.pad_token_id).sum().item()
            tot_loss += base_loss.item() * n_tok
            n_tokens += n_tok

            self.window_loss += base_loss.item() * n_tok
            self.window_toks += n_tok

            if train and step % WINDOW_SIZE == 0 and step > 0:
                avg_loss = self.window_loss / self.window_toks
                print(f"[window {step - WINDOW_SIZE:5d} >> {step:5d}] "
                      f"loss={avg_loss:.4f}  "
                      f"ppl={math.exp(avg_loss):.2f}  "
                      f"scale={self.loss_scale}  "
                      f"bad={self.bad_batches}  "
                      f"dt_window={time.time()-tw0:.1f}s  "
                      f"dt_total={time.time() - t0:.1f}s")
                self.window_loss = self.window_toks = 0
                tw0 = time.time()

        ppl = math.exp(tot_loss / n_tokens)
        print(f"[{'epoch' if train else 'val'}] "
              f"ppl={ppl:.2f}  "
              f"tokens={n_tokens / 1e6:.2f}M  "
              f"time={time.time() - t0:.1f}s")
        return ppl


if __name__ == "__main__":
    import os

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")

    # Prevent CPU over-subscription for data loader
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr, BLIS, etc.
    os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP (fallback)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Apple Accelerate

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    ds = prepare_dataset(ds)

    val_ds = load_dataset("roneneldan/TinyStories", split="validation")
    val_ds = prepare_dataset(val_ds)

    lengths = ds["len"]
    sampler = BucketBatchSampler(lengths, batch_size=BATCH_SIZE,
                                 n_buckets=N_BUCKETS, shuffle=True)

    train_loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=4,
        persistent_workers=True,
        pin_memory=False,
        worker_init_fn=dataloader_worker_init,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        collate_fn=collate,
        num_workers=2,
        persistent_workers=True,
        shuffle=False,
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
        eps=ADAM_EPS,
        fused=False,
        amsgrad=True,
    )
    num_steps = EPOCHS * len(train_loader)
    sched = get_cosine_schedule_with_warmup(optimizer,
                                            int(num_steps * WARMUP_FRAC),
                                            num_steps)

    loss_fn = nn.CrossEntropyLoss(ignore_index=TOK.pad_token_id)

    trainer = Trainer()
    best_val = float("inf")
    for ep in range(1, EPOCHS+1):
        print(f"--- epoch {ep}/{EPOCHS} ---")
        trainer.run_epoch(train_loader, train=True)
        val_ppl = trainer.run_epoch(val_loader, train=False)

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
