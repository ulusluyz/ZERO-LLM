import argparse
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import (
    ModelConfig,
    get_tiny_config,
    get_100m_config,
    get_300m_config,
    get_500m_config,
    get_1b_config,
)
from model import ZEROModel
from tokenizer import Tokenizer
from dataset import TextDataset
from checkpoint import save_checkpoint, load_checkpoint
from utils import set_seed, get_device, get_gpu_memory_info, get_model_summary


def train():
    parser = argparse.ArgumentParser(description="ZERO LLM - Training Engine")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "100m", "300m", "500m", "1b"])
    parser.add_argument("--corpus_file", type=str, default=None, help="Path to text corpus file")
    parser.add_argument("--tokenizer_file", type=str, default=None, help="Path to trained tokenizer JSON")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Select configuration
    presets = {
        "tiny": get_tiny_config,
        "100m": get_100m_config,
        "300m": get_300m_config,
        "500m": get_500m_config,
        "1b": get_1b_config,
    }
    config = presets[args.preset]()

    # Prepare tokenizer & data
    if args.corpus_file and os.path.exists(args.corpus_file):
        with open(args.corpus_file, "r", encoding="utf-8") as f:
            corpus_text = f.read()
    else:
        corpus_text = (
            "ZERO LLM sıfırdan boş dil modeli çekirdeğidir. "
            "Model başlangıçta herhangi bir dil bilgisine sahip değildir. "
            "Türkçe harfler ç Ç ğ Ğ ı İ ö Ö ş Ş ü Ü tam ve eksiksiz desteklenir. "
        ) * 50

    if args.tokenizer_file and os.path.exists(args.tokenizer_file):
        tokenizer = Tokenizer.load(args.tokenizer_file)
    else:
        tokenizer = Tokenizer()
        tokenizer.train(corpus_text, vocab_size=config.vocab_size)

    config.vocab_size = tokenizer.vocab_size
    token_ids = tokenizer.encode(corpus_text)

    # Dataset & DataLoader
    dataset = TextDataset(token_ids, seq_len=min(config.max_seq_len, 64))
    if len(dataset) > 1:
        val_size = max(1, int(len(dataset) * 0.1))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, dataset

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Instantiate ZEROModel
    model = ZEROModel(config).to(device)
    summary = get_model_summary(model)
    print("\n--- Model Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("---------------------\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")

    start_epoch = 0
    global_step = 0

    if args.resume_from and os.path.exists(args.resume_from):
        ckpt_meta = load_checkpoint(args.resume_from, model, optimizer, scheduler, device=str(device))
        global_step = ckpt_meta.get("step", 0)
        print(f"Resumed from step {global_step}")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        total_tokens = 0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            batch_tokens = x.numel()

            with torch.amp.autocast(device_type=device.type, enabled=args.mixed_precision and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accum_steps
            total_tokens += batch_tokens

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_logits = model(x_val)
                v_loss = criterion(val_logits.view(-1, val_logits.size(-1)), y_val.view(-1))
                val_loss += v_loss.item()
        avg_val_loss = val_loss / len(val_loader)

        gpu_mem = get_gpu_memory_info()
        print(
            f"Epoch {epoch+1}/{args.epochs} | Step {global_step} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Tokens/sec: {tokens_per_sec:.2f} | GPU Mem: {gpu_mem['allocated_mb']:.1f}MB"
        )

        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{global_step}.pt")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, step=global_step)
        print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    train()
