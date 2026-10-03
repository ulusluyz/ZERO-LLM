import argparse
import os
import torch

from config import get_tiny_config, get_100m_config, get_300m_config, get_500m_config, get_1b_config
from model import ZEROModel
from tokenizer import Tokenizer
from checkpoint import load_checkpoint
from inference import generate_tokens
from utils import get_device


def main():
    parser = argparse.ArgumentParser(description="ZERO LLM - Token Generation CLI")
    parser.add_argument("--prompt", type=str, default="ZERO LLM", help="Input prompt string")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "100m", "300m", "500m", "1b"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint file")
    parser.add_argument("--tokenizer_file", type=str, default=None, help="Path to tokenizer JSON file")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = get_device()

    # Load Tokenizer
    if args.tokenizer_file and os.path.exists(args.tokenizer_file):
        tokenizer = Tokenizer.load(args.tokenizer_file)
    else:
        tokenizer = Tokenizer()
        tokenizer.train(args.prompt, vocab_size=max(256 + len(tokenizer.special_tokens), 300))

    # Preset Config
    presets = {
        "tiny": get_tiny_config,
        "100m": get_100m_config,
        "300m": get_300m_config,
        "500m": get_500m_config,
        "1b": get_1b_config,
    }
    config = presets[args.preset]()
    config.vocab_size = tokenizer.vocab_size

    # Instantiate Model
    model = ZEROModel(config).to(device)

    # Load Checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=str(device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("Running on randomly initialized (untrained) ZERO LLM core.")

    prompt_ids = tokenizer.encode(args.prompt)
    output_ids = generate_tokens(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    generated_text = tokenizer.decode(output_ids)
    print("\n--- Generation Result ---")
    print(f"Prompt: '{args.prompt}'")
    print(f"Generated Output: '{generated_text}'")
    print("-------------------------\n")


if __name__ == "__main__":
    main()
