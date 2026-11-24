import argparse

from config import ModelConfig, get_tiny_config, get_100m_config, get_300m_config, get_500m_config, get_1b_config
from model import ZEROModel
from tokenizer import Tokenizer
from checkpoint import read_checkpoint
from inference import generate_tokens
from utils import get_device


def main():
    parser = argparse.ArgumentParser(description="ZERO LLM - Token Generation CLI")
    parser.add_argument("--prompt", default="ZERO LLM")
    parser.add_argument("--preset", choices=["tiny", "100m", "300m", "500m", "1b"], default=None,
                        help="Architecture for untrained runs; checkpoint config takes precedence")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokenizer_file")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    device = get_device()
    supplied = Tokenizer.load(args.tokenizer_file) if args.tokenizer_file else None
    checkpoint = read_checkpoint(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None:
        if checkpoint.get("config") is None or checkpoint.get("tokenizer") is None:
            raise ValueError("Legacy checkpoint lacks config/tokenizer; create a new training bundle")
        config = ModelConfig(**checkpoint["config"])
        tokenizer = Tokenizer.from_dict(checkpoint["tokenizer"])
        if supplied is not None and supplied.fingerprint != tokenizer.fingerprint:
            raise ValueError("Supplied tokenizer does not match checkpoint")
    else:
        tokenizer = supplied or Tokenizer()
        presets = {"tiny": get_tiny_config, "100m": get_100m_config, "300m": get_300m_config,
                   "500m": get_500m_config, "1b": get_1b_config}
        config = presets[args.preset or "tiny"]()
        config.vocab_size = tokenizer.vocab_size
    model = ZEROModel(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("Running on randomly initialized (untrained) ZERO LLM core.")
    # BOS makes an empty prompt a valid request and matches the training format.
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(args.prompt)
    output_ids = generate_tokens(model, prompt_ids, args.max_new_tokens, args.temperature,
                                 args.top_k, args.top_p, eos_id=tokenizer.eos_id,
                                 device=device, use_cache=not args.no_cache)
    print("\n--- Generation Result ---")
    print(f"Prompt: {args.prompt!r}")
    print(f"Generated Output: {tokenizer.decode(output_ids[1:])!r}")


if __name__ == "__main__":
    main()
