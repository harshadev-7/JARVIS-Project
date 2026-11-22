import argparse
import torch
from pathlib import Path

# Import inference symbols from the core module (avoid package name clash)
from .. import inference_core as inference_mod
LocalModelInference = inference_mod.LocalModelInference

LocalModelInference = inference_mod.LocalModelInference
from ..tokenizer import TextTokenizer
from ..checkpoint_utils import load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Run inference with a saved J.A.R.V.I.S. checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved .pth checkpoint.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to generate from.")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=0, help="Top‑k sampling (0 = disabled).")
    args = parser.parse_args()

    # Load checkpoint components
    model, optimizer, tokenizer, epoch = load_checkpoint(args.checkpoint, device="cpu")

    inference = LocalModelInference(model=model, tokenizer=tokenizer, device="cpu")
    generated = inference.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("Generated text:")
    print(generated)

if __name__ == "__main__":
    main()
