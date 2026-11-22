"""J.A.R.V.I.S. Autoregressive Text Generation Engine.

Generates text token-by-token using the custom decoder-only Transformer.
Features:
- Temperature scaling for controlling randomness/creativity
- Top-k filtering for nucleus sampling stability
- Repetition penalty to discourage repetitive loops
- Early termination on <eos> token
- Clean string decoding via TextTokenizer
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F

from .sampling import sample_next_token


class TextGenerator:
    """Autoregressive text generation for J.A.R.V.I.S. models."""

    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 20,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Autoregressively generate text from a string prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate beyond prompt
            temperature: Sampling temperature (lower = more deterministic, higher = more random)
            top_k: Number of highest-probability tokens to sample from
            top_p: Nucleus cumulative probability threshold (0.0 to 1.0)
            repetition_penalty: Multiplier penalty (>1.0) applied to previously generated tokens

        Returns:
            Generated text string
        """
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt, add_bos=True)
        if not prompt_tokens:
            prompt_tokens = [self.tokenizer.bos_token_id]

        generated = prompt_tokens[:]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Context window cropping to fit model context length
                context = generated[-self.model.max_seq_len:]
                input_tensor = torch.tensor(
                    [context], dtype=torch.long, device=self.device
                )

                # Forward pass: shape (1, T, vocab_size)
                logits = self.model(input_tensor)

                # Extract logits for the final time step
                next_token_logits = logits[0, -1, :]

                # Sample next token using composed logit modulations
                next_token = sample_next_token(
                    logits=next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    history=generated,
                )

                generated.append(next_token)

                # Stop if end-of-sequence token is generated
                if next_token == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(generated)


__all__ = ["TextGenerator"]
