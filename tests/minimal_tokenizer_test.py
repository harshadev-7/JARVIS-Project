import sys
import os

# Ensure project root is on PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from custom_ai.tokenizer import Tokenizer

def test_tokenizer_roundtrip():
    # tiny corpus
    corpus = ["hello world", "test sentence"]
    tokenizer = Tokenizer()
    tokenizer.fit(corpus)
    text = "hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text, f"Round‑trip failed: {decoded!r} != {text!r}"

if __name__ == "__main__":
    test_tokenizer_roundtrip()
    print("minimal tokenizer test passed")
