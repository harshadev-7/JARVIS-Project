"""Compatibility wrapper for the moved inference CLI.

The original CLI entry point lived at ``custom_ai.inference.run_inference``.
After renaming the package to ``inference_cli`` we keep this thin wrapper so
``python -m custom_ai.inference.run_inference`` continues to work.
"""

from ..inference_cli.run_inference import main as _main

if __name__ == "__main__":
    _main()
