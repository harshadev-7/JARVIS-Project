import os
import subprocess
import sys
import tempfile

def test_cli_training_creates_checkpoint():
    # Use a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare arguments: use example corpus, small model config, output dir
        corpus_path = os.path.abspath(os.path.join('examples', 'hello.txt'))
        checkpoint_dir = tmpdir
        cmd = [sys.executable, '-m', 'custom_ai.training.run_training',
               '--corpus', corpus_path,
               '--epochs', '1',
               '--batch_size', '2',
               '--output_dir', checkpoint_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Ensure the script completed successfully
        assert result.returncode == 0, f"Training script failed: {result.stderr}"
        # Look for checkpoint file (we expect a .pt or .pth file)
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
        assert ckpt_files, "No checkpoint file was created"

if __name__ == "__main__":
    test_cli_training_creates_checkpoint()
