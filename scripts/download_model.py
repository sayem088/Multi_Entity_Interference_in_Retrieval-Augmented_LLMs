#!/usr/bin/env python3
"""
Utility script to download Qwen2.5-7B-Instruct model weights directly into the
local project folder `models/Qwen2.5-7B-Instruct`.

Once downloaded, you can copy the entire `mei-research/` folder to a USB drive or
drive storage and take it to the University GPU desktop to run offline.
"""

import sys
from pathlib import Path

def download_model():
    target_dir = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-7B-Instruct"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"===========================================================")
    print(f"Resuming download of Qwen2.5-7B-Instruct...")
    print(f"Destination: {target_dir}")
    print(f"===========================================================")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\n[Error] huggingface_hub is not installed.")
        print("Please run: pip install huggingface_hub\n")
        sys.exit(1)

    # Use max_workers=2 to prevent memory allocation crashes on home devices
    snapshot_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        max_workers=2
    )

    print("\n[Success] Model download complete!")
    print(f"Model saved to: {target_dir}")
    print("You can now copy the entire `mei-research` folder to your University Desktop.")

if __name__ == "__main__":
    download_model()
