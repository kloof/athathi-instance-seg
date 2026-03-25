#!/bin/bash
# Run this on the vast.ai instance after SSH-ing in.
# Usage: bash vastai_setup.sh

set -e

echo "=== Installing dependencies ==="
pip install torch numpy Pillow pyyaml tqdm onnx onnxruntime scipy huggingface_hub

echo "=== Downloading + extracting with pipeline ==="
export HF_TOKEN="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"
cd /workspace

python3 -c "
from huggingface_hub import hf_hub_download
import tarfile, os, threading, time
from pathlib import Path

repo = 'Pointcept/structured3d-compressed'
dst = Path('/workspace/wall-seg/data/structured3d_pointcept')
dst.mkdir(parents=True, exist_ok=True)

# Two separate download dirs so they don't conflict
dl_dirs = [Path('/workspace/dl_a'), Path('/workspace/dl_b')]
for d in dl_dirs:
    d.mkdir(parents=True, exist_ok=True)

files = [f'structured3d_{i:02d}.tar.gz' for i in range(1, 16)]
downloaded = {}
lock = threading.Lock()

def download(name, dl_dir):
    t0 = time.time()
    local = hf_hub_download(repo, name, repo_type='dataset', local_dir=str(dl_dir))
    elapsed = time.time() - t0
    print(f'  DL  {name}: {elapsed:.0f}s', flush=True)
    with lock:
        downloaded[name] = local

def extract_and_delete(local, name):
    t0 = time.time()
    with tarfile.open(local, 'r:gz') as t:
        t.extractall(dst)
    os.remove(local)
    elapsed = time.time() - t0
    print(f'  EXT {name}: {elapsed:.0f}s (deleted tar)', flush=True)

print(f'Starting pipeline for {len(files)} files...', flush=True)

# Download first file
download(files[0], dl_dirs[0])

for i in range(len(files)):
    name = files[i]

    # Start downloading next in background using alternate dir
    dl_thread = None
    if i + 1 < len(files):
        next_name = files[i + 1]
        next_dir = dl_dirs[(i + 1) % 2]
        dl_thread = threading.Thread(target=download, args=(next_name, next_dir))
        dl_thread.start()

    # Extract current
    with lock:
        local = downloaded[name]
    extract_and_delete(local, name)

    if dl_thread:
        dl_thread.join()

print('\nAll done!', flush=True)

import shutil
for d in dl_dirs:
    shutil.rmtree(str(d), ignore_errors=True)
"

echo "=== Preprocessing into blocks ==="
cd /workspace/wall-seg
python scripts/preprocess_pointcept.py --workers $(nproc)

echo "=== Deleting raw data (keep only blocks) ==="
rm -rf /workspace/wall-seg/data/structured3d_pointcept
echo "Freed disk space"

echo "=== Ready to train! ==="
df -h /workspace
echo ""
echo "Run: python scripts/run_training.py --config config.yaml --model dgcnn --k 20"
echo "Or:  python scripts/run_training.py --config config_pointnet.yaml --model pointnet"
