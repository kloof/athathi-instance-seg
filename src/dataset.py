"""PyTorch Dataset — chunk-sequential loading with intra-chunk shuffling."""

import numpy as np
import torch
from torch.utils.data import IterableDataset
from pathlib import Path


class WallSegDataset(IterableDataset):
    """Streams chunks sequentially. Shuffles blocks within each chunk.

    This avoids the random-access cache thrashing problem while still
    providing randomization. Chunks are loaded one at a time (~156 MB each).

    Uses IterableDataset so DataLoader doesn't need shuffle=True.
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_points: int = 4096,
        block_size: float = 1.0,
        augment: bool = False,
        augment_cfg: dict | None = None,
        min_points: int = 100,
        shuffle_chunks: bool = True,
    ):
        data_dir = Path(data_dir)
        blocks_dir = data_dir / "blocks"

        if not blocks_dir.exists():
            raise FileNotFoundError(f"No blocks/ directory in {data_dir}")

        self.chunk_files = sorted(blocks_dir.glob("blocks_*.npz"))
        if not self.chunk_files:
            raise FileNotFoundError(f"No block chunk files in {blocks_dir}")

        self.shuffle_chunks = shuffle_chunks

        # Count total blocks for __len__ (used by DataLoader for progress)
        self.total_blocks = 0
        for cf in self.chunk_files:
            with np.load(cf) as data:
                self.total_blocks += data["points"].shape[0]

        print(f"  {self.total_blocks:,} blocks in {len(self.chunk_files)} chunks")

    def __len__(self) -> int:
        return self.total_blocks

    def __iter__(self):
        chunk_order = list(range(len(self.chunk_files)))
        if self.shuffle_chunks:
            np.random.shuffle(chunk_order)

        for ci in chunk_order:
            data = np.load(self.chunk_files[ci])
            points = data["points"]      # (N, 4096, 3) float32
            labels = data["labels"]      # (N, 4096) int64
            room_mins = data["room_mins"]
            room_maxs = data["room_maxs"]
            n = points.shape[0]

            # Shuffle blocks within this chunk
            order = np.random.permutation(n) if self.shuffle_chunks else np.arange(n)

            for i in order:
                yield (
                    torch.from_numpy(points[i].astype(np.float16).copy()),
                    torch.from_numpy(labels[i].astype(np.int8).copy()),
                    torch.from_numpy(room_mins[i].astype(np.float16).copy()),
                    torch.from_numpy(room_maxs[i].astype(np.float16).copy()),
                )
