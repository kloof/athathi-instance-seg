"""PyTorch Dataset for instance segmentation.

Each room has:
  - coord.npy (N, 3) — points
  - semantic.npy (N,) — class IDs [0-15]
  - instance.npy (N,) — instance IDs (0 = stuff/no instance)

Subsamples large rooms to max_points in the dataset (CPU side)
so the GPU never sees more than max_points per room.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class InstanceSegDataset(Dataset):
    def __init__(self, split_dir: str | Path, max_points: int = 100000):
        split_dir = Path(split_dir)
        self.max_points = max_points
        self.rooms = sorted([
            d for d in split_dir.iterdir()
            if d.is_dir() and (d / "instance.npy").exists()
        ])
        print(f"  InstanceSegDataset [{split_dir.name}]: {len(self.rooms)} rooms, max_points={max_points}")

    def __len__(self):
        return len(self.rooms)

    def __getitem__(self, idx):
        room_dir = self.rooms[idx]
        pts = np.load(str(room_dir / "coord.npy"))
        sem = np.load(str(room_dir / "semantic.npy"))
        inst = np.load(str(room_dir / "instance.npy"))

        # Subsample large rooms on CPU before anything touches GPU
        N = len(pts)
        if N > self.max_points:
            choice = np.random.choice(N, self.max_points, replace=False)
            pts = pts[choice]
            sem = sem[choice]
            inst = inst[choice]

        # Center XY
        center_xy = pts[:, :2].mean(axis=0)
        pts[:, 0] -= center_xy[0]
        pts[:, 1] -= center_xy[1]

        # Compute offset targets
        offsets = np.zeros((len(pts), 3), dtype=np.float32)
        for iid in np.unique(inst):
            if iid == 0:
                continue
            mask = inst == iid
            centroid = pts[mask].mean(axis=0)
            offsets[mask] = centroid - pts[mask]

        return {
            "points": torch.from_numpy(pts).float(),
            "sem_labels": torch.from_numpy(sem).long(),
            "inst_labels": torch.from_numpy(inst).long(),
            "offsets": torch.from_numpy(offsets).float(),
        }


def instance_collate(batch):
    """Pad points to max N in batch."""
    max_n = max(item["points"].shape[0] for item in batch)
    result = {"points": [], "sem_labels": [], "inst_labels": [], "offsets": []}

    for item in batch:
        n = item["points"].shape[0]
        if n < max_n:
            pad_idx = torch.randint(0, n, (max_n - n,))
            result["points"].append(torch.cat([item["points"], item["points"][pad_idx]]))
            result["sem_labels"].append(torch.cat([item["sem_labels"], item["sem_labels"][pad_idx]]))
            result["inst_labels"].append(torch.cat([item["inst_labels"], item["inst_labels"][pad_idx]]))
            result["offsets"].append(torch.cat([item["offsets"], item["offsets"][pad_idx]]))
        else:
            for k in result:
                result[k].append(item[k])

    return {k: torch.stack(v) for k, v in result.items()}
