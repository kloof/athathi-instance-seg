"""Point cloud segmentation models: PointNet, DGCNN, RandLA-Net."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Find k nearest neighbors for each point using pairwise distances.

    Args:
        x: (B, C, N) point features.
        k: number of neighbors.

    Returns:
        (B, N, k) indices of k nearest neighbors.
    """
    # Pairwise distance: (B, N, N)
    inner = -2 * torch.bmm(x.transpose(2, 1), x)          # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)            # (B, 1, N)
    dist = -xx - inner - xx.transpose(2, 1)                # (B, N, N) negative distances
    _, idx = dist.topk(k=k, dim=-1)                        # (B, N, k) largest = nearest
    return idx


def get_edge_features(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Build edge features: concatenate [xi, xj - xi] for each neighbor.

    Args:
        x: (B, C, N) point features.
        idx: (B, N, k) neighbor indices.

    Returns:
        (B, 2*C, N, k) edge features.
    """
    B, C, N = x.shape
    k = idx.shape[2]

    # Gather neighbor features
    idx_flat = idx.reshape(B, -1)                                      # (B, N*k)
    neighbors = torch.gather(
        x, 2, idx_flat.unsqueeze(1).expand(-1, C, -1)
    ).reshape(B, C, N, k)                                              # (B, C, N, k)

    # Center point features expanded
    center = x.unsqueeze(-1).expand(-1, -1, -1, k)                    # (B, C, N, k)

    # Edge feature: [center, neighbor - center]
    edge = torch.cat([center, neighbors - center], dim=1)              # (B, 2*C, N, k)
    return edge


class EdgeConv(nn.Module):
    """Single EdgeConv layer: KNN -> edge features -> MLP -> max pool over neighbors."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) point features.
        Returns:
            (B, out_channels, N) updated features.
        """
        idx = knn(x, self.k)                    # (B, N, k)
        edge = get_edge_features(x, idx)        # (B, 2*C, N, k)
        out = self.conv(edge)                   # (B, out_channels, N, k)
        out = out.max(dim=-1)[0]                # (B, out_channels, N)
        return out


class DGCNNSegmentation(nn.Module):
    """DGCNN for per-point segmentation.

    Architecture:
        4 EdgeConv layers (64 -> 64 -> 128 -> 256) with dynamic KNN
        Concatenate all layer outputs -> 512
        Global max pool -> 512
        Per-point: concat [local 512 + global 512] -> 1024
        Decoder MLP: 1024 -> 512 -> 256 -> num_classes

    Input:  (B, input_dim, N) -- channels first
    Output: (B, num_classes, N) -- per-point logits
    """

    def __init__(self, input_dim: int = 6, num_classes: int = 2, k: int = 20):
        super().__init__()
        self.k = k

        self.ec1 = EdgeConv(input_dim, 64, k)
        self.ec2 = EdgeConv(64, 64, k)
        self.ec3 = EdgeConv(64, 128, k)
        self.ec4 = EdgeConv(128, 256, k)

        # Fuse all EdgeConv outputs
        self.fuse = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder: local (512) + global (512) -> num_classes
        self.decoder = nn.Sequential(
            nn.Conv1d(1024, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim, N) point features.
        Returns:
            (B, num_classes, N) per-point logits.
        """
        B, _, N = x.shape

        x1 = self.ec1(x)       # (B, 64, N)
        x2 = self.ec2(x1)      # (B, 64, N)
        x3 = self.ec3(x2)      # (B, 128, N)
        x4 = self.ec4(x3)      # (B, 256, N)

        # Concatenate all layer outputs
        local_features = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 512, N)
        local_features = self.fuse(local_features)              # (B, 512, N)

        # Global feature
        global_features = local_features.max(dim=2, keepdim=True)[0]  # (B, 512, 1)
        global_features = global_features.expand(-1, -1, N)            # (B, 512, N)

        # Combine local + global
        combined = torch.cat([local_features, global_features], dim=1)  # (B, 1024, N)

        logits = self.decoder(combined)  # (B, num_classes, N)
        return logits


class PointNetSegmentation(nn.Module):
    """PointNet segmentation network (kept for comparison).

    Input:  (B, input_dim, N) -- channels first
    Output: (B, num_classes, N) -- per-point logits
    """

    def __init__(self, input_dim: int = 6, num_classes: int = 2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.global_encoder = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, N = x.shape
        local_features = self.encoder(x)
        global_features = self.global_encoder(local_features)
        global_features = torch.max(global_features, dim=2, keepdim=True)[0]
        global_features = global_features.expand(-1, -1, N)
        combined = torch.cat([local_features, global_features], dim=1)
        logits = self.decoder(combined)
        return logits


# ============================================================
# RandLA-Net — Random sampling + Local Feature Aggregation
# ============================================================
# Reference: "RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds"
# Key insight: uses random sampling (O(1)) instead of FPS (O(N²)),
# compensated by Local Feature Aggregation with attentive pooling.
# All ops are Conv1d + gather — no KNN distance matrix, ONNX-friendly.


class SharedMLP(nn.Module):
    """Conv1d + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class AttentivePooling(nn.Module):
    """Attentive pooling: learn attention scores over K neighbors, then weighted sum."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, 1),  # attention score per neighbor
        )
        self.mlp = SharedMLP(in_ch, out_ch)

    def forward(self, x):
        """x: (B, C, N, K) -> (B, out_ch, N)"""
        scores = self.score_fn(x)               # (B, 1, N, K)
        scores = F.softmax(scores, dim=-1)      # normalize over K
        pooled = (x * scores).sum(dim=-1)       # (B, C, N)
        return self.mlp(pooled)                  # (B, out_ch, N)


class LocalFeatureAggregation(nn.Module):
    """LFA module: gather K-nearest features, apply attentive pooling twice, residual."""

    def __init__(self, in_ch: int, out_ch: int, K: int = 16):
        super().__init__()
        self.K = K
        self.mlp_pre = SharedMLP(in_ch, out_ch)
        self.att_pool1 = AttentivePooling(out_ch, out_ch)
        self.att_pool2 = AttentivePooling(out_ch, out_ch)
        self.mlp_post = SharedMLP(out_ch, out_ch)
        self.shortcut = SharedMLP(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def _gather_neighbors(self, x, idx):
        """Gather K neighbor features. x: (B,C,N), idx: (B,N,K) -> (B,C,N,K)"""
        B, C, N = x.shape
        K = idx.shape[2]
        idx_flat = idx.reshape(B, -1)  # (B, N*K)
        gathered = torch.gather(
            x, 2, idx_flat.unsqueeze(1).expand(-1, C, -1)
        )  # (B, C, N*K)
        return gathered.reshape(B, C, N, K)

    def _random_knn(self, x):
        """Approximate KNN via random candidate sampling. O(N*K) not O(N²).
        x: (B, C, N) -> idx: (B, N, K)"""
        B, C, N = x.shape
        K = self.K
        # Sample K*4 random candidates, pick K nearest
        n_cand = min(K * 4, N)
        cand_idx = torch.randint(N, (B, N, n_cand), device=x.device)

        # Gather candidate features
        x_t = x.permute(0, 2, 1)  # (B, N, C)
        src = x_t.unsqueeze(2).expand(-1, -1, n_cand, -1)  # (B, N, n_cand, C)
        cand_pts = torch.gather(
            x_t.unsqueeze(1).expand(-1, N, -1, -1),
            2,
            cand_idx.unsqueeze(3).expand(-1, -1, -1, C),
        )  # (B, N, n_cand, C)

        # Distances
        dists = torch.norm(src - cand_pts, dim=3)  # (B, N, n_cand)

        # Top-K nearest from candidates
        _, topk = dists.topk(K, dim=2, largest=False)  # (B, N, K)
        idx = torch.gather(cand_idx, 2, topk)  # (B, N, K)
        return idx

    def forward(self, x):
        """x: (B, C, N) -> (B, out_ch, N)"""
        shortcut = self.shortcut(x)
        x = self.mlp_pre(x)

        # Approximate KNN
        idx = self._random_knn(x)

        # First attentive pooling
        neighbors = self._gather_neighbors(x, idx)  # (B, C, N, K)
        x = self.att_pool1(neighbors)

        # Second attentive pooling (re-gather with same idx)
        neighbors = self._gather_neighbors(x, idx)
        x = self.att_pool2(neighbors)

        x = self.mlp_post(x)
        return F.leaky_relu(x + shortcut, 0.2)


class RandLANetSegmentation(nn.Module):
    """RandLA-Net for per-point segmentation.

    Encoder: 4 LFA layers with random downsampling (N -> N/4 -> N/16 -> N/64 -> N/256)
    Decoder: 4 upsample + skip connection layers

    All ops are Conv1d + gather — ONNX exportable, runs on CPU/RPi4.

    Input:  (B, input_dim, N) -- channels first
    Output: (B, num_classes, N) -- per-point logits
    """

    def __init__(self, input_dim: int = 10, num_classes: int = 2, K: int = 16):
        super().__init__()
        # Encoder dims: input -> 32 -> 64 -> 128 -> 256
        self.enc0 = SharedMLP(input_dim, 32)
        self.lfa1 = LocalFeatureAggregation(32, 64, K)
        self.lfa2 = LocalFeatureAggregation(64, 128, K)
        self.lfa3 = LocalFeatureAggregation(128, 256, K)

        # Decoder with skip connections
        self.dec3 = SharedMLP(256 + 128, 128)
        self.dec2 = SharedMLP(128 + 64, 64)
        self.dec1 = SharedMLP(64 + 32, 32)

        self.classifier = nn.Sequential(
            SharedMLP(32, 32),
            nn.Dropout(0.3),
            nn.Conv1d(32, num_classes, 1),
        )

    def _random_downsample(self, x, ratio=4):
        """Randomly downsample points. x: (B, C, N) -> (B, C, N//ratio), idx"""
        B, C, N = x.shape
        n_keep = max(N // ratio, 1)
        idx = torch.stack([torch.randperm(N, device=x.device)[:n_keep] for _ in range(B)])
        idx_sorted, _ = idx.sort(dim=1)
        down = torch.gather(x, 2, idx_sorted.unsqueeze(1).expand(-1, C, -1))
        return down, idx_sorted

    def _upsample(self, x_low, x_high):
        """Nearest-neighbor upsample: repeat low-res features to high-res count.
        x_low: (B, C, N_low), x_high: (B, C_skip, N_high) -> (B, C+C_skip, N_high)
        """
        B, C, N_low = x_low.shape
        N_high = x_high.shape[2]
        # Simple: repeat each low-res point to fill high-res
        # For random downsampling, nearest-neighbor interp via index repeat
        idx = torch.arange(N_high, device=x_low.device).unsqueeze(0).expand(B, -1)
        idx = idx * N_low // N_high  # approximate mapping
        idx = idx.clamp(0, N_low - 1)
        up = torch.gather(x_low, 2, idx.unsqueeze(1).expand(-1, C, -1))
        return torch.cat([up, x_high], dim=1)

    def forward(self, x):
        B, _, N = x.shape

        # Encoder
        e0 = self.enc0(x)                              # (B, 32, N)

        e1 = self.lfa1(e0)                             # (B, 64, N)
        d1, _ = self._random_downsample(e1, 4)         # (B, 64, N/4)

        e2 = self.lfa2(d1)                             # (B, 128, N/4)
        d2, _ = self._random_downsample(e2, 4)         # (B, 128, N/16)

        e3 = self.lfa3(d2)                             # (B, 256, N/16)

        # Decoder
        u2 = self._upsample(e3, e2)                    # (B, 256+128, N/4)
        u2 = self.dec3(u2)                              # (B, 128, N/4)

        u1 = self._upsample(u2, e1)                    # (B, 128+64, N)
        u1 = self.dec2(u1)                              # (B, 64, N)

        u0 = torch.cat([u1, e0], dim=1)                # (B, 64+32, N)
        u0 = self.dec1(u0)                              # (B, 32, N)

        logits = self.classifier(u0)                    # (B, num_classes, N)
        return logits
