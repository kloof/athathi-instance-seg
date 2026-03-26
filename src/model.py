"""Point cloud segmentation models: PointNet, DGCNN, RandLA-Net, PTv2-lite."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


_USE_TORCH_CLUSTER = True  # Set False for ONNX export

try:
    from torch_cluster import knn as _tc_knn
except ImportError:
    _tc_knn = None

# Cache for batch index tensors: (B, N, device) -> tensor
_batch_cache: dict[tuple, torch.Tensor] = {}
_offset_cache: dict[tuple, torch.Tensor] = {}


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Find k nearest neighbors. Uses torch_cluster when available (CUDA fast),
    falls back to brute-force for ONNX export or CPU.

    Args:
        x: (B, C, N) point features.
        k: number of neighbors.

    Returns:
        (B, N, k) indices of k nearest neighbors.
    """
    B, C, N = x.shape
    if _USE_TORCH_CLUSTER and _tc_knn is not None and not torch.jit.is_tracing():
        x_flat = x.permute(0, 2, 1).reshape(B * N, C)

        # Cache batch/offset tensors — same shape reuses the same tensor
        cache_key = (B, N, x.device)
        if cache_key not in _batch_cache:
            _batch_cache[cache_key] = torch.arange(B, device=x.device).repeat_interleave(N)
            _offset_cache[cache_key] = (torch.arange(B, device=x.device) * N).unsqueeze(1).unsqueeze(2)
        batch = _batch_cache[cache_key]
        offsets = _offset_cache[cache_key]

        edge_index = _tc_knn(x_flat, x_flat, k, batch, batch)
        idx = edge_index[0].reshape(B, N, k)
        idx = idx - offsets
        return idx

    # Brute-force fallback (ONNX export / CPU / no torch_cluster)
    # torch.cdist is fused and avoids materializing the full N×N intermediate
    pts = x.transpose(2, 1)  # (B, N, C)
    dists = torch.cdist(pts, pts)  # (B, N, N)
    _, idx = dists.topk(k=k, dim=-1, largest=False)
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
        """KNN using torch_cluster (CUDA optimized). x: (B, C, N) -> idx: (B, N, K)"""
        return knn(x, self.K)

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
        """Downsample points. Uses stride for ONNX compat, random during training."""
        B, C, N = x.shape
        n_keep = max(N // ratio, 1)
        if self.training:
            idx = torch.stack([torch.randperm(N, device=x.device)[:n_keep] for _ in range(B)])
            idx_sorted, _ = idx.sort(dim=1)
        else:
            # Fixed stride — ONNX exportable
            idx_sorted = torch.arange(0, N, ratio, device=x.device)[:n_keep].unsqueeze(0).expand(B, -1)
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


# ============================================================
# PTv2-lite — Point Transformer V2 inspired, dense batched ops
# ============================================================
# Grouped Vector Attention on (B, C, N) tensors.
# No pointops/torch_geometric/torch_scatter needed.
# ONNX exportable. Uses our existing knn() for neighbor queries.


class GroupedVectorAttention(nn.Module):
    """Grouped Vector Attention from Point Transformer V2.

    For each point, attends to K neighbors using per-group attention weights.
    Middle ground between scalar attention (weak) and full vector attention (expensive).
    """

    def __init__(self, channels: int, num_groups: int = 8, K: int = 16):
        super().__init__()
        self.channels = channels
        self.num_groups = num_groups
        self.group_dim = channels // num_groups
        self.K = K

        self.proj_q = nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True))
        self.proj_k = nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True))
        self.proj_v = nn.Conv1d(channels, channels, 1, bias=False)

        # Position encoding: relative xyz -> channels
        self.pe_mlp = nn.Sequential(
            nn.Conv2d(3, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

        # Weight encoding: channels -> num_groups (attention weights per group)
        self.weight_enc = nn.Sequential(
            nn.Conv2d(channels, num_groups, 1, bias=False), nn.BatchNorm2d(num_groups), nn.ReLU(True),
            nn.Conv2d(num_groups, num_groups, 1, bias=False),
        )

        self.proj_out = nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels))

    def _fast_knn(self, xyz, K):
        """KNN using torch_cluster (CUDA optimized). xyz: (B, 3, N) -> idx: (B, N, K)"""
        return knn(xyz, K)

    def forward(self, x, xyz):
        """x: (B, C, N), xyz: (B, 3, N) -> (B, C, N)"""
        B, C, N = x.shape
        G = self.num_groups
        D = self.group_dim
        K = min(self.K, N)

        # Fast approximate KNN on xyz
        knn_idx = self._fast_knn(xyz, K)  # (B, N, K)

        # Project Q, K, V
        q = self.proj_q(x)  # (B, C, N)
        k = self.proj_k(x)  # (B, C, N)
        v = self.proj_v(x)  # (B, C, N)

        # Gather K neighbor keys and values
        idx_flat = knn_idx.reshape(B, -1)  # (B, N*K)
        k_neighbors = torch.gather(k, 2, idx_flat.unsqueeze(1).expand(-1, C, -1)).reshape(B, C, N, K)
        v_neighbors = torch.gather(v, 2, idx_flat.unsqueeze(1).expand(-1, C, -1)).reshape(B, C, N, K)

        # Relation: key_neighbors - query
        relation = k_neighbors - q.unsqueeze(3)  # (B, C, N, K)

        # Position encoding from relative xyz
        xyz_neighbors = torch.gather(xyz, 2, idx_flat.unsqueeze(1).expand(-1, 3, -1)).reshape(B, 3, N, K)
        rel_pos = xyz_neighbors - xyz.unsqueeze(3)  # (B, 3, N, K)
        pe = self.pe_mlp(rel_pos)  # (B, C, N, K)

        # Add position bias to relation and value
        relation = relation + pe
        v_neighbors = v_neighbors + pe

        # Weight encoding: (B, C, N, K) -> (B, G, N, K) -> softmax over K
        weights = self.weight_enc(relation)  # (B, G, N, K)
        weights = F.softmax(weights, dim=-1)  # (B, G, N, K)

        # Grouped aggregation
        # v_neighbors: (B, C, N, K) -> (B, G, D, N, K)
        v_grouped = v_neighbors.reshape(B, G, D, N, K)
        # weights: (B, G, N, K) -> (B, G, 1, N, K)
        w = weights.unsqueeze(2)
        # Weighted sum over K: (B, G, D, N)
        out = (v_grouped * w).sum(dim=-1)  # (B, G, D, N)
        out = out.reshape(B, C, N)

        return self.proj_out(out)


class PTv2Block(nn.Module):
    """Single PTv2 encoder block: GVA + FFN with residual."""

    def __init__(self, channels: int, num_groups: int = 8, K: int = 16, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(channels)
        self.attn = GroupedVectorAttention(channels, num_groups, K)
        self.norm2 = nn.BatchNorm1d(channels)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels, 1, bias=False),
        )
        self.drop_path = drop_path

    def forward(self, x, xyz):
        # Pre-norm + attention + residual
        shortcut = x
        x = self.norm1(x)
        x = shortcut + self.attn(x, xyz)

        # Pre-norm + FFN + residual
        shortcut = x
        x = self.norm2(x)
        x = shortcut + self.ffn(x)
        return x


class PTv2GridPool(nn.Module):
    """Grid-based pooling: downsample by grouping points into grid cells."""

    def __init__(self, in_ch: int, out_ch: int, ratio: int = 4):
        super().__init__()
        self.ratio = ratio
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x, xyz):
        """Downsample by random sampling (simple, ONNX-friendly).
        Returns (x_down, xyz_down, upsample_idx)."""
        B, C, N = x.shape
        x = self.proj(x)
        C_new = x.shape[1]
        n_keep = max(N // self.ratio, 1)

        # Random subsample
        idx = torch.stack([torch.randperm(N, device=x.device)[:n_keep] for _ in range(B)])
        idx_sorted, _ = idx.sort(dim=1)

        x_down = torch.gather(x, 2, idx_sorted.unsqueeze(1).expand(-1, C_new, -1))
        xyz_down = torch.gather(xyz, 2, idx_sorted.unsqueeze(1).expand(-1, 3, -1))

        return x_down, xyz_down, idx_sorted


class PTv2LiteSegmentation(nn.Module):
    """Point Transformer V2 — lightweight dense-batched variant.

    Grouped Vector Attention in a U-Net encoder-decoder.
    No sparse ops, no external dependencies, ONNX exportable.

    Input:  (B, input_dim, N) -- channels first
    Output: (B, num_classes, N) -- per-point logits
    """

    def __init__(self, input_dim: int = 10, num_classes: int = 2, K: int = 16):
        super().__init__()
        # Lightweight config: ~3-4M params
        enc_dims = [48, 96, 192, 256]
        enc_groups = [6, 12, 24, 32]
        enc_depths = [1, 1, 3, 1]
        dec_dims = [192, 96, 48]

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv1d(input_dim, enc_dims[0], 1, bias=False),
            nn.BatchNorm1d(enc_dims[0]),
            nn.ReLU(True),
        )

        # Encoder stages
        self.enc_blocks = nn.ModuleList()
        self.enc_pools = nn.ModuleList()
        for i in range(4):
            blocks = nn.ModuleList([
                PTv2Block(enc_dims[i], enc_groups[i], K) for _ in range(enc_depths[i])
            ])
            self.enc_blocks.append(blocks)
            if i < 3:  # no pool after last stage
                self.enc_pools.append(PTv2GridPool(enc_dims[i], enc_dims[i + 1], ratio=4))

        # Decoder stages (3 upsample steps)
        self.dec_projs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(3):
            in_ch = enc_dims[3 - i] + enc_dims[2 - i]  # skip + upsampled
            self.dec_projs.append(nn.Sequential(
                nn.Conv1d(in_ch, dec_dims[i], 1, bias=False),
                nn.BatchNorm1d(dec_dims[i]),
                nn.ReLU(True),
            ))
            self.dec_blocks.append(PTv2Block(dec_dims[i], dec_dims[i] // 8 or 1, K))

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv1d(dec_dims[-1], dec_dims[-1], 1, bias=False),
            nn.BatchNorm1d(dec_dims[-1]),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(dec_dims[-1], num_classes, 1),
        )

    def _upsample(self, x_low, N_high):
        """Nearest-neighbor upsample by repeating."""
        B, C, N_low = x_low.shape
        idx = torch.arange(N_high, device=x_low.device).unsqueeze(0).expand(B, -1)
        idx = (idx * N_low // N_high).clamp(0, N_low - 1)
        return torch.gather(x_low, 2, idx.unsqueeze(1).expand(-1, C, -1))

    def forward(self, x):
        B, _, N = x.shape
        xyz = x[:, :3, :]  # First 3 channels are local_xyz

        # Patch embed
        x = self.patch_embed(x)

        # Encoder with skip connections
        skips = []
        xyzs = [xyz]
        Ns = [N]

        for i in range(4):
            for block in self.enc_blocks[i]:
                x = block(x, xyzs[-1])
            skips.append(x)

            if i < 3:
                x, xyz_down, _ = self.enc_pools[i](x, xyzs[-1])
                xyzs.append(xyz_down)
                Ns.append(x.shape[2])

        # Decoder
        for i in range(3):
            skip_idx = 2 - i  # skips[2], skips[1], skips[0]
            x_up = self._upsample(x, Ns[skip_idx])
            x = torch.cat([x_up, skips[skip_idx]], dim=1)
            x = self.dec_projs[i](x)
            x = self.dec_blocks[i](x, xyzs[skip_idx])

        logits = self.seg_head(x)
        return logits
