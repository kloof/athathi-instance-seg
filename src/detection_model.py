"""LitePT-based instance segmentation model.

Implements the LitePT (CVPR 2026) design philosophy in pure PyTorch:
  - Conv-early (stages 0-2): cheap Conv1d blocks, no KNN, no attention
  - Attention-late (stages 3-4): serialized window attention + PointROPE
  - Grid pooling between stages

No spconv, no flash_attn, no torch_scatter required.
Uses F.scaled_dot_product_attention (PyTorch 2.0+) for efficient attention.

Input:  (B, input_dim, N)
Output: sem_logits (B, num_classes, N), offsets (B, 3, N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Z-Order Serialization (from LitePT)
# ============================================================

class ZOrderEncoder:
    """Encode 3D integer coordinates to Z-order (Morton) curve codes."""

    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        self._lut = {}
        # Precompute LUT on CPU
        self._lut_cpu = (
            self._xyz2key(r256, zero, zero, 8),
            self._xyz2key(zero, r256, zero, 8),
            self._xyz2key(zero, zero, r256, 8),
        )

    def _xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key |= ((x & mask) << (2 * i + 2)) | ((y & mask) << (2 * i + 1)) | ((z & mask) << (2 * i))
        return key

    def _get_lut(self, device):
        if device not in self._lut:
            self._lut[device] = tuple(e.to(device) for e in self._lut_cpu)
        return self._lut[device]

    @torch.no_grad()
    def encode(self, coords, depth=16):
        """Encode (N, 3) integer coords to (N,) Z-order codes."""
        EX, EY, EZ = self._get_lut(coords.device)
        x, y, z = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
        mask = 255
        key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
        if depth > 8:
            mask2 = (1 << (depth - 8)) - 1
            key16 = EX[(x >> 8) & mask2] | EY[(y >> 8) & mask2] | EZ[(z >> 8) & mask2]
            key = key16 << 24 | key
        return key


_z_encoder = ZOrderEncoder()


@torch.no_grad()
def serialize_points(xyz, grid_size=0.02):
    """Serialize 3D points into Z-order for windowed attention.

    Args:
        xyz: (B, 3, N) point coordinates
        grid_size: voxel size for quantization

    Returns:
        order: (B, N) indices that sort points by Z-order
        grid_coords: (B, N, 3) quantized integer coordinates
    """
    B, _, N = xyz.shape
    pts = xyz.permute(0, 2, 1)  # (B, N, 3)

    orders = []
    all_grid = []
    for b in range(B):
        coords = pts[b]  # (N, 3)
        # Quantize to grid
        grid = ((coords - coords.min(0).values) / grid_size).int()
        grid = grid.clamp(0, 65535)  # 16-bit max
        all_grid.append(grid)
        # Z-order encode
        codes = _z_encoder.encode(grid)
        order = codes.argsort()
        orders.append(order)

    return torch.stack(orders), torch.stack(all_grid)


# ============================================================
# PointROPE — Rotary Positional Encoding for 3D (from LitePT)
# ============================================================

class PointROPE(nn.Module):
    """Pure PyTorch PointROPE. Splits features into 3 axis subspaces,
    applies 1D RoPE per axis using point grid coordinates."""

    # Fixed max position to avoid unbounded cache growth
    MAX_POS = 4096

    def __init__(self, freq=100.0):
        super().__init__()
        self.base = freq
        self._cache = {}

    def _get_cos_sin(self, D, device, dtype):
        key = (D, device, dtype)
        if key not in self._cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2, device=device).float() / D))
            t = torch.arange(self.MAX_POS, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat([freqs, freqs], dim=-1)
            self._cache[key] = (freqs.cos(), freqs.sin())
        return self._cache[key]

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, tokens, positions):
        """
        Args:
            tokens: (B, H, N, D) where D must be divisible by 3
            positions: (B, N, 3) integer grid coordinates

        Returns:
            tokens with PointROPE applied
        """
        D = tokens.shape[-1]
        assert D % 3 == 0, f"PointROPE requires channels divisible by 3, got {D}"
        d = D // 3
        assert d % 2 == 0, f"PointROPE requires (channels // 3) to be even, got {d}"
        # Clamp positions to MAX_POS to avoid index overflow
        positions = positions.clamp(0, self.MAX_POS - 1)
        cos, sin = self._get_cos_sin(d, tokens.device, tokens.dtype)

        x, y, z = tokens.chunk(3, dim=-1)
        for chunk, axis in [(x, 0), (y, 1), (z, 2)]:
            pos = positions[:, :, axis].long()  # (B, N)
            c = F.embedding(pos, cos).unsqueeze(1)  # (B, 1, N, d)
            s = F.embedding(pos, sin).unsqueeze(1)
            chunk_rot = chunk * c + self._rotate_half(chunk) * s
            if axis == 0:
                x = chunk_rot
            elif axis == 1:
                y = chunk_rot
            else:
                z = chunk_rot

        return torch.cat([x, y, z], dim=-1)


# ============================================================
# Building Blocks
# ============================================================

class ConvBlock(nn.Module):
    """Conv1d block for early stages (cheap, no KNN, no attention).

    Simple: Conv1d + BN + GELU + Conv1d + BN + residual.
    Operates per-point (1x1 convolution). Local geometry comes implicitly
    from the densification and the attention in later stages.
    """

    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.drop_path = drop_path

    def forward(self, x, xyz=None):
        """x: (B, C, N), xyz: unused (kept for API compat with AttnBlock)"""
        return x + self.conv(x)


class AttnBlock(nn.Module):
    """Serialized window attention block with PointROPE.

    Sorts points by Z-order curve, splits into windows of patch_size,
    runs multi-head self-attention within each window.
    """

    def __init__(self, channels, num_heads, patch_size=1024, rope_freq=100.0, drop_path=0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.patch_size = patch_size

        assert channels % num_heads == 0
        assert channels % 3 == 0, "channels must be divisible by 3 for PointROPE"

        # Attention
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.rope = PointROPE(freq=rope_freq)

        # FFN
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x, xyz):
        """x: (B, C, N), xyz: (B, 3, N)"""
        B, C, N = x.shape
        H = self.num_heads
        K = self.patch_size

        # Serialize points by Z-order
        order, grid_coords = serialize_points(xyz)  # (B, N), (B, N, 3)

        # Sort features by Z-order
        x_sorted = torch.gather(x, 2, order.unsqueeze(1).expand(-1, C, -1))  # (B, C, N)
        grid_sorted = torch.gather(grid_coords, 1, order.unsqueeze(2).expand(-1, -1, 3))

        # Transpose to (B, N, C) for attention
        x_t = x_sorted.permute(0, 2, 1)

        # --- Attention ---
        shortcut = x_t
        x_t = self.norm1(x_t)

        # Pad to multiple of patch_size
        pad = (K - N % K) % K
        if pad > 0:
            x_t = F.pad(x_t, (0, 0, 0, pad))
            grid_sorted = F.pad(grid_sorted, (0, 0, 0, pad))

        N_pad = x_t.shape[1]
        n_windows = N_pad // K

        # QKV
        qkv = self.qkv(x_t).reshape(B, N_pad, 3, H, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N_pad, H, D)

        # Apply PointROPE
        q = q.permute(0, 2, 1, 3)  # (B, H, N_pad, D)
        k = k.permute(0, 2, 1, 3)

        q = self.rope(q, grid_sorted)
        k = self.rope(k, grid_sorted)

        # Reshape into windows: (B * n_windows, H, K, D)
        q = q.reshape(B, H, n_windows, K, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * n_windows, H, K, self.head_dim)
        k = k.reshape(B, H, n_windows, K, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * n_windows, H, K, self.head_dim)
        v = v.permute(0, 2, 1, 3).reshape(B, H, n_windows, K, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * n_windows, H, K, self.head_dim)

        # No attention mask — keeps flash attention enabled (massive VRAM savings).
        # Padding tokens in the last window are a negligible quality issue.
        out = F.scaled_dot_product_attention(q, k, v)  # (B*W, H, K, D)

        # Reshape back
        out = out.reshape(B, n_windows, H, K, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B, N_pad, C)

        # Remove padding
        out = out[:, :N]
        out = self.proj(out)
        x_t = shortcut + out

        # --- FFN ---
        shortcut = x_t
        x_t = shortcut + self.ffn(self.norm2(x_t))

        # Unsort: scatter back to original point order
        x_out = x_t.permute(0, 2, 1)  # (B, C, N)
        # Inverse of order: scatter sorted features back
        inv_order = torch.argsort(order)
        x_out = torch.gather(x_out, 2, inv_order.unsqueeze(1).expand(-1, C, -1))

        return x_out


class GridPool(nn.Module):
    """Downsample by random subsampling + linear projection."""

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.stride = stride
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x, xyz):
        B, C, N = x.shape
        x = self.proj(x)
        C_new = x.shape[1]
        n_keep = max(N // self.stride, 1)

        idx = torch.stack([torch.randperm(N, device=x.device)[:n_keep] for _ in range(B)])
        idx_sorted, _ = idx.sort(dim=1)

        x_down = torch.gather(x, 2, idx_sorted.unsqueeze(1).expand(-1, C_new, -1))
        xyz_down = torch.gather(xyz, 2, idx_sorted.unsqueeze(1).expand(-1, 3, -1))

        return x_down, xyz_down


# ============================================================
# LitePT Instance Segmentation Model
# ============================================================

class LitePTInstanceSeg(nn.Module):
    """LitePT-S architecture adapted for instance segmentation.

    Conv-early (stages 0-2) + Attention-late (stages 3-4) with PointROPE.
    Dual heads: semantic classification + offset regression.

    LitePT-S config: C=(36,72,144,252,504), B=(2,2,2,6,2)
    Adapted: C=(48,96,192,252,504), B=(2,2,2,4,2) for our use case.
    """

    def __init__(self, input_dim=10, num_classes=16, patch_size=1024, rope_freq=100.0):
        super().__init__()

        # Encoder config (LitePT-S inspired)
        # Attention stages need head_dim divisible by 8 (flash attn) and 3 (PointROPE)
        # head_dim=24: 24÷8=3✓ 24÷3=8✓
        enc_dims = [48, 96, 192, 288, 576]
        enc_depths = [2, 2, 2, 4, 2]
        enc_heads = [0, 0, 0, 12, 24]  # 288/12=24, 576/24=24
        strides = [2, 2, 2, 2]

        # Decoder config
        dec_dims = [288, 192, 96, 48]

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv1d(input_dim, enc_dims[0], 1, bias=False),
            nn.BatchNorm1d(enc_dims[0]),
            nn.GELU(),
        )

        # Encoder stages
        self.enc_blocks = nn.ModuleList()
        self.enc_pools = nn.ModuleList()

        for s in range(5):
            blocks = nn.ModuleList()
            for _ in range(enc_depths[s]):
                if enc_heads[s] == 0:
                    blocks.append(ConvBlock(enc_dims[s]))
                else:
                    blocks.append(AttnBlock(
                        enc_dims[s], enc_heads[s],
                        patch_size=patch_size,
                        rope_freq=rope_freq,
                    ))
            self.enc_blocks.append(blocks)
            if s < 4:
                self.enc_pools.append(GridPool(enc_dims[s], enc_dims[s + 1], stride=strides[s]))

        # Decoder (lightweight: just upsample + skip + project)
        self.dec_projs = nn.ModuleList()
        for i in range(4):
            in_ch = enc_dims[4 - i] + enc_dims[3 - i]  # upsampled + skip
            self.dec_projs.append(nn.Sequential(
                nn.Conv1d(in_ch, dec_dims[i], 1, bias=False),
                nn.BatchNorm1d(dec_dims[i]),
                nn.GELU(),
            ))

        # Heads
        final_ch = dec_dims[-1]
        self.sem_head = nn.Sequential(
            nn.Conv1d(final_ch, final_ch, 1, bias=False),
            nn.BatchNorm1d(final_ch),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(final_ch, num_classes, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv1d(final_ch, final_ch, 1, bias=False),
            nn.BatchNorm1d(final_ch),
            nn.GELU(),
            nn.Conv1d(final_ch, 3, 1),
        )

    def _upsample(self, x_low, N_high):
        B, C, N_low = x_low.shape
        idx = torch.arange(N_high, device=x_low.device).unsqueeze(0).expand(B, -1)
        idx = (idx * N_low // N_high).clamp(0, N_low - 1)
        return torch.gather(x_low, 2, idx.unsqueeze(1).expand(-1, C, -1))

    def forward(self, x):
        """
        Args:
            x: (B, input_dim, N). First 3 channels = local_xyz.
        Returns:
            sem_logits: (B, num_classes, N)
            offsets: (B, 3, N)
        """
        B, _, N = x.shape
        xyz = x[:, :3, :]

        x = self.patch_embed(x)

        # Encoder with skip connections
        skips = []
        xyzs = [xyz]
        Ns = [N]

        for s in range(5):
            for block in self.enc_blocks[s]:
                x = block(x, xyzs[-1])
            skips.append(x)
            if s < 4:
                x, xyz_down = self.enc_pools[s](x, xyzs[-1])
                xyzs.append(xyz_down)
                Ns.append(x.shape[2])

        # Decoder
        for i in range(4):
            skip_idx = 3 - i
            x_up = self._upsample(x, Ns[skip_idx])
            x = torch.cat([x_up, skips[skip_idx]], dim=1)
            x = self.dec_projs[i](x)

        return self.sem_head(x), self.offset_head(x)
