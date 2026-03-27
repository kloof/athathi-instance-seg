"""PTv3 wrapper for instance segmentation.

Wraps Pointcept's PointTransformerV3 to work with our batched (B, C, N) data format.
Converts batched tensors to PTv3's sparse dict format, runs the backbone,
then adds semantic + offset heads.

Input:  (B, input_dim, N) batched point features (first 3 channels = xyz)
Output: sem_logits (B, num_classes, N), offsets (B, 3, N)
"""

import sys
import torch
import torch.nn as nn

# Patch spconv to use Native algo globally (avoids ConvTunerSimple assert failures)
import spconv.pytorch as spconv_pytorch
from spconv.core import ConvAlgo
_orig_subm3d_init = spconv_pytorch.SubMConv3d.__init__
def _patched_subm3d_init(self, *args, **kwargs):
    kwargs["algo"] = ConvAlgo.Native
    _orig_subm3d_init(self, *args, **kwargs)
spconv_pytorch.SubMConv3d.__init__ = _patched_subm3d_init

_orig_sparseconv3d_init = spconv_pytorch.SparseConv3d.__init__
def _patched_sparseconv3d_init(self, *args, **kwargs):
    kwargs["algo"] = ConvAlgo.Native
    _orig_sparseconv3d_init(self, *args, **kwargs)
spconv_pytorch.SparseConv3d.__init__ = _patched_sparseconv3d_init

sys.path.insert(0, "/workspace")
from PointTransformerV3.model import PointTransformerV3


class PTv3InstanceSeg(nn.Module):
    """PTv3 backbone + dual heads for instance segmentation."""

    def __init__(
        self,
        input_dim=10,
        num_classes=16,
        grid_size=0.03,
        # PTv3 config — smaller variant to fit RTX 3090
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(6, 6, 12, 24),
        dec_patch_size=(256, 256, 256, 256),
        enable_flash=True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim

        # Feature projection: our 10-dim features -> PTv3 in_channels
        # PTv3 expects raw features (not xyz — it gets coord separately)
        feat_dim = input_dim - 3  # exclude xyz which goes to coord
        if feat_dim < 1:
            feat_dim = 3  # fallback: use xyz as features too

        self.backbone = PointTransformerV3(
            in_channels=feat_dim,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            stride=(2, 2, 2, 2),
            enable_flash=enable_flash,
            upcast_attention=False,
            upcast_softmax=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
            enable_rpe=False,
            drop_path=0.1,
        )

        # Output channel = first decoder channel
        out_ch = dec_channels[0]

        self.sem_head = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(out_ch, num_classes),
        )

        self.offset_head = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Linear(out_ch, 3),
        )

    def forward(self, x):
        """
        Args:
            x: (B, input_dim, N) batched features. First 3 channels = local xyz.
        Returns:
            sem_logits: (B, num_classes, N)
            offsets: (B, 3, N)
        """
        B, C, N = x.shape
        device = x.device

        # Extract xyz and features
        xyz = x[:, :3, :].permute(0, 2, 1)  # (B, N, 3)
        if C > 3:
            feat = x[:, 3:, :].permute(0, 2, 1)  # (B, N, feat_dim)
        else:
            feat = xyz.clone()  # use xyz as features

        # Convert batched format to PTv3 sparse format
        # PTv3 expects: coord (M, 3), feat (M, F), offset (B,) cumulative
        coords_list = []
        feats_list = []
        offset = []
        cumsum = 0

        for b in range(B):
            coords_list.append(xyz[b])  # (N, 3)
            feats_list.append(feat[b])  # (N, F)
            cumsum += N
            offset.append(cumsum)

        coord = torch.cat(coords_list, dim=0)  # (B*N, 3)
        feat_flat = torch.cat(feats_list, dim=0)  # (B*N, F)
        offset = torch.tensor(offset, device=device, dtype=torch.long)

        # Build data dict for PTv3
        data_dict = dict(
            coord=coord.float(),
            feat=feat_flat.float(),
            grid_size=self.grid_size,
            offset=offset,
        )

        # Run PTv3 backbone
        point = self.backbone(data_dict)

        # Extract output features
        out_feat = point.feat  # (M', F_out) — may be different size due to voxelization

        # PTv3 voxelizes internally, so output may have different N per sample.
        # We need to map back to original N points per batch element.
        out_offset = point.offset  # cumulative counts
        out_batch = point.batch if hasattr(point, 'batch') else None

        # Reconstruct batched output
        # Use the inverse map if available, otherwise pad/truncate
        batch_feats = []
        prev = 0
        for b in range(B):
            cur = out_offset[b].item()
            sample_feat = out_feat[prev:cur]  # (Nb, F_out)
            Nb = sample_feat.shape[0]

            if Nb >= N:
                sample_feat = sample_feat[:N]
            else:
                # Pad by repeating last point
                pad = sample_feat[-1:].expand(N - Nb, -1)
                sample_feat = torch.cat([sample_feat, pad], dim=0)

            batch_feats.append(sample_feat)
            prev = cur

        out = torch.stack(batch_feats, dim=0)  # (B, N, F_out)
        out_flat = out.reshape(B * N, -1)  # (B*N, F_out)

        # Heads
        sem_logits = self.sem_head(out_flat).reshape(B, N, -1).permute(0, 2, 1)  # (B, C, N)
        offsets = self.offset_head(out_flat).reshape(B, N, 3).permute(0, 2, 1)  # (B, 3, N)

        return sem_logits, offsets
