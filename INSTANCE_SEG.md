# PTv2 Instance Segmentation

Per-point semantic classification (16 classes) + instance grouping from point clouds.
Derives oriented bounding boxes from segmented instances at inference time.

## Classes

| ID | Name | Type |
|----|------|------|
| 0 | wall | stuff |
| 1 | floor | stuff |
| 2 | ceiling | stuff |
| 3 | door | thing |
| 4 | window | thing |
| 5 | cabinet | thing |
| 6 | bed | thing |
| 7 | chair | thing |
| 8 | sofa | thing |
| 9 | table | thing |
| 10 | bookshelf | thing |
| 11 | desk | thing |
| 12 | dresser | thing |
| 13 | toilet | thing |
| 14 | sink | thing |
| 15 | other | stuff |

"Stuff" = no instance IDs (walls, floors, etc).
"Thing" = each object gets a unique instance ID.

## Data

Source: [Structured3D](https://structured3d-dataset.org/) panorama depth + semantic + instance images.

### Small dataset (~1115 rooms from panorama_00)

```bash
python scripts/build_small_detection_dataset.py
```

Output: `data/detection_small/{train,val,test}/` with `coord.npy`, `semantic.npy`, `instance.npy` per room.

### Full dataset (~21K rooms from all 18 panorama zips)

Auto-downloads any missing zips with 16 parallel connections:

```bash
python scripts/build_detection_dataset.py
```

Output: `data/detection_full/{train,val,test}/`

## Training

```bash
python scripts/run_detection_training.py --config config_detection.yaml
```

Config: `config_detection.yaml`

### Training pipeline (per batch, on GPU)

```
raw room (~40-60K pts)
  -> densify 10x (keep all ~400-600K pts)
  -> augment (rotation, flip, scale, 5-20mm noise)
  -> compute 10-dim features
  -> PTv2 encoder-decoder
  -> semantic head (16 classes) + offset head (3D to instance center)
  -> loss = CrossEntropy + L1 offset
```

### Key config options

```yaml
training:
  batch_size: 1         # full dense rooms are large
  epochs: 100
  learning_rate: 0.001
  offset_weight: 1.0    # weight for offset loss vs semantic loss

augmentation:
  densify: true
  densify_multiplier: 10
```

Checkpoints saved to `checkpoints/det_run_XXX/`. Epoch 1 always saves.

### Resume training

```bash
python scripts/run_detection_training.py --resume checkpoints/det_run_001/checkpoint_epoch_10.pth
```

## Inference on real LiDAR

```bash
python scripts/instance_inference.py \
  --input path/to/room.ply \
  --checkpoint checkpoints/det_run_001/best_model.pth
```

Pipeline:
1. Predict semantic class + 3D offset per point
2. For each "thing" class, shift points by predicted offset
3. Cluster shifted points (nearby shifted points = same instance)
4. Compute oriented bounding box per instance cluster
5. Save `room_instances.ply` with instance colors + box wireframes

### Options

```
--threshold 0.3         detection score threshold
--cluster_eps 0.3       clustering distance (meters)
--min_instance_pts 100  minimum points per instance
```

## Visualization

Preview the data pipeline on a room before training:

```bash
python scripts/visualize_detection_sample.py
python scripts/visualize_detection_sample.py --room data/detection_small/train/scene_00123_room1195
```

Outputs to `data/detection_small/viz/`:
- `1_raw_semantic.ply` — original points colored by class
- `1_raw_instance.ply` — original points colored by instance ID
- `2_densified.ply` — 10x dense, instance colors
- `3_augmented.ply` — densified + augmented (what the model sees)

## Model

`PTv2InstanceSeg` — Point Transformer V2 encoder-decoder with grouped vector attention.

- Encoder: 4 stages [48, 96, 192, 256] channels, 4x downsampling between stages
- Decoder: 3 upsample stages with skip connections [192, 96, 48]
- Semantic head: 48 -> 16 classes
- Offset head: 48 -> 3D offset
- ~2.4M parameters

## Files

```
config_detection.yaml                    # config
scripts/build_small_detection_dataset.py # extract small dataset (panorama_00)
scripts/build_detection_dataset.py       # extract full dataset (all 18 zips, auto-download)
scripts/run_detection_training.py        # train
scripts/instance_inference.py            # inference on .ply
scripts/visualize_detection_sample.py    # visualize data pipeline
scripts/parallel_download.py             # multi-connection downloader
src/detection_model.py                   # PTv2InstanceSeg model
src/detection_dataset.py                 # InstanceSegDataset + collate
src/detection_train.py                   # training loop + eval + GPU augment
```
