import os, numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

d = r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\structured3d_panorama_processed"
room = "scene_00686_room813889"
pts = np.load(os.path.join(d, room, "coord.npy"))
lbl = np.load(os.path.join(d, room, "label.npy"))

rng = np.random.default_rng(42)
tree = cKDTree(pts)
K = 6
dists, indices = tree.query(pts, k=K+1)
max_dist = 0.08

all_new_pts = []
all_new_lbl = []
for _ in range(5):
    nc = rng.integers(1, K+1, size=len(pts))
    ni = indices[np.arange(len(pts)), nc]
    nd = dists[np.arange(len(pts)), nc]
    valid = (nd < max_dist) & (lbl == lbl[ni])
    t = rng.uniform(0.2, 0.8, size=valid.sum())
    all_new_pts.append(pts[valid] * (1 - t[:, None]) + pts[ni[valid]] * t[:, None])
    all_new_lbl.append(lbl[valid])

dense_pts = np.concatenate([pts] + all_new_pts).astype(np.float32)
dense_lbl = np.concatenate([lbl] + all_new_lbl)

noise_mag = rng.uniform(0.005, 0.020, size=len(dense_pts))
noise_dir = rng.normal(0, 1, size=dense_pts.shape).astype(np.float32)
noise_dir /= np.linalg.norm(noise_dir, axis=1, keepdims=True) + 1e-8
dense_pts += (noise_dir * noise_mag[:, None]).astype(np.float32)

colors = np.full((len(dense_pts), 3), 0.5)
colors[dense_lbl == 1] = [1, 0, 0]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(dense_pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

out = r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\random_room_dense_x6_noise.ply"
ok = o3d.io.write_point_cloud(out, pcd)
print(f"Write: {ok}, exists: {os.path.exists(out)}, size: {os.path.getsize(out) if os.path.exists(out) else 0}")
print(f"{len(dense_pts):,} pts, noise=5-20mm")
