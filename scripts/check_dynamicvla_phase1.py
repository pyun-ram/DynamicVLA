#!/usr/bin/env python3
"""Check and visualize DynamicVLA Phase-1 raw HDF5 episodes.

This checker validates the *known* DynamicVLA Phase-1 camera convention and
outputs RGB/depth/point-cloud visualizations for quick geometry inspection.

DynamicVLA camera parameter definitions used here
-------------------------------------------------
DynamicVLA configures scene cameras in ``simulations/simulate.py`` via
``get_camera_pose()`` and ``CameraCfg.OffsetCfg(..., convention="opengl")``.
For Phase-1 raw H5 files we save these Isaac Lab camera data fields:

    {cam}_K
        Isaac Lab ``camera.data.intrinsic_matrices``. This is the 3x3 pinhole
        intrinsics matrix in image pixels. It is used with optical/ROS camera
        coordinates: x right, y down, z forward.

    {cam}_depth
        Isaac Lab ``distance_to_image_plane`` in meters. It is metric z-depth
        along the optical image plane convention above. Background/no-hit
        pixels may be +inf; this checker clips them to ``--far`` for
        visualization and PCD metrics.

    {cam}_pos_w
        Camera origin position in Isaac/world coordinates.

    {cam}_quat_w
        Isaac Lab ``quat_w_world`` in wxyz order. Despite DynamicVLA's camera
        offset convention being ``opengl``, this saved quaternion is the Isaac
        *world camera convention*: local +X is camera forward and local +Z is
        camera up. It is not directly an optical/ROS or OpenGL camera pose.

Therefore, to reconstruct a world-frame point cloud from metric depth:

    optical point = [(u-cx)/fx*z, (v-cy)/fy*z, z]
    Isaac-world-camera local point = [z, -x, -y]
    world point = R(quat_w_world) @ local + pos_w

The same transform is used to project world-frame EE poses back into images.
DynamicVLA's ``ee_pos`` is robot-relative, so by default this checker reads the
paired ``.json`` robot init pose and converts EE back to world before overlaying
or computing EE-to-PCD distances.

Expected DynamicVLA keys per camera prefix, e.g. ``opst_cam`` and ``side_cam``::

    {cam}_rgb      (T, H, W, 3) uint8
    {cam}_depth    (T, H, W, 1) float metric depth, may contain +inf background
    {cam}_K        (T, 3, 3) or (3, 3)
    {cam}_pos_w    (T, 3) or (3,)
    {cam}_quat_w   (T, 4) or (4,), wxyz Isaac ``quat_w_world``

Outputs, per sampled frame:
  - RGB/depth panels with EE projection.
  - 3D point-cloud overlap plot and optional PLY.
  - JSON summaries with field, depth, overlap, and EE distance metrics.

The checker does not mutate input data.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image, ImageDraw

# DynamicVLA saves Isaac Lab quat_w_world: local +X forward, local +Z up.
# Depth unprojection gives optical/ROS coordinates: x right, y down, z forward.
# Convert optical -> Isaac-world-camera local: [x,y,z] -> [z,-x,-y].
DYNAMICVLA_OPTICAL_TO_ISAAC_WORLD = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


@dataclass(frozen=True)
class CameraSpec:
    src: str
    name: str


@dataclass
class CameraFrame:
    src: str
    name: str
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    c2w_quat_world: np.ndarray


def parse_camera_specs(specs: Sequence[str]) -> List[CameraSpec]:
    cams: List[CameraSpec] = []
    for spec in specs:
        src, name = spec.split(":", 1) if ":" in spec else (spec, spec)
        src, name = src.strip(), name.strip()
        if not src or not name:
            raise ValueError(f"Invalid camera spec: {spec!r}")
        cams.append(CameraSpec(src=src, name=name))
    return cams


def find_episode_files(src_dir: Path, limit: Optional[int]) -> List[Path]:
    files = [src_dir] if src_dir.is_file() else sorted(src_dir.glob("*.h5"))
    return files[:limit] if limit is not None else files


def require_keys(h5: h5py.File, keys: Iterable[str]) -> List[str]:
    return [key for key in keys if key not in h5]


def at_frame(arr: np.ndarray, frame: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1 and arr.shape[0] in (3, 4):
        return arr
    if arr.ndim == 2 and arr.shape in {(3, 3), (4, 4)}:
        return arr
    return arr[frame]


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError(f"Zero-norm quaternion: {q}")
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_c2w_quat_world(pos_w: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = quat_wxyz_to_rotmat(quat_wxyz)
    c2w[:3, 3] = np.asarray(pos_w, dtype=np.float64).reshape(3)
    return c2w


def parse_vec_arg(text: Optional[str], n: int, name: str) -> Optional[np.ndarray]:
    if text is None:
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"--{name} expects {n} comma-separated floats, got {text!r}")
    return np.asarray(vals, dtype=np.float64)


def robot_relative_to_world(point_r: np.ndarray, robot_pos_w: np.ndarray, robot_quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_wxyz_to_rotmat(robot_quat_wxyz) @ np.asarray(point_r, dtype=np.float64).reshape(3) + robot_pos_w


def _is_numeric_vec(x, n: int) -> bool:
    if not isinstance(x, (list, tuple)) or len(x) != n:
        return False
    try:
        [float(v) for v in x]
        return True
    except (TypeError, ValueError):
        return False


def _find_robot_pose_recursive(obj) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Best-effort extraction of robot init_state pos/rot from DynamicVLA JSON."""
    if isinstance(obj, dict):
        if "init_state" in obj and isinstance(obj["init_state"], dict):
            init = obj["init_state"]
            if _is_numeric_vec(init.get("pos"), 3) and _is_numeric_vec(init.get("rot"), 4):
                return np.asarray(init["pos"], dtype=np.float64), np.asarray(init["rot"], dtype=np.float64)
        for key in ("robot", "scene"):
            if key in obj:
                found = _find_robot_pose_recursive(obj[key])
                if found is not None:
                    return found
        for value in obj.values():
            found = _find_robot_pose_recursive(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_robot_pose_recursive(value)
            if found is not None:
                return found
    return None


def load_robot_pose_for_episode(
    episode_path: Path,
    robot_pos_arg: Optional[np.ndarray],
    robot_quat_arg: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if robot_pos_arg is not None or robot_quat_arg is not None:
        if robot_pos_arg is None or robot_quat_arg is None:
            raise ValueError("--robot_pos and --robot_quat must be provided together")
        return robot_pos_arg, robot_quat_arg, "cli"
    json_path = episode_path.with_suffix(".json")
    if not json_path.exists():
        return None, None, "missing_json"
    with json_path.open("r", encoding="utf-8") as f:
        found = _find_robot_pose_recursive(json.load(f))
    if found is None:
        return None, None, "not_found_in_json"
    return found[0], found[1], "json"


def sanitize_depth(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected depth HxW after squeeze, got {depth.shape}")
    return np.nan_to_num(depth, nan=far, posinf=far, neginf=near).clip(near, far)


def load_camera_frame(h5: h5py.File, cam: CameraSpec, frame: int, near: float, far: float) -> CameraFrame:
    rgb = np.asarray(at_frame(h5[f"{cam.src}_rgb"], frame))
    depth = sanitize_depth(at_frame(h5[f"{cam.src}_depth"], frame), near=near, far=far)
    K = np.asarray(at_frame(h5[f"{cam.src}_K"], frame), dtype=np.float64)
    pos = np.asarray(at_frame(h5[f"{cam.src}_pos_w"], frame), dtype=np.float64)
    quat = np.asarray(at_frame(h5[f"{cam.src}_quat_w"], frame), dtype=np.float64)
    return CameraFrame(src=cam.src, name=cam.name, rgb=rgb, depth_m=depth, K=K, c2w_quat_world=make_c2w_quat_world(pos, quat))


def camera_points_optical_from_depth(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Return HxWx3 optical/ROS points: +x right, +y down, +z forward."""
    H, W = depth_m.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        raise ValueError(f"Invalid intrinsics with near-zero focal: {K}")
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth_m.astype(np.float32, copy=False)
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    return np.stack([x, y, z], axis=-1)


def depth_to_world_points(depth_m: np.ndarray, K: np.ndarray, c2w_quat_world: np.ndarray) -> np.ndarray:
    optical = camera_points_optical_from_depth(depth_m, K)
    local = optical @ DYNAMICVLA_OPTICAL_TO_ISAAC_WORLD.astype(np.float32).T
    R = c2w_quat_world[:3, :3]
    t = c2w_quat_world[:3, 3]
    return local @ R.T + t


def project_world_to_pixel(point_w: np.ndarray, K: np.ndarray, c2w_quat_world: np.ndarray) -> Optional[Tuple[float, float, float]]:
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = point_w
    local = (np.linalg.inv(c2w_quat_world) @ point_h)[:3]
    # local = M @ optical, so optical = M.T @ local.
    x_opt, y_opt, z_opt = DYNAMICVLA_OPTICAL_TO_ISAAC_WORLD.T @ local
    if z_opt <= 1e-9:
        return None
    u = float(K[0, 0] * x_opt / z_opt + K[0, 2])
    v = float(K[1, 1] * y_opt / z_opt + K[1, 2])
    return u, v, float(z_opt)


def depth_stats(raw_depth: np.ndarray, depth_m: np.ndarray, near: float, far: float) -> Dict[str, float]:
    raw = np.asarray(raw_depth).squeeze()
    finite = np.isfinite(raw)
    in_range = finite & (raw >= near) & (raw <= far)
    finite_vals = raw[finite]
    return {
        "finite_ratio": float(finite.mean()) if raw.size else 0.0,
        "inf_ratio": float(np.isinf(raw).mean()) if raw.size else 0.0,
        "in_range_ratio": float(in_range.sum() / float(raw.size)) if raw.size else 0.0,
        "min_finite": float(np.min(finite_vals)) if finite_vals.size else math.nan,
        "max_finite": float(np.max(finite_vals)) if finite_vals.size else math.nan,
        "p50_sanitized": float(np.percentile(depth_m, 50)),
        "p95_sanitized": float(np.percentile(depth_m, 95)),
    }


def choose_frames(T: int, frame_spec: str) -> List[int]:
    out: List[int] = []
    for token in [x.strip() for x in frame_spec.split(",") if x.strip()]:
        if token in {"first", "start"}:
            out.append(0)
        elif token in {"mid", "middle"}:
            out.append(max(0, T // 2))
        elif token in {"last", "end"}:
            out.append(max(0, T - 1))
        else:
            out.append(int(token))
    return sorted(set(max(0, min(T - 1, f)) for f in out))


def ensure_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb[..., :3]


def depth_to_uint8_vis(depth_m: np.ndarray, far: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m < far - 1e-4)
    if valid.any():
        lo = float(np.percentile(depth_m[valid], 2))
        hi = float(np.percentile(depth_m[valid], 98))
        if hi <= lo:
            hi = lo + 1e-6
    else:
        lo, hi = 0.0, far
    gray = (255.0 * (1.0 - ((depth_m - lo) / (hi - lo)).clip(0, 1))).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[~valid] = np.array([40, 70, 180], dtype=np.uint8)
    return rgb


def draw_ee_projection(img: np.ndarray, projection: Optional[Tuple[float, float, float]]) -> np.ndarray:
    pil = Image.fromarray(ensure_uint8_rgb(img)).copy()
    if projection is None:
        return np.asarray(pil)
    u, v, _ = projection
    draw = ImageDraw.Draw(pil)
    x, y, r = int(round(u)), int(round(v)), 7
    w, h = pil.size
    if 0 <= x < w and 0 <= y < h:
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=3)
        draw.line((x - 2 * r, y, x + 2 * r, y), fill=(255, 0, 0), width=2)
        draw.line((x, y - 2 * r, x, y + 2 * r), fill=(255, 0, 0), width=2)
    return np.asarray(pil)


def save_rgb_depth_panel(out_path: Path, cameras: Sequence[CameraFrame], ee_pos: Optional[np.ndarray], far: float) -> None:
    tiles: List[Image.Image] = []
    for cam in cameras:
        projection = None if ee_pos is None else project_world_to_pixel(ee_pos, cam.K, cam.c2w_quat_world)
        tiles.append(Image.fromarray(draw_ee_projection(cam.rgb, projection)))
        tiles.append(Image.fromarray(draw_ee_projection(depth_to_uint8_vis(cam.depth_m, far=far), projection)))
    W = max(im.width for im in tiles) if tiles else 1
    H = max(im.height for im in tiles) if tiles else 1
    canvas = Image.new("RGB", (W * 2, H * len(cameras)), color=(255, 255, 255))
    for row in range(len(cameras)):
        canvas.paste(tiles[2 * row], (0, row * H))
        canvas.paste(tiles[2 * row + 1], (W, row * H))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def voxel_overlap(points_by_cam: Mapping[str, np.ndarray], voxel_size: float) -> Dict[str, float]:
    sets: Dict[str, set] = {}
    for name, pts in points_by_cam.items():
        q = np.floor(pts / voxel_size).astype(np.int64) if pts.size else np.empty((0, 3), dtype=np.int64)
        sets[name] = {tuple(row) for row in q}
    metrics: Dict[str, float] = {}
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            denom = max(1, min(len(sets[a]), len(sets[b])))
            metrics[f"{a}__{b}"] = float(len(sets[a] & sets[b]) / denom)
    return metrics


def sample_valid_points(points: np.ndarray, depth_m: np.ndarray, near: float, far: float, max_points: int, rng: np.random.Generator) -> np.ndarray:
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth_m) & (depth_m > near) & (depth_m < far - 1e-4)
    flat = points[valid].reshape(-1, 3)
    if flat.shape[0] > max_points:
        flat = flat[rng.choice(flat.shape[0], size=max_points, replace=False)]
    return flat.astype(np.float32, copy=False)


def subsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.size == 0 or points.shape[0] <= max_points:
        return points
    return points[rng.choice(points.shape[0], size=max_points, replace=False)]


def crop_points_near(point: Optional[np.ndarray], points: np.ndarray, radius: float) -> np.ndarray:
    if point is None or points.size == 0 or radius <= 0:
        return points
    return points[np.linalg.norm(points - point.reshape(1, 3), axis=1) <= radius]


def nearest_distances_chunked(src: np.ndarray, dst: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    if src.size == 0 or dst.size == 0:
        return np.empty((0,), dtype=np.float32)
    out = np.empty((src.shape[0],), dtype=np.float32)
    dst_f = dst.astype(np.float32, copy=False)
    for start in range(0, src.shape[0], chunk_size):
        block = src[start : start + chunk_size].astype(np.float32, copy=False)
        diff = block[:, None, :] - dst_f[None, :, :]
        out[start : start + block.shape[0]] = np.sqrt(np.min(np.einsum("bnc,bnc->bn", diff, diff), axis=1))
    return out


def pairwise_nn_metrics(points_by_cam: Mapping[str, np.ndarray], rng: np.random.Generator, max_points: int, threshold_m: float, prefix: str = "") -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    names = sorted(points_by_cam)
    sampled = {name: subsample_points(points_by_cam[name], max_points=max_points, rng=rng) for name in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            A, B = sampled[a], sampled[b]
            key = f"{prefix}{a}__{b}"
            if A.size == 0 or B.size == 0:
                metrics[key] = {"symmetric_median_m": math.nan, "symmetric_p90_m": math.nan, "under_threshold_ratio": 0.0, "num_a": int(A.shape[0]), "num_b": int(B.shape[0])}
                continue
            d = np.concatenate([nearest_distances_chunked(A, B), nearest_distances_chunked(B, A)])
            metrics[key] = {
                "symmetric_median_m": float(np.median(d)),
                "symmetric_p90_m": float(np.percentile(d, 90)),
                "under_threshold_ratio": float(np.mean(d <= threshold_m)),
                "num_a": int(A.shape[0]),
                "num_b": int(B.shape[0]),
            }
    return metrics


def nearest_distance(point: Optional[np.ndarray], points: np.ndarray) -> float:
    if point is None or points.size == 0:
        return math.nan
    return float(np.min(np.linalg.norm(points - point.reshape(1, 3), axis=1)))


def equalize_3d_axes(ax, pts: np.ndarray, ee_pos: Optional[np.ndarray]) -> None:
    if pts.size == 0:
        return
    all_pts = np.concatenate([pts, ee_pos.reshape(1, 3)], axis=0) if ee_pos is not None else pts
    center = np.nanmedian(all_pts, axis=0)
    span = np.nanpercentile(all_pts, 97, axis=0) - np.nanpercentile(all_pts, 3, axis=0)
    radius = max(float(np.nanmax(span)) / 2.0, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_pcd_plot(out_path: Path, points_by_cam: Mapping[str, np.ndarray], ee_pos: Optional[np.ndarray], ee_traj: Optional[np.ndarray], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    all_pts = []
    for idx, (name, pts) in enumerate(points_by_cam.items()):
        if pts.size:
            all_pts.append(pts)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.25, alpha=0.35, c=colors[idx % len(colors)], label=name)
    if ee_traj is not None and ee_traj.size > 0:
        ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], c="black", linewidth=1.4, label="ee trajectory")
    if ee_pos is not None:
        ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], c="red", s=70, marker="*", label="ee pose")
    if all_pts:
        equalize_3d_axes(ax, np.concatenate(all_pts, axis=0), ee_pos)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_zlabel("world z")
    ax.set_title(title)
    ax.legend(loc="best", markerscale=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_ascii_ply(out_path: Path, points_by_cam: Mapping[str, np.ndarray], ee_pos: Optional[np.ndarray]) -> None:
    palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189)]
    rows: List[Tuple[float, float, float, int, int, int]] = []
    for idx, (_name, pts) in enumerate(points_by_cam.items()):
        rgb = palette[idx % len(palette)]
        rows.extend((float(p[0]), float(p[1]), float(p[2]), *rgb) for p in pts)
    if ee_pos is not None:
        for axis in range(3):
            for delta in (-0.02, 0.0, 0.02):
                p = ee_pos.copy()
                p[axis] += delta
                rows.append((float(p[0]), float(p[1]), float(p[2]), 255, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for row in rows:
            f.write("%.6f %.6f %.6f %d %d %d\n" % row)


def get_episode_T(h5: h5py.File) -> int:
    for key in ("action", "ee_pos", "joints"):
        if key in h5:
            return int(h5[key].shape[0])
    rgb_keys = [k for k in h5.keys() if k.endswith("_rgb")]
    if rgb_keys:
        return int(h5[rgb_keys[0]].shape[0])
    raise ValueError("Cannot infer episode length")


def camera_static_error(h5: h5py.File, cam: CameraSpec) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for suffix in ("K", "pos_w", "quat_w"):
        key = f"{cam.src}_{suffix}"
        if key in h5:
            arr = np.asarray(h5[key])
            out[suffix] = float(np.max(np.abs(arr - arr[0]))) if arr.ndim >= 2 and not (arr.ndim == 2 and suffix == "K") else 0.0
    return out


def check_one_episode(
    episode_path: Path,
    out_dir: Path,
    cams: Sequence[CameraSpec],
    frame_spec: str,
    near: float,
    far: float,
    max_points_per_cam: int,
    voxel_size: float,
    nn_max_points: int,
    nn_threshold: float,
    ee_metric_radius: float,
    robot_pos_arg: Optional[np.ndarray],
    robot_quat_arg: Optional[np.ndarray],
    ee_frame: str,
    write_ply: bool,
    rng: np.random.Generator,
) -> Dict:
    summary: MutableMapping[str, object] = {"episode": str(episode_path), "ok": True, "missing_keys": [], "frames": {}}
    robot_pos_w, robot_quat_wxyz, robot_pose_source = load_robot_pose_for_episode(episode_path, robot_pos_arg, robot_quat_arg)
    summary["robot_pose_source"] = robot_pose_source
    if robot_pos_w is not None:
        summary["robot_pos_w"] = robot_pos_w.tolist()
        summary["robot_quat_wxyz"] = robot_quat_wxyz.tolist()

    with h5py.File(episode_path, "r") as h5:
        required = ["action", "ee_pos", "ee_quat", "joints"]
        for cam in cams:
            required.extend([f"{cam.src}_rgb", f"{cam.src}_depth", f"{cam.src}_K", f"{cam.src}_pos_w", f"{cam.src}_quat_w"])
        missing = require_keys(h5, required)
        summary["missing_keys"] = missing
        if missing:
            summary["ok"] = False
            return dict(summary)

        T = get_episode_T(h5)
        summary["T"] = T
        summary["keys"] = sorted(h5.keys())
        summary["camera_static_max_abs"] = {cam.name: camera_static_error(h5, cam) for cam in cams}
        ep_out = out_dir / episode_path.stem
        ep_out.mkdir(parents=True, exist_ok=True)
        ee_all = np.asarray(h5["ee_pos"])

        for frame in choose_frames(T, frame_spec):
            frame_summary: MutableMapping[str, object] = {"cameras": {}, "geometry": {}}
            ee_raw = np.asarray(at_frame(h5["ee_pos"], frame), dtype=np.float64).reshape(3)
            if ee_frame == "world" or (ee_frame == "auto" and robot_pos_w is None):
                ee_pos = ee_raw
                ee_traj = ee_all[: frame + 1].astype(np.float64) if ee_all.ndim == 2 else None
                ee_frame_used = "world_raw" if ee_frame == "world" else "world_raw_auto_no_robot_pose"
            else:
                if robot_pos_w is None or robot_quat_wxyz is None:
                    ee_pos = ee_raw
                    ee_traj = ee_all[: frame + 1].astype(np.float64) if ee_all.ndim == 2 else None
                    ee_frame_used = "robot_relative_unconverted_missing_robot_pose"
                else:
                    ee_pos = robot_relative_to_world(ee_raw, robot_pos_w, robot_quat_wxyz)
                    ee_traj = np.stack([robot_relative_to_world(e[:3], robot_pos_w, robot_quat_wxyz) for e in ee_all[: frame + 1]], axis=0)
                    ee_frame_used = f"robot_relative_to_world_from_{robot_pose_source}"
            frame_summary["ee_raw"] = ee_raw.tolist()
            frame_summary["ee_world_for_metrics"] = ee_pos.tolist()
            frame_summary["ee_frame_used"] = ee_frame_used

            cam_frames = [load_camera_frame(h5, cam, frame, near, far) for cam in cams]
            for cam, cf in zip(cams, cam_frames):
                raw_depth = np.asarray(at_frame(h5[f"{cam.src}_depth"], frame)).squeeze()
                frame_summary["cameras"][cam.name] = {
                    "rgb_shape": list(cf.rgb.shape),
                    "depth_shape": list(cf.depth_m.shape),
                    "K": cf.K.tolist(),
                    "c2w_quat_world": cf.c2w_quat_world.tolist(),
                    "depth": depth_stats(raw_depth, cf.depth_m, near, far),
                }

            save_rgb_depth_panel(ep_out / f"frame_{frame:04d}_rgb_depth.png", cam_frames, ee_pos, far)
            points_by_cam: Dict[str, np.ndarray] = {}
            for cf in cam_frames:
                pts = depth_to_world_points(cf.depth_m, cf.K, cf.c2w_quat_world)
                points_by_cam[cf.name] = sample_valid_points(pts, cf.depth_m, near, far, max_points_per_cam, rng)
            all_pts = np.concatenate([p for p in points_by_cam.values() if p.size], axis=0) if any(p.size for p in points_by_cam.values()) else np.empty((0, 3), dtype=np.float32)
            near_ee_points_by_cam = {name: crop_points_near(ee_pos, pts, ee_metric_radius) for name, pts in points_by_cam.items()}
            metrics = {
                "camera_pose_convention": "DynamicVLA Isaac quat_w_world; optical [x,y,z] -> local [z,-x,-y]",
                "voxel_overlap": voxel_overlap(points_by_cam, voxel_size),
                "pcd_nn_all": pairwise_nn_metrics(points_by_cam, rng, nn_max_points, nn_threshold),
                "pcd_nn_near_ee": pairwise_nn_metrics(near_ee_points_by_cam, rng, nn_max_points, nn_threshold, prefix=f"ee_radius_{ee_metric_radius:.2f}m__"),
                "ee_to_nearest_pcd_m": nearest_distance(ee_pos, all_pts),
                "num_points": {name: int(pts.shape[0]) for name, pts in points_by_cam.items()},
                "num_points_near_ee": {name: int(pts.shape[0]) for name, pts in near_ee_points_by_cam.items()},
            }
            frame_summary["geometry"] = metrics
            save_pcd_plot(ep_out / f"frame_{frame:04d}_pcd_overlap.png", points_by_cam, ee_pos, ee_traj, f"{episode_path.name} frame={frame}")
            if write_ply:
                write_ascii_ply(ep_out / f"frame_{frame:04d}_pcd_overlap.ply", points_by_cam, ee_pos)
            summary["frames"][str(frame)] = dict(frame_summary)

    with (out_dir / episode_path.stem / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return dict(summary)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src_dir", type=Path, required=True, help="DynamicVLA raw H5 file or directory containing *.h5")
    p.add_argument("--out_dir", type=Path, default=Path("phase1_raw_check_viz"), help="Directory for visualization outputs")
    p.add_argument("--camera", action="append", default=["opst_cam:front", "side_cam:left_shoulder"], help="Camera mapping as DynamicVLA_prefix:display_name. Repeatable.")
    p.add_argument("--frames", default="first,mid,last", help="Comma list of frame ids or tokens: first,mid,last")
    p.add_argument("--near", type=float, default=0.01, help="Near plane used for clipping/sanity stats")
    p.add_argument("--far", type=float, default=4.5, help="Far plane; +inf depth is clipped to this value")
    p.add_argument("--limit", type=int, default=None, help="Max number of H5 episodes to inspect")
    p.add_argument("--max_points_per_cam", type=int, default=12000, help="Max sampled PCD points per camera for plots")
    p.add_argument("--voxel_size", type=float, default=0.03, help="Voxel size in meters for rough multi-view overlap metric")
    p.add_argument("--nn_max_points", type=int, default=2500, help="Max sampled points per camera for pairwise nearest-neighbor PCD metrics")
    p.add_argument("--nn_threshold", type=float, default=0.05, help="Distance threshold in meters for NN overlap ratio")
    p.add_argument("--ee_metric_radius", type=float, default=1.2, help="Radius in meters around EE pose for local PCD overlap metrics")
    p.add_argument("--ee_frame", choices=["auto", "robot", "world"], default="auto", help="Interpret H5 ee_pos as robot-relative or world. auto uses same-name DynamicVLA JSON robot pose when found.")
    p.add_argument("--robot_pos", default=None, help="Override robot world position as 'x,y,z' when JSON cannot be parsed")
    p.add_argument("--robot_quat", default=None, help="Override robot world quaternion wxyz as 'w,x,y,z' when JSON cannot be parsed")
    p.add_argument("--write_ply", action="store_true", help="Also write ASCII PLY point clouds for Open3D/MeshLab inspection")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    cams = parse_camera_specs(args.camera)
    robot_pos_arg = parse_vec_arg(args.robot_pos, 3, "robot_pos")
    robot_quat_arg = parse_vec_arg(args.robot_quat, 4, "robot_quat")
    files = find_episode_files(args.src_dir, args.limit)
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {args.src_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summaries = []
    for path in files:
        print(f"[check] {path}")
        summary = check_one_episode(
            path,
            out_dir=args.out_dir,
            cams=cams,
            frame_spec=args.frames,
            near=args.near,
            far=args.far,
            max_points_per_cam=args.max_points_per_cam,
            voxel_size=args.voxel_size,
            nn_max_points=args.nn_max_points,
            nn_threshold=args.nn_threshold,
            ee_metric_radius=args.ee_metric_radius,
            robot_pos_arg=robot_pos_arg,
            robot_quat_arg=robot_quat_arg,
            ee_frame=args.ee_frame,
            write_ply=args.write_ply,
            rng=rng,
        )
        summaries.append(summary)
        if not summary.get("ok", False):
            print(f"  FAIL missing={summary.get('missing_keys')}")
            continue
        for frame_id, frame_summary in summary.get("frames", {}).items():
            metrics = frame_summary.get("geometry", {})
            nn_all = next(iter(metrics.get("pcd_nn_all", {}).values()), {})
            nn_near = next(iter(metrics.get("pcd_nn_near_ee", {}).values()), {})
            print(
                f"  frame={frame_id} "
                f"ee_nn={metrics['ee_to_nearest_pcd_m']:.4f}m "
                f"symNN={nn_all.get('symmetric_median_m', math.nan):.4f}m "
                f"nearEE_symNN={nn_near.get('symmetric_median_m', math.nan):.4f}m "
                f"overlap={metrics['voxel_overlap']}"
            )
    manifest = {
        "src_dir": str(args.src_dir),
        "out_dir": str(args.out_dir),
        "near": args.near,
        "far": args.far,
        "camera": args.camera,
        "camera_pose_convention": "DynamicVLA Isaac quat_w_world; optical [x,y,z] -> local [z,-x,-y]",
        "summaries": summaries,
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    ok = all(s.get("ok", False) for s in summaries)
    print(f"[done] wrote {args.out_dir}; ok={ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
