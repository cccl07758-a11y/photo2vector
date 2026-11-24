"""阶段二共用的轻量几何工具。只处理已经压缩过的笔画，不碰整图像素图。"""

from __future__ import annotations

import numpy as np

_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def as_f32(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr


def arc_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    delta = np.diff(pts, axis=0)
    return float(np.hypot(delta[:, 0], delta[:, 1]).sum())


def principal_axis(sample: np.ndarray, hint: np.ndarray) -> np.ndarray:
    """2×2 协方差的主方向，符号顺着 hint。替代小点集上的 SVD。"""
    pts = np.asarray(sample, dtype=np.float32)
    hint_u = unit(hint)
    if len(pts) < 2:
        return hint_u if float(np.linalg.norm(hint_u)) > 0.5 else np.array([1.0, 0.0], np.float32)
    centered = pts - pts.mean(axis=0)
    xx = float(centered[:, 0] @ centered[:, 0])
    xy = float(centered[:, 0] @ centered[:, 1])
    yy = float(centered[:, 1] @ centered[:, 1])
    if abs(xy) + abs(xx - yy) < 1e-8:
        direction = hint_u
    else:
        theta = 0.5 * float(np.arctan2(2.0 * xy, xx - yy))
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    if float(np.dot(direction, hint_u)) < 0.0:
        direction = -direction
    out = unit(direction)
    if float(np.linalg.norm(out)) < 0.5:
        return hint_u if float(np.linalg.norm(hint_u)) > 0.5 else np.array([1.0, 0.0], np.float32)
    return out


def unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.zeros_like(arr)
    return arr / np.float32(norm)


def dot(a, b) -> float:
    return float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))


def cross2(a, b) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(aa[0] * bb[1] - aa[1] * bb[0])


def angle_between(a, b) -> float:
    """两向量夹角，单位度，范围 0~180。"""
    aa = unit(a)
    bb = unit(b)
    if float(np.linalg.norm(aa)) < 0.5 or float(np.linalg.norm(bb)) < 0.5:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(aa, bb), -1.0, 1.0))))


def turn_through(t1, t2) -> float:
    """两支指向交点的切向穿过交点时的转角。0° 是直线贯通，180° 是折返。"""
    return float(np.degrees(np.arccos(np.clip(-dot(t1, t2), -1.0, 1.0))))


def chord_params(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.zeros(len(pts), np.float32)
    delta = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cumulative = np.concatenate(([0.0], np.cumsum(delta)))
    total = float(cumulative[-1])
    if total < 1e-6:
        return np.linspace(0.0, 1.0, len(pts)).astype(np.float32)
    return (cumulative / total).astype(np.float32)


def resample_polyline(pts: np.ndarray, step: float, radii: np.ndarray | None = None):
    pts = as_f32(pts)
    if len(pts) < 2:
        rad = None if radii is None else np.asarray(radii, dtype=np.float32)
        return pts.copy(), rad
    delta = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cumulative = np.concatenate(([0.0], np.cumsum(delta))).astype(np.float32)
    total = float(cumulative[-1])
    if total < 1e-4:
        rad = None if radii is None else np.asarray(radii[:1], dtype=np.float32)
        return pts[:1].copy(), rad
    count = max(2, int(np.ceil(total / max(step, 0.4))) + 1)
    targets = np.linspace(0.0, total, count, dtype=np.float32)
    xy = np.stack(
        (
            np.interp(targets, cumulative, pts[:, 0]),
            np.interp(targets, cumulative, pts[:, 1]),
        ),
        axis=1,
    ).astype(np.float32)
    if radii is None:
        return xy, None
    rad = np.interp(targets, cumulative, np.asarray(radii, dtype=np.float32)).astype(np.float32)
    return xy, rad


def max_deviation(pts: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    if len(pts) == 0:
        return 0.0
    chord = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    length = float(np.hypot(chord[0], chord[1]))
    if length < 1e-6:
        rel = pts - np.asarray(start, dtype=np.float32)
        return float(np.hypot(rel[:, 0], rel[:, 1]).max()) if len(pts) else 0.0
    rel = pts - np.asarray(start, dtype=np.float32)
    return float(np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]).max() / length)


def point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    denom = float(np.dot(ab, ab))
    rel = points - np.asarray(a, dtype=np.float32)
    if denom < 1e-8:
        return np.hypot(rel[:, 0], rel[:, 1]).astype(np.float32)
    t = np.clip((rel @ ab) / np.float32(denom), 0.0, 1.0)
    proj = np.asarray(a, dtype=np.float32) + t[:, None] * ab
    delta = points - proj
    return np.hypot(delta[:, 0], delta[:, 1]).astype(np.float32)


def sample_nearest(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    pts = as_f32(xy)
    height, width = image.shape[:2]
    x = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, height - 1)
    return image[y, x]


def sample_bilinear(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """xy 是 (N, 2) 的 x,y。越界夹到边缘。"""
    pts = as_f32(xy)
    height, width = image.shape[:2]
    x = np.clip(pts[:, 0], 0.0, width - 1.001)
    y = np.clip(pts[:, 1], 0.0, height - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    fx = (x - x0.astype(np.float32))[:, None]
    fy = (y - y0.astype(np.float32))[:, None]
    top = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    bot = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def eval_cubic(t: np.ndarray, p0, h1, h2, p3) -> np.ndarray:
    tt = np.asarray(t, dtype=np.float32)[:, None]
    u = 1.0 - tt
    return (
        (u * u * u) * np.asarray(p0, dtype=np.float32)
        + (3.0 * u * u * tt) * np.asarray(h1, dtype=np.float32)
        + (3.0 * u * tt * tt) * np.asarray(h2, dtype=np.float32)
        + (tt * tt * tt) * np.asarray(p3, dtype=np.float32)
    ).astype(np.float32)


def eval_cubic_derivs(t: np.ndarray, p0, h1, h2, p3):
    tt = np.asarray(t, dtype=np.float32)[:, None]
    u = 1.0 - tt
    p0 = np.asarray(p0, dtype=np.float32)
    h1 = np.asarray(h1, dtype=np.float32)
    h2 = np.asarray(h2, dtype=np.float32)
    p3 = np.asarray(p3, dtype=np.float32)
    point = (u * u * u) * p0 + (3 * u * u * tt) * h1 + (3 * u * tt * tt) * h2 + (tt * tt * tt) * p3
    d1 = 3 * u * u * (h1 - p0) + 6 * u * tt * (h2 - h1) + 3 * tt * tt * (p3 - h2)
    d2 = 6 * u * (h2 - 2 * h1 + p0) + 6 * tt * (p3 - 2 * h2 + h1)
    return point.astype(np.float32), d1.astype(np.float32), d2.astype(np.float32)


def sample_cubic_polyline(p0, h1, h2, p3, step: float = 1.15) -> np.ndarray:
    chord = float(np.linalg.norm(np.asarray(p3, dtype=np.float32) - np.asarray(p0, dtype=np.float32)))
    count = max(4, int(np.ceil(chord / max(step, 0.5))) + 1)
    ts = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return eval_cubic(ts, p0, h1, h2, p3)


def dedup_chain(pts, radii, cusp, tol: float = 0.35):
    pts = as_f32(pts)
    radii = np.asarray(radii, dtype=np.float32)
    cusp = np.asarray(cusp, dtype=bool)
    if len(pts) <= 2:
        return pts, radii, cusp
    steps = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    if float(steps.min()) >= tol and not np.any(cusp):
        return pts, radii, cusp
    keep = [0]
    for index in range(1, len(pts) - 1):
        if bool(cusp[index]) or float(np.hypot(*(pts[index] - pts[keep[-1]]))) >= tol:
            keep.append(index)
    keep.append(len(pts) - 1)
    chosen = np.asarray(keep, dtype=np.int32)
    return pts[chosen], radii[chosen], cusp[chosen]


def neighbor_table(ys: np.ndarray, xs: np.ndarray, height: int, width: int) -> np.ndarray:
    """链像素的 8 邻域索引表，-1 表示没有骨架邻居。"""
    count = int(len(ys))
    grid = np.full((height + 2, width + 2), -1, np.int32)
    if count:
        grid[ys + 1, xs + 1] = np.arange(count, dtype=np.int32)
    table = np.full((count, 8), -1, np.int32)
    for slot, (dy, dx) in enumerate(_OFFSETS):
        if count:
            table[:, slot] = grid[ys + 1 + dy, xs + 1 + dx]
    return table
