"""无 AI 主体概率场。Boundary Connectivity（Zhu CVPR 2014）+ 色散布度（Cheng 2013）。"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

from color_space import oklab_to_srgb


def compute(
    lab_work: np.ndarray,
    w_tex: np.ndarray | None = None,
    energy_norm: np.ndarray | None = None,
) -> np.ndarray:
    """返回与 lab_work 同分辨率的 S_fg，范围 [0, 1]。"""
    lab_work = np.ascontiguousarray(lab_work, dtype=np.float32)
    height, width = lab_work.shape[:2]
    small, scale = _resize_long(lab_work, 480)
    labels = _segment(small, 310, 280, 340)
    means, xy, counts = _stats(small, labels)
    count = int(means.shape[0])
    if count < 2:
        return np.full((height, width), 0.35, np.float32)

    colors = means * np.float32(100.0)
    d_app = _pairwise(colors)
    edges = _adjacency(labels)
    cuts = _cut_distance(small, labels)
    s_map = _solve(colors, xy, d_app, edges, cuts, labels, small, w_tex, energy_norm, scale)
    filtered = _guide(small, s_map)
    full = cv2.resize(filtered, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(full, 0.0, 1.0).astype(np.float32)


def _resize_long(lab: np.ndarray, longest: int) -> tuple[np.ndarray, float]:
    height, width = lab.shape[:2]
    long_side = max(height, width)
    if long_side <= longest:
        return lab, 1.0
    scale = longest / float(long_side)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(lab, size, interpolation=cv2.INTER_AREA), scale


def _segment(lab: np.ndarray, target: int, lo: int, hi: int, iterations: int = 2) -> np.ndarray:
    rgb = np.clip(np.round(oklab_to_srgb(lab) * 255.0), 0, 255).astype(np.uint8)
    labels = _slic(rgb, target, lo, hi, iterations)
    if labels is None or int(labels.max()) < 1:
        labels = _watershed(rgb, target)
    return _compact(labels)


def _slic(rgb: np.ndarray, target: int, lo: int, hi: int, iterations: int = 2) -> np.ndarray | None:
    ximg = getattr(cv2, "ximgproc", None)
    if ximg is None:
        return None
    algo = getattr(ximg, "SLICO", getattr(ximg, "SLIC", None))
    if algo is None:
        return None
    height, width = rgb.shape[:2]
    region = max(8, int(round(math.sqrt(height * width / max(target, 1)))))
    bgr = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    try:
        engine = ximg.createSuperpixelSLIC(bgr, int(algo), int(region), 12.0)
        engine.iterate(max(1, int(iterations)))
        labels = engine.getLabels().astype(np.int32)
        count = int(engine.getNumberOfSuperpixels())
    except cv2.error:
        return None
    if count <= 1:
        return None
    _ = (lo, hi)
    return labels


def _watershed(rgb: np.ndarray, target: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    cols = max(1, int(round(math.sqrt(target * width / max(height, 1)))))
    rows = max(1, int(round(target / max(cols, 1))))
    markers = np.zeros((height, width), np.int32)
    seed = 1
    for row in range(rows):
        y = min(height - 1, int((row + 0.5) * height / rows))
        for col in range(cols):
            x = min(width - 1, int((col + 0.5) * width / cols))
            markers[y, x] = seed
            seed += 1
    marks = markers.copy()
    cv2.watershed(np.ascontiguousarray(rgb), marks)
    marks[marks <= 0] = 0
    return _fill_unknown(marks)


def _fill_unknown(labels: np.ndarray) -> np.ndarray:
    filled = labels.astype(np.int32, copy=True)
    for _ in range(8):
        unknown = filled <= 0
        if not np.any(unknown):
            break
        acc = np.full(filled.shape, -1, np.int32)
        shifted = (
            _shift(filled, 1, 0),
            _shift(filled, -1, 0),
            _shift(filled, 0, 1),
            _shift(filled, 0, -1),
        )
        for item in shifted:
            take = unknown & (acc < 0) & (item > 0)
            acc[take] = item[take]
        if not np.any(acc > 0):
            break
        hit = acc > 0
        filled[hit] = acc[hit]
    filled[filled <= 0] = 1
    return filled


def _shift(labels: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.full(labels.shape, -1, np.int32)
    if dy > 0:
        out[dy:, :] = labels[:-dy, :]
    elif dy < 0:
        out[:dy, :] = labels[-dy:, :]
    if dx > 0:
        moved = out.copy()
        moved[:, dx:] = out[:, :-dx]
        out = moved
    elif dx < 0:
        moved = out.copy()
        moved[:, :dx] = out[:, -dx:]
        out = moved
    if dy == 0 and dx > 0:
        out[:, dx:] = labels[:, :-dx]
        out[:, :dx] = -1
    elif dy == 0 and dx < 0:
        out[:, :dx] = labels[:, -dx:]
        out[:, dx:] = -1
    elif dx == 0 and dy > 0:
        out[dy:, :] = labels[:-dy, :]
        out[:dy, :] = -1
    elif dx == 0 and dy < 0:
        out[:dy, :] = labels[-dy:, :]
        out[dy:, :] = -1
    return out


def _compact(labels: np.ndarray) -> np.ndarray:
    _uniq, inverse = np.unique(labels.ravel(), return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def _stats(lab: np.ndarray, labels: np.ndarray):
    count = int(labels.max()) + 1
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=count).astype(np.float64)
    sums = np.stack(
        [np.bincount(flat, weights=lab[..., channel].ravel().astype(np.float64), minlength=count) for channel in range(3)],
        axis=1,
    )
    means = (sums / np.maximum(areas, 1.0)[:, None]).astype(np.float32)
    height, width = labels.shape
    ys, xs = np.mgrid[0:height, 0:width]
    xn = ((xs.ravel() + 0.5) / max(width, 1)).astype(np.float64)
    yn = ((ys.ravel() + 0.5) / max(height, 1)).astype(np.float64)
    sumx = np.bincount(flat, weights=xn, minlength=count)
    sumy = np.bincount(flat, weights=yn, minlength=count)
    xy = np.stack((sumx / np.maximum(areas, 1.0), sumy / np.maximum(areas, 1.0)), axis=1).astype(np.float32)
    return means, xy, areas


def _pairwise(colors: np.ndarray) -> np.ndarray:
    gram = colors.astype(np.float32)
    square = np.sum(gram * gram, axis=1, dtype=np.float32)
    dist = square[:, None] + square[None, :] - np.float32(2.0) * (gram @ gram.T)
    np.maximum(dist, 0.0, out=dist)
    np.sqrt(dist, out=dist)
    return dist


def _adjacency(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = int(labels.max()) + 1
    keys = np.concatenate(
        (
            _edge_keys(labels[:, :-1], labels[:, 1:], count),
            _edge_keys(labels[:-1, :], labels[1:, :], count),
        )
    )
    if keys.size == 0:
        return np.zeros((0, 2), np.int32), np.zeros(0, np.int32)
    uniq = np.unique(keys)
    left = (uniq // count).astype(np.int32)
    right = (uniq % count).astype(np.int32)
    return np.stack((left, right), axis=1), uniq


def _edge_keys(left: np.ndarray, right: np.ndarray, count: int) -> np.ndarray:
    mask = left != right
    if not np.any(mask):
        return np.zeros(0, np.int64)
    aa = left[mask].astype(np.int64)
    bb = right[mask].astype(np.int64)
    lo = np.minimum(aa, bb)
    hi = np.maximum(aa, bb)
    return lo * count + hi


def _cut_distance(lab: np.ndarray, labels: np.ndarray) -> dict[tuple[int, int], float]:
    """邻接超像素在共享边上的像素色差。均值差会漏掉细高光轮廓，边界本身的色差不会。"""
    count = int(labels.max()) + 1

    def pack(left, right, dist):
        mask = left != right
        if not np.any(mask):
            return np.zeros(0, np.int64), np.zeros(0, np.float64)
        aa = left[mask].astype(np.int64)
        bb = right[mask].astype(np.int64)
        lo = np.minimum(aa, bb)
        hi = np.maximum(aa, bb)
        return lo * count + hi, dist[mask].astype(np.float64)

    horizontal = np.linalg.norm(lab[:, :-1] - lab[:, 1:], axis=2)
    vertical = np.linalg.norm(lab[:-1, :] - lab[1:, :], axis=2)
    keys, dists = pack(labels[:, :-1], labels[:, 1:], horizontal)
    keys2, dists2 = pack(labels[:-1, :], labels[1:, :], vertical)
    key = np.concatenate((keys, keys2))
    dist = np.concatenate((dists, dists2))
    if key.size == 0:
        return {}
    uniq, inverse = np.unique(key, return_inverse=True)
    mean = np.bincount(inverse, weights=dist) / np.maximum(np.bincount(inverse), 1)
    return {(int(item // count), int(item % count)): float(value) for item, value in zip(uniq, mean)}


def _solve(colors, xy, d_app, edges, cuts, labels, lab, w_tex, energy_norm, scale):
    count = colors.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    edge_ij = edges[0]
    for i, j in edge_ij:
        pixel = cuts.get((int(i), int(j)), 0.0) * 100.0
        weight = max(float(d_app[i, j]), pixel, 1e-3)
        rows.extend((int(i), int(j)))
        cols.extend((int(j), int(i)))
        weights.extend((weight, weight))
    boundary = _boundary_ids(labels)
    if boundary.size >= 2:
        sub = d_app[np.ix_(boundary, boundary)]
        ii, jj = np.triu_indices(boundary.size, 1)
        close = sub[ii, jj] < 16.0
        for a, b, dist in zip(boundary[ii[close]], boundary[jj[close]], sub[ii, jj][close]):
            # 同色边界才互连：蓝砖、暗拱廊、青底可以绕框连通，贴边肤色不会抄近路。
            weight = max(float(dist), 1e-3)
            rows.extend((int(a), int(b)))
            cols.extend((int(b), int(a)))
            weights.extend((weight, weight))
    if weights:
        graph = csr_matrix((weights, (rows, cols)), shape=(count, count), dtype=np.float64)
    else:
        graph = csr_matrix((count, count), dtype=np.float64)
    geo = dijkstra(graph, directed=True)
    geo = np.where(np.isfinite(geo), geo, 1.0e6)
    similar = np.exp(-(geo * geo) / (2.0 * 100.0)).astype(np.float64)
    area = similar.sum(axis=1)
    if boundary.size:
        length = similar[:, boundary].sum(axis=1)
    else:
        length = np.zeros(count, np.float64)
    bnd = length / np.sqrt(area + 1e-6)
    w_bg = 1.0 - np.exp(-(bnd * bnd) / 2.0)

    sigma_pos = 0.25
    delta = xy.astype(np.float32)
    pos = (delta[:, None, 0] - delta[None, :, 0]) ** 2 + (delta[:, None, 1] - delta[None, :, 1]) ** 2
    spatial = np.exp(-pos / (2.0 * sigma_pos * sigma_pos))
    contrast = (d_app.astype(np.float32) * spatial * w_bg.astype(np.float32)[None, :]).sum(axis=1)
    contrast /= float(np.max(contrast) + 1e-6)

    color_sq = d_app.astype(np.float32) ** 2
    sim = np.exp(-color_sq / (2.0 * 8.0 * 8.0)).astype(np.float32)
    sim_sum = sim.sum(axis=1, keepdims=True) + np.float32(1e-6)
    center = (sim @ xy.astype(np.float32)) / sim_sum
    spread_dist = xy.astype(np.float32)[None, :, :] - center[:, None, :]
    variance = (sim * np.sum(spread_dist * spread_dist, axis=2)).sum(axis=1) / np.maximum(sim.sum(axis=1), 1e-6)
    low, high = np.percentile(variance, [20, 85])
    if float(high - low) < 1e-4:
        compact = np.ones(count, np.float32)
    else:
        compact = np.clip((high - variance) / (high - low), 0.0, 1.0).astype(np.float32)
    w_fg = np.clip(contrast * (0.25 + 0.75 * compact), 0.0, 1.0).astype(np.float64)

    affinity_r: list[int] = []
    affinity_c: list[int] = []
    affinity_w: list[float] = []
    for i, j in edge_ij:
        weight = math.exp(-(float(d_app[i, j]) ** 2) / (2.0 * 12.0 * 12.0))
        affinity_r.extend((int(i), int(j)))
        affinity_c.extend((int(j), int(i)))
        affinity_w.extend((weight, weight))
    if affinity_w:
        affinity = csr_matrix((affinity_w, (affinity_r, affinity_c)), shape=(count, count), dtype=np.float64)
    else:
        affinity = csr_matrix((count, count), dtype=np.float64)
    degree = np.asarray(affinity.sum(axis=1)).ravel()
    data = 2.4 * w_bg + 5.5 * w_fg + 1e-3
    matrix = diags(degree + data) - affinity
    try:
        solved = spsolve(matrix.tocsc(), 5.5 * w_fg)
    except Exception:
        solved = np.linalg.solve(matrix.toarray(), 5.5 * w_fg)
    solved = np.clip(np.nan_to_num(solved, nan=0.0), 0.0, 1.0)

    clarity = _clarity(lab, labels, count, energy_norm, scale)
    center_bias = np.exp(-((xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2) / (2.0 * 0.36**2))
    solved = np.clip(solved * (0.86 + 0.14 * center_bias) + 0.05 * clarity * (1.0 - w_bg), 0.0, 1.0)
    if w_tex is not None:
        tex = _resize_to(np.asarray(w_tex, dtype=np.float32), labels.shape)
        tex_mean = _mean_of(tex, labels, count)
        solved = np.clip(solved * (1.0 - 0.16 * tex_mean * w_bg), 0.0, 1.0)
    return solved.astype(np.float32)[labels]


def _grow_foreground(solved: np.ndarray, d_app: np.ndarray, cuts: dict, edges: np.ndarray) -> np.ndarray:
    """从高对比种子沿弱边界长进同一材质，在实轮廓处停下。"""
    values = solved.copy()
    active = values > 0.50
    if not np.any(active):
        return values
    for _ in range(8):
        grown = False
        for i, j in edges:
            left, right = int(i), int(j)
            if bool(active[left]) == bool(active[right]):
                continue
            src, dst = (left, right) if active[left] else (right, left)
            key = (left, right) if left < right else (right, left)
            if cuts.get(key, 1.0) >= 0.075 or float(d_app[left, right]) >= 22.0:
                continue
            values[dst] = max(float(values[dst]), 0.52)
            active[dst] = True
            grown = True
            _ = src
        if not grown:
            break
    return values


def _boundary_ids(labels: np.ndarray) -> np.ndarray:
    mask = np.zeros(labels.shape, np.bool_)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return np.unique(labels[mask]).astype(np.int32)


def _clarity(lab, labels, count, energy_norm, scale) -> np.ndarray:
    luma = lab[..., 0]
    fine = _grad(luma)
    blur = cv2.GaussianBlur(luma, (0, 0), 2.4)
    coarse = _grad(blur)
    ratio = fine / (coarse + np.float32(0.015))
    mean = _mean_of(ratio, labels, count)
    if energy_norm is not None:
        energy = _resize_to(np.asarray(energy_norm, dtype=np.float32), labels.shape)
        energy_mean = _mean_of(energy, labels, count)
        peak = float(np.percentile(energy_mean, 95)) + 1e-6
        mean = 0.55 * mean + 0.45 * (energy_mean / peak)
    peak = float(np.percentile(mean, 95)) + 1e-6
    _ = scale
    return np.clip(mean / peak, 0.0, 1.0).astype(np.float64)


def _grad(image: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(image, cv2.CV_32F, 1, 0) * np.float32(1.0 / 32.0)
    gy = cv2.Scharr(image, cv2.CV_32F, 0, 1) * np.float32(1.0 / 32.0)
    return cv2.magnitude(gx, gy)


def _mean_of(image: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    flat = labels.ravel()
    acc = np.bincount(flat, weights=image.ravel().astype(np.float64), minlength=count)
    area = np.bincount(flat, minlength=count).astype(np.float64)
    return acc / np.maximum(area, 1.0)


def _resize_to(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.ndim == 3:
        image = image[..., 0]
    if image.shape[:2] == shape:
        return image.astype(np.float32, copy=False)
    return cv2.resize(image.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _guide(lab: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    guide = np.clip(oklab_to_srgb(lab), 0.0, 1.0).astype(np.float32)
    src = np.clip(saliency, 0.0, 1.0).astype(np.float32)
    ximg = getattr(cv2, "ximgproc", None)
    if ximg is not None and hasattr(ximg, "guidedFilter"):
        try:
            smoothed = ximg.guidedFilter(guide, src, 4, 2e-3)
            return np.clip(smoothed, 0.0, 1.0).astype(np.float32)
        except cv2.error:
            pass
    return cv2.GaussianBlur(src, (0, 0), 1.6)
