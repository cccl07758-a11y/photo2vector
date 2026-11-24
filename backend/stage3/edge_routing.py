"""实边归线、虚边归色，以及 OKLab 同色系深色线。"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from color_space import oklab_to_srgb


def measure(
    paths: list[dict],
    lab: np.ndarray,
    saliency: np.ndarray,
    display_size: tuple[int, int],
) -> list[dict]:
    """沿阶段二路径测算模糊宽度、实度和较暗一侧的衬底色。坐标仍在显示分辨率。"""
    disp_w, disp_h = display_size
    height, width = lab.shape[:2]
    sx = width / max(disp_w, 1)
    sy = height / max(disp_h, 1)
    diagonal = math.hypot(disp_w, disp_h)
    prepared = []
    samples = []
    normals = []
    for path in paths:
        pts = _flatten(path.get("commands") or [], 4.0)
        if len(pts) < 2:
            continue
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if length < 4.0:
            continue
        closed = bool(path.get("closed"))
        sample = _subsample(pts, 36)
        prepared.append((path, pts, length, closed, sample))
        samples.append(sample)
        normals.append(_normals(sample, closed))
    stats = _profile_batch(samples, normals, lab, saliency, sx, sy) if prepared else []
    measured = []
    for (path, pts, length, closed, _sample), (sigma_blur, delta_e, s_mean, darker, side_l) in zip(prepared, stats):
        solidity = math.exp(-0.8 * sigma_blur) * float(np.clip(delta_e / 0.09, 0.0, 1.0)) * (0.35 + 0.65 * s_mean)
        if side_l[0] > 0.78 and side_l[1] > 0.78 and delta_e < 0.045 and sigma_blur > 1.5:
            solidity = 0.0
        solidity = float(np.clip(solidity, 0.0, 1.0))
        tier = str(path.get("tier") or "secondary")
        long_primary = tier == "primary" and length >= max(48.0, 0.055 * diagonal) and delta_e >= 0.05
        strong_edge = delta_e >= 0.12 and tier == "primary" and length >= 24.0
        poly = np.stack((pts[:, 0] * sx, pts[:, 1] * sy), axis=1).astype(np.float32)
        is_dam = bool(solidity >= 0.42 or long_primary or strong_edge)
        measured.append(
            {
                "commands": path.get("commands") or [],
                "anchors": path.get("anchors") or [],
                "tier": tier,
                "score": float(path.get("score") or 0.0),
                "stroke_width": float(path.get("stroke_width") or 1.6),
                "closed": closed,
                "length": length,
                "solidity": solidity,
                "sigma_blur": float(sigma_blur),
                "delta_e": float(delta_e),
                "s_mean": float(s_mean),
                "darker_lab": darker,
                "poly_work": poly if is_dam else None,
                "dam": is_dam,
            }
        )
    return measured


def select(measured: list[dict], gate: float, display_size: tuple[int, int]) -> list[dict]:
    diagonal = math.hypot(display_size[0], display_size[1])
    short_limit = max(36.0, 0.05 * diagonal)
    high_cut = max(0.50, float(gate) + 0.20)
    kept = []
    for item in measured:
        short = (not item["closed"]) and item["length"] < short_limit
        if item["solidity"] < gate or (item["s_mean"] < 0.20 and short):
            continue
        long_main = item["tier"] == "primary" and item["solidity"] >= 0.42 and item["length"] >= max(72.0, 0.10 * diagonal)
        tier = "primary" if item["solidity"] >= high_cut or long_main else "secondary"
        if tier == "secondary":
            opacity = 0.58 + 0.42 * float(np.clip((item["solidity"] - gate) / 0.22, 0.0, 1.0))
            width = max(1.05, item["stroke_width"] * 0.75)
        else:
            opacity = 1.0
            width = max(1.35, item["stroke_width"])
        kept.append({**item, "paint_tier": tier, "opacity": float(opacity), "paint_width": float(width)})
    return kept


def stroke_hex(lab: np.ndarray, mode: str) -> str:
    if mode == "black":
        return "#18181b"
    if mode == "none":
        return "#18181b"
    luma, aa, bb = (float(lab[0]), float(lab[1]), float(lab[2]))
    proposed = 0.16 + 0.38 * luma
    upper = max(0.06, luma - 0.12)
    lower = 0.06
    if upper < lower:
        upper = lower
    darkened = min(max(proposed, lower), upper)
    chroma = math.hypot(aa, bb)
    if chroma >= 0.035:
        hue = math.atan2(bb, aa)
        boosted = min(chroma * 1.12 + 0.016, 0.22)
        a2 = boosted * math.cos(hue)
        b2 = boosted * math.sin(hue)
    else:
        a2 = aa - 0.010
        b2 = bb - 0.012
    return lab_to_hex((darkened, a2, b2))


def lab_to_hex(lab) -> str:
    rgb = oklab_to_srgb(np.asarray(lab, dtype=np.float32).reshape(1, 1, 3))[0, 0]
    quant = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return f"#{int(quant[0]):02x}{int(quant[1]):02x}{int(quant[2]):02x}"


def _flatten(commands: list[dict], step: float) -> np.ndarray:
    points: list[np.ndarray] = []
    cursor = None
    start = None
    for cmd in commands:
        op = cmd.get("op")
        if op == "M" and cmd.get("p"):
            cursor = np.asarray(cmd["p"], dtype=np.float32)
            start = cursor.copy()
            points.append(cursor)
        elif op == "L" and cursor is not None and cmd.get("p"):
            end = np.asarray(cmd["p"], dtype=np.float32)
            _line(points, cursor, end, step)
            cursor = end
        elif op == "C" and cursor is not None and cmd.get("p") and cmd.get("h1") and cmd.get("h2"):
            end = np.asarray(cmd["p"], dtype=np.float32)
            h1 = np.asarray(cmd["h1"], dtype=np.float32)
            h2 = np.asarray(cmd["h2"], dtype=np.float32)
            span = float(np.linalg.norm(end - cursor) + np.linalg.norm(h1 - cursor) + np.linalg.norm(end - h2))
            count = max(2, int(math.ceil(span / max(step, 0.5) / 2.0)))
            ts = np.linspace(0.0, 1.0, count, dtype=np.float32)[1:]
            samples = _cubic(ts, cursor, h1, h2, end)
            points.extend(samples)
            cursor = end
        elif op == "Z" and cursor is not None and start is not None:
            _line(points, cursor, start, step)
            cursor = start.copy()
    if not points:
        return np.zeros((0, 2), np.float32)
    return np.asarray(points, dtype=np.float32)


def _line(points: list[np.ndarray], start: np.ndarray, end: np.ndarray, step: float) -> None:
    dist = float(np.linalg.norm(end - start))
    count = max(1, int(math.ceil(dist / max(step, 0.5))))
    for index in range(1, count + 1):
        points.append(start + (end - start) * (index / count))


def _cubic(ts: np.ndarray, p0, h1, h2, p1) -> np.ndarray:
    u = 1.0 - ts
    return (
        (u * u * u)[:, None] * p0
        + (3.0 * u * u * ts)[:, None] * h1
        + (3.0 * u * ts * ts)[:, None] * h2
        + (ts * ts * ts)[:, None] * p1
    ).astype(np.float32)


def _subsample(pts: np.ndarray, limit: int) -> np.ndarray:
    if len(pts) <= limit:
        return pts
    stride = int(math.ceil(len(pts) / limit))
    picked = pts[::stride]
    if not np.array_equal(picked[-1], pts[-1]):
        picked = np.vstack((picked, pts[-1:]))
    return picked


def _normals(pts: np.ndarray, closed: bool) -> np.ndarray:
    if closed and len(pts) >= 3:
        chain = np.vstack((pts[-1:], pts, pts[:1]))
        tang = np.gradient(chain, axis=0)[1:-1]
    else:
        tang = np.gradient(pts, axis=0)
    length = np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    tang = tang / length
    return np.stack((-tang[:, 1], tang[:, 0]), axis=1).astype(np.float32)


def _profile_batch(samples, normals, lab, saliency, sx, sy):
    if not samples:
        return []
    pts = np.concatenate(samples, axis=0)
    nrm = np.concatenate(normals, axis=0)
    offsets = np.linspace(-10.0, 10.0, 17, dtype=np.float32)
    disp = pts[:, None, :] + nrm[:, None, :] * offsets[None, :, None]
    luma = _bilinear(lab[..., 0], disp[..., 0] * sx, disp[..., 1] * sy)
    fine = gaussian_filter1d(luma, 0.8, axis=1, mode="nearest")
    coarse = gaussian_filter1d(luma, 2.2, axis=1, mode="nearest")
    spacing = float(offsets[1] - offsets[0])
    peak_fine = np.abs(np.gradient(fine, spacing, axis=1)).max(axis=1)
    peak_coarse = np.abs(np.gradient(coarse, spacing, axis=1)).max(axis=1)
    ratio = peak_fine / (peak_coarse + 1e-8)
    valid = (ratio > 1.05) & (peak_fine > 4e-4)
    sigma = np.full(len(pts), 4.0, np.float32)
    if np.any(valid):
        r2 = np.square(ratio[valid])
        num = 2.2**2 - r2 * (0.8**2)
        den = np.maximum(r2 - 1.0, 1e-4)
        sigma[valid] = np.sqrt(np.clip(num / den, 0.04, 36.0)).astype(np.float32)
        sigma[valid & (ratio > 2.7)] = 0.25
    side = 7.0
    left_xy = pts - nrm * side
    right_xy = pts + nrm * side
    left = _bilinear_lab(lab, left_xy[:, 0] * sx, left_xy[:, 1] * sy)
    right = _bilinear_lab(lab, right_xy[:, 0] * sx, right_xy[:, 1] * sy)
    delta = np.linalg.norm(left - right, axis=1)
    darker = np.where((left[:, 0] <= right[:, 0])[:, None], left, right)
    center_s = _bilinear(saliency, pts[:, 0] * sx, pts[:, 1] * sy)
    out = []
    cursor = 0
    for sample in samples:
        count = len(sample)
        sl = slice(cursor, cursor + count)
        cursor += count
        weights = np.clip(peak_fine[sl], 0.0, None)
        if float(weights.sum()) < 1e-6:
            sigma_blur = 2.8
        else:
            sigma_blur = _weighted_median(sigma[sl], weights)
        chunk_delta = delta[sl]
        delta_e = float(np.median(chunk_delta)) if len(chunk_delta) else 0.0
        out.append(
            (
                float(sigma_blur),
                delta_e,
                float(np.clip(np.mean(center_s[sl]), 0.0, 1.0)),
                darker[sl].mean(axis=0).astype(np.float32),
                (float(np.mean(left[sl, 0])), float(np.mean(right[sl, 0]))),
            )
        )
    return out


def _profile(pts, normals, lab, saliency, sx, sy):
    offsets = np.linspace(-10.0, 10.0, 21, dtype=np.float32)
    disp = pts[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    xs = disp[..., 0] * sx
    ys = disp[..., 1] * sy
    luma = _bilinear(lab[..., 0], xs, ys)
    fine = gaussian_filter1d(luma, 0.8, axis=1, mode="nearest")
    coarse = gaussian_filter1d(luma, 2.2, axis=1, mode="nearest")
    spacing = float(offsets[1] - offsets[0])
    d_fine = np.abs(np.gradient(fine, spacing, axis=1))
    d_coarse = np.abs(np.gradient(coarse, spacing, axis=1))
    peak_fine = d_fine.max(axis=1)
    peak_coarse = d_coarse.max(axis=1)
    ratio = peak_fine / (peak_coarse + 1e-8)
    valid = (ratio > 1.05) & (peak_fine > 4e-4)
    sigma = np.full(len(pts), 4.0, np.float32)
    if np.any(valid):
        r2 = ratio[valid] ** 2
        num = 2.2**2 - r2 * (0.8**2)
        den = r2 - 1.0
        sigma2 = np.clip(num / np.maximum(den, 1e-4), 0.04, 36.0)
        sigma[valid] = np.sqrt(sigma2).astype(np.float32)
        sigma[valid & (ratio > 2.7)] = 0.25
    weights = np.clip(peak_fine, 0.0, None)
    if float(weights.sum()) < 1e-6:
        sigma_blur = 2.8
    else:
        sigma_blur = _weighted_median(sigma, weights)

    side = 7.0
    left_xy = pts - normals * side
    right_xy = pts + normals * side
    left = _bilinear_lab(lab, left_xy[:, 0] * sx, left_xy[:, 1] * sy)
    right = _bilinear_lab(lab, right_xy[:, 0] * sx, right_xy[:, 1] * sy)
    delta = np.linalg.norm(left - right, axis=1)
    delta_e = float(np.median(delta)) if len(delta) else 0.0
    left_l = float(np.mean(left[:, 0])) if len(left) else 0.5
    right_l = float(np.mean(right[:, 0])) if len(right) else 0.5
    darker_samples = np.where((left[:, 0] <= right[:, 0])[:, None], left, right)
    darker = darker_samples.mean(axis=0).astype(np.float32)
    centers = _bilinear(saliency, pts[:, 0] * sx, pts[:, 1] * sy)
    s_mean = float(np.clip(np.mean(centers), 0.0, 1.0))
    return float(sigma_blur), delta_e, s_mean, darker, (left_l, right_l)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    seq = values[order]
    acc = np.cumsum(weights[order])
    if acc[-1] <= 0:
        return float(np.median(values))
    index = int(np.searchsorted(acc, 0.5 * acc[-1]))
    return float(seq[min(index, len(seq) - 1)])


def _bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(xs, 0.0, width - 1.001)
    y = np.clip(ys, 0.0, height - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bot = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _bilinear_lab(lab: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    channels = [_bilinear(lab[..., index], xs, ys) for index in range(3)]
    return np.stack(channels, axis=-1).astype(np.float32)


def dam_polylines(measured: list[dict]) -> list[tuple[np.ndarray, bool]]:
    lines = []
    for item in measured:
        if not item.get("dam"):
            continue
        poly = item.get("poly_work")
        if poly is None or len(poly) < 2:
            continue
        lines.append((poly, bool(item["closed"])))
    return lines


def raster_dams(shape: tuple[int, int], lines: list[tuple[np.ndarray, bool]]) -> np.ndarray:
    height, width = shape
    dam = np.zeros((height, width), np.uint8)
    for poly, closed in lines:
        pts = np.round(poly).astype(np.int32)
        if len(pts) < 2:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.polylines(dam, [pts.reshape(-1, 1, 2)], closed, 255, 2, cv2.LINE_8)
    dam = cv2.morphologyEx(dam, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return dam > 0
