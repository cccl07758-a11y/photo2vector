"""P0：面积开闭 + Meyer leveling，把麻点拉平且不移动大轮廓。"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import area_closing, area_opening

from color_space import oklab_to_srgb

_LEVELS = 2048


def flatten(lab: np.ndarray, saliency: np.ndarray, dust_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回平涂 OKLab、锁定后的明度 L*、以及 sRGB 预览。"""
    lab = np.ascontiguousarray(lab, dtype=np.float32)
    saliency = np.clip(np.asarray(saliency, dtype=np.float32), 0.0, 1.0)
    if saliency.shape[:2] != lab.shape[:2]:
        saliency = cv2.resize(saliency, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)
    luma = np.clip(lab[..., 0], 0.0, 1.0)
    height, width = luma.shape
    area_px = max(1, int(round(float(dust_ratio) * height * width)))
    area_px = int(np.clip(area_px, 1, max(1, int(0.012 * height * width))))
    quant = np.clip(np.round(luma * _LEVELS), 0, _LEVELS).astype(np.uint16)
    opened = area_opening(quant, area_threshold=area_px, connectivity=1)
    closed = area_closing(opened, area_threshold=area_px, connectivity=1)
    l_aoc = closed.astype(np.float32) / np.float32(_LEVELS)
    l_star = _meyer(l_aoc)
    filled = _propagate_chroma(lab, l_star, saliency)
    preview = _srgb(filled)
    return filled, l_star, preview


def _meyer(reference: np.ndarray) -> np.ndarray:
    """Meyer 2004：平滑标记在参考图的测地上下界之间来回夹紧，直到跳变锁死。"""
    quant = np.clip(np.round(reference * _LEVELS), 0, _LEVELS).astype(np.uint16)
    marker = cv2.medianBlur(quant, 5).astype(np.float32) / np.float32(_LEVELS)
    guide = np.ascontiguousarray(reference, dtype=np.float32)
    current = np.ascontiguousarray(marker, dtype=np.float32)
    kernel = np.ones((3, 3), np.uint8)
    step = np.float32(1.0 / _LEVELS)
    for _ in range(40):
        dilated = cv2.dilate(current, kernel)
        eroded = cv2.erode(current, kernel)
        nxt = np.maximum(np.minimum(guide, dilated), eroded)
        if float(np.max(np.abs(nxt - current))) <= float(step):
            current = nxt
            break
        current = nxt
    return current.astype(np.float32)


def _propagate_chroma(lab: np.ndarray, l_star: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    luma = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    changed = np.abs(l_star - luma) > np.float32(0.014)
    vivid = (saliency > np.float32(0.45)) & (chroma > np.float32(0.11))
    exempt = _large_components(vivid, 80)
    changed &= ~exempt
    if np.any(changed) and np.any(~changed):
        _dist, indices = distance_transform_edt(changed, return_indices=True)
        taken_a = lab[..., 1][indices[0], indices[1]]
        taken_b = lab[..., 2][indices[0], indices[1]]
    else:
        taken_a = lab[..., 1]
        taken_b = lab[..., 2]
    flat = lab.copy()
    flat[..., 0] = l_star
    flat[..., 1] = np.where(changed, taken_a, lab[..., 1])
    flat[..., 2] = np.where(changed, taken_b, lab[..., 2])
    flat[..., 1:] = _guide_chroma(l_star, flat[..., 1:])
    flat[exempt] = lab[exempt]
    flat[..., 0] = np.clip(flat[..., 0], 0.0, 1.0)
    return flat.astype(np.float32)


def _large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    if count <= 1:
        return np.zeros(mask.shape, np.bool_)
    areas = np.bincount(labels.ravel(), minlength=count)
    keep = areas >= int(min_area)
    keep[0] = False
    return keep[labels]


def _guide_chroma(guide: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    guided = np.empty_like(chroma)
    source_guide = np.ascontiguousarray(guide, dtype=np.float32)
    ximg = getattr(cv2, "ximgproc", None)
    for channel in range(2):
        src = np.ascontiguousarray(chroma[..., channel], dtype=np.float32)
        if ximg is not None and hasattr(ximg, "guidedFilter"):
            try:
                guided[..., channel] = ximg.guidedFilter(source_guide, src, 2, 2.5e-4)
                continue
            except cv2.error:
                pass
        guided[..., channel] = cv2.bilateralFilter(src, 5, 0.03, 3)
    return guided.astype(np.float32)


def _srgb(lab: np.ndarray) -> np.ndarray:
    rgb = oklab_to_srgb(lab)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
