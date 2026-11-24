"""Stage 1D：双阈值滞后连接、连通域净网、可选 1px 骨架。"""

from __future__ import annotations

import cv2
import numpy as np
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology

# 0.24 把该函数从 morphology 挪到了 filters，两边都试，调用的是同一个滞后连接。
apply_hysteresis_threshold = getattr(
    sk_morphology,
    "apply_hysteresis_threshold",
    sk_filters.apply_hysteresis_threshold,
)


def finalize(soft_ink: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    mode = params.get("threshold_mode", "hysteresis")
    low = float(params["thresh_low"])
    high = float(params["thresh_high"])
    if low > high:
        low, high = high, low
    min_area = int(params["min_component_size"])
    min_span = float(params["min_span"])
    thin = bool(params.get("thin_1px", False))
    # 拷贝一份，避免滞后阈值或后续写入污染管线缓存里的软墨线。
    soft = np.clip(np.array(soft_ink, dtype=np.float32, copy=True), 0.0, 1.0)

    if mode == "hard":
        mask = soft >= high
        mask, count = filter_components(mask, min_area, min_span)
        if thin:
            mask = skeletonize(mask)
            count = _count_mask(mask)
        ink = mask.astype(np.float32)
    elif mode == "soft":
        ink, count = _soft_clean(soft, max(low, 0.0), min_area, min_span)
        if thin:
            mask = skeletonize(ink >= max(low, 1e-3))
            count = _count_mask(mask)
            ink = mask.astype(np.float32)
    else:
        # 滞后连接：低阈值像素只有连到高阈值主干上才保留。
        seed_low = max(low, 1e-3)
        mask = apply_hysteresis_threshold(soft, seed_low, max(high, seed_low))
        mask = np.asarray(mask, dtype=bool)
        mask, count = filter_components(mask, min_area, min_span)
        if thin:
            mask = skeletonize(mask)
            count = _count_mask(mask)
            ink = mask.astype(np.float32)
        else:
            ink = _antialias(mask, soft)

    line_ratio = float(np.mean(ink >= 0.35))
    info = {"components": int(count), "line_ratio": line_ratio}
    return np.ascontiguousarray(ink, dtype=np.float32), info


def filter_components(mask: np.ndarray, min_area: int, min_span: float) -> tuple[np.ndarray, int]:
    """抹掉面积和包围盒跨度都过小的孤立墨点。长细线因为跨度够，会被留下。"""
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    if min_area <= 0 and min_span <= 0:
        count, _ = cv2.connectedComponents(binary, connectivity=8)
        return mask.astype(bool), max(0, int(count) - 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool), 0
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    span = np.sqrt(widths * widths + heights * heights)
    a_thr = float(max(min_area, 0))
    s_thr = float(max(min_span, 0.0))
    # 1) 极小碎屑（面积 < 0.35 * min_area 且跨度不够长）一律清除；
    # 2) 团块状噪点（跨度 < min_span）清除，但跨度 >= 1.25 * min_span 的细长轮廓线予以保留。
    tiny_dust = (areas < max(8.0, a_thr * 0.35)) & (span < max(6.0, s_thr * 0.75))
    compact_speck = (areas < a_thr) & (span < s_thr)
    short_blob = (span < s_thr * 0.65) & (areas < a_thr * 1.6)
    drop = tiny_dust | compact_speck | short_blob
    drop[0] = False
    removed = int(np.count_nonzero(drop))
    if removed == 0:
        return mask.astype(bool), count - 1
    keep = np.ones(count, np.uint8)
    keep[drop] = 0
    keep[0] = 0
    kept = keep[labels] > 0
    return kept, count - 1 - removed


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 单像素中心线，给阶段二的曲线拟合做预览。"""
    u8 = np.ascontiguousarray(mask.astype(np.uint8) * 255)
    if not np.any(u8):
        return np.zeros_like(mask, dtype=bool)
    thinned = cv2.ximgproc.thinning(u8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return thinned > 0


def _soft_clean(soft: np.ndarray, low: float, min_area: int, min_span: float) -> tuple[np.ndarray, int]:
    support = soft >= max(low, 1e-3)
    support, count = filter_components(support, min_area, min_span)
    # 被删掉的连通域整块清零，其余灰阶保留。
    killed = (soft >= max(low, 1e-3)) & ~support
    out = soft.copy()
    out[killed] = 0.0
    return out, count


def _antialias(mask: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """主干保持实色，边缘 1px 用软墨线做灰度过渡，避免滞后阈值的锯齿台阶。"""
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_u8, kernel) > 0
    boundary = mask & ~eroded
    ring = (cv2.dilate(mask_u8, kernel) > 0) & ~mask
    out = np.zeros_like(soft, dtype=np.float32)
    out[eroded] = 1.0
    # 1px 细线没有内核，整段落在 boundary 上，用软墨线保住深浅。
    out[boundary] = np.clip(0.58 + 0.42 * soft[boundary], 0.0, 1.0)
    out[ring] = np.clip(soft[ring] * 0.45, 0.0, 0.42)
    return out


def _count_mask(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(0, int(count) - 1)
