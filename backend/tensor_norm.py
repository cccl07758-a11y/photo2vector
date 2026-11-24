"""Stage 1B：彩色结构张量与跨动态范围除法归一化。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TensorFields:
    energy: np.ndarray
    energy_norm: np.ndarray
    coherence: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    L: np.ndarray
    a: np.ndarray
    b: np.ndarray


def structure_tensor(
    oklab: np.ndarray,
    params: dict,
    roi_mask: np.ndarray | None = None,
) -> TensorFields:
    """在去纹理 OKLab 上估计色彩边缘能量、相干度与法向/切向。支持传入亮度带 ROI 掩膜做带内独立标定。"""
    lab = np.ascontiguousarray(oklab, dtype=np.float32)
    chroma = float(params.get("chroma_weight", 1.0))
    w_ab = np.float32(chroma * 2.2)
    sigma_g = float(params.get("sigma_g", 1.35))
    sigma_norm = float(params.get("sigma_norm", 16.0))
    quietness = float(params.get("quietness", 0.5))
    contrast_norm = float(params.get("contrast_norm", 0.74))

    gx = cv2.Scharr(lab, cv2.CV_32F, 1, 0, borderType=cv2.BORDER_REFLECT) * np.float32(1.0 / 32.0)
    gy = cv2.Scharr(lab, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_REFLECT) * np.float32(1.0 / 32.0)
    weights = np.array([1.0, w_ab, w_ab], dtype=np.float32)
    gx *= weights
    gy *= weights

    jxx = np.sum(gx * gx, axis=2, dtype=np.float32)
    jxy = np.sum(gx * gy, axis=2, dtype=np.float32)
    jyy = np.sum(gy * gy, axis=2, dtype=np.float32)
    packed = cv2.merge([jxx, jxy, jyy])
    packed = cv2.GaussianBlur(packed, (0, 0), max(sigma_g, 0.3), borderType=cv2.BORDER_REFLECT)
    jxx, jxy, jyy = cv2.split(packed)

    trace = jxx + jyy
    diff = jxx - jyy
    disc = np.sqrt(np.maximum(diff * diff + 4.0 * jxy * jxy, 0.0))
    lambda1 = np.maximum(0.5 * (trace + disc), 0.0)
    lambda2 = np.maximum(0.5 * (trace - disc), 0.0)
    energy = np.sqrt(lambda1).astype(np.float32)
    coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)).astype(np.float32)
    coherence = np.clip(coherence, 0.0, 1.0)

    # 最大特征值对应的方向即法向。切向沿等值线。符号翻转不影响 ± 采样。
    angle = 0.5 * np.arctan2(2.0 * jxy, diff)
    nx = np.cos(angle).astype(np.float32)
    ny = np.sin(angle).astype(np.float32)
    tx = (-ny).astype(np.float32)
    ty = nx

    energy_norm = divisive_normalize(energy, sigma_norm, quietness, contrast_norm, roi_mask=roi_mask)
    return TensorFields(
        energy=energy,
        energy_norm=energy_norm,
        coherence=coherence,
        nx=nx,
        ny=ny,
        tx=tx,
        ty=ty,
        L=lab[..., 0],
        a=lab[..., 1],
        b=lab[..., 2],
    )


def divisive_normalize(
    energy: np.ndarray,
    sigma_norm: float,
    quietness: float,
    contrast_norm: float,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    """E / (σ_local + ε) 拉齐亮部与暗部，再用局部底噪信噪比按死真正的平坦区。

    若提供 roi_mask（例如暗部层或高光层优势区），分位数统计严格限制在带内像素上，
    彻底防止带外的极端反差（如骑士8%面积的金属高光、海滩3%面积的黑鞋）把本层真实轮廓压扁。
    """
    e = np.maximum(energy, 0.0).astype(np.float32)
    q = float(np.clip(quietness, 0.0, 1.0))
    c = float(np.clip(contrast_norm, 0.0, 1.0))

    sample_e = e[roi_mask] if (roi_mask is not None and int(np.count_nonzero(roi_mask)) >= 256) else e
    p25, p50, p75, p99 = _pcts(sample_e, (25.0, 50.0, 75.0, 99.0))
    if p99 < 1e-8:
        return np.zeros_like(e)

    e_global = np.clip(e / np.float32(p99 + 1e-8), 0.0, 1.0)
    local_ms = cv2.GaussianBlur(e * e, (0, 0), max(sigma_norm, 1.0), borderType=cv2.BORDER_REFLECT)
    sigma_loc = np.sqrt(local_ms + np.float32(1e-8))

    # 分母温和标定，弱对比区的真实阶跃在除法之后能显影，同时压住厚涂笔触底噪。
    eps_floor = (0.24 + 0.92 * q) * p50 + (0.06 + 0.25 * q) * max(p75 - p50, 0.0)
    eps_floor = max(float(eps_floor), 1e-6)
    e_div = e / (sigma_loc + np.float32(eps_floor))
    sample_div = e_div[roi_mask] if (roi_mask is not None and int(np.count_nonzero(roi_mask)) >= 256) else e_div
    div_hi = _pcts(sample_div, (99.0,))[0]
    e_div_n = np.clip(e_div / np.float32(div_hi + 1e-8), 0.0, 1.0)

    mixed = (1.0 - c) * e_global + c * e_div_n

    # 弱梯度分位数估计平坦区噪声底：不拿 p92 砍弱边，同时拦住厚涂底噪泛滥进滞后连接。
    noise_floor = max(float(p25), float(p50) * 0.52, 1e-6)
    snr = e / np.float32(noise_floor)
    snr_lo = 1.25 + 1.55 * q
    snr_hi = snr_lo + 1.25 + 1.65 * q
    gate = _smoothstep(snr_lo, snr_hi, snr)
    out = (mixed * gate).astype(np.float32)
    sample_out = out[roi_mask] if (roi_mask is not None and int(np.count_nonzero(roi_mask)) >= 256) else out
    hi = _pcts(sample_out, (99.0,))[0]
    if hi < 1e-8:
        return np.zeros_like(out)
    return np.clip(out / np.float32(hi), 0.0, 1.0).astype(np.float32)


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    span = np.float32(hi - lo)
    t = np.clip((x - np.float32(lo)) / span, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _pcts(arr: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    step = max(1, flat.size // 180_000)
    return np.percentile(flat[::step], list(qs)).astype(np.float64)
