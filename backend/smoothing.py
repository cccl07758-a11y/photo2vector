"""Stage 1A：前置去纹理、保结构滤波。"""

from __future__ import annotations

import os

import cv2
import numpy as np
from scipy import fft

from color_space import oklab_to_srgb, srgb_to_oklab

_FFT_WORKERS = min(8, os.cpu_count() or 4)


def _spatial_sigma(texture_scale: float) -> float:
    t = float(np.clip(texture_scale, 0.0, 10.0)) / 10.0
    return 1.5 + (10.0 - 1.5) * t


def _lambda_l0(texture_scale: float) -> float:
    t = float(np.clip(texture_scale, 0.0, 10.0)) / 10.0
    return 0.002 + (0.035 - 0.002) * t


def remove_texture(
    rgb_u8: np.ndarray,
    method: str,
    texture_scale: float,
    w_tex_256: np.ndarray | None = None,
    spatial_adaptive: bool = False,
    spatial_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (去纹理 sRGB uint8 预览, OKLab float32)。

    L0 / RTV 在 OKLab 上求解，预览经精确逆变换回到 sRGB。
    双边与滚动引导走 OpenCV 的 uint8 实现，再转入 OKLab。
    分区模式在 OKLab 残差里把重度结果和极轻基底按 W_tex 融合，避免 RGB 混合光晕。
    """
    method = (method or "none").lower()
    scale = float(texture_scale)
    if method == "none" or scale <= 1e-6:
        rgb_f = rgb_u8.astype(np.float32) / 255.0
        return rgb_u8, srgb_to_oklab(rgb_f)
    if _use_spatial(method, scale, w_tex_256, spatial_adaptive, spatial_strength):
        return _remove_spatial(rgb_u8, method, scale, w_tex_256, float(spatial_strength))
    return _remove_global(rgb_u8, method, scale)


def _use_spatial(method, scale, w_tex_256, spatial_adaptive, spatial_strength) -> bool:
    return (
        bool(spatial_adaptive)
        and float(spatial_strength) > 1e-6
        and method not in ("", "none")
        and float(scale) > 1e-6
        and w_tex_256 is not None
    )


def _remove_global(rgb_u8: np.ndarray, method: str, scale: float) -> tuple[np.ndarray, np.ndarray]:
    lab_s = _heavy_lab(rgb_u8, None, method, scale)
    preview = _u8_from_float(oklab_to_srgb(lab_s))
    return preview, lab_s


def _remove_spatial(
    rgb_u8: np.ndarray,
    method: str,
    scale: float,
    w_tex_256: np.ndarray,
    spatial_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    lab = srgb_to_oklab(rgb_u8.astype(np.float32) / 255.0)
    lab_light = _light_base(lab)
    lab_heavy = _heavy_lab(rgb_u8, lab, method, scale)
    height, width = lab.shape[:2]
    weight = cv2.resize(np.asarray(w_tex_256, dtype=np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    weight = np.clip(weight, 0.0, 1.0).astype(np.float32)
    # W=1 全用重度去纹理；W=0 时按 spatial_strength 留在轻度基底上。
    heavy_weight = np.clip(1.0 - spatial_strength * (1.0 - weight), 0.0, 1.0).astype(np.float32)
    lab_out = lab_light + heavy_weight[..., None] * (lab_heavy - lab_light)
    lab_out = np.ascontiguousarray(lab_out, dtype=np.float32)
    preview = _u8_from_float(oklab_to_srgb(lab_out))
    return preview, lab_out


def _heavy_lab(rgb_u8: np.ndarray, lab: np.ndarray | None, method: str, scale: float) -> np.ndarray:
    if method == "bilateral":
        out = _bilateral(rgb_u8, scale)
        return srgb_to_oklab(out.astype(np.float32) / 255.0)
    if method == "rgf":
        out = _rolling_guidance(rgb_u8, scale)
        return srgb_to_oklab(out.astype(np.float32) / 255.0)
    if lab is None:
        lab = srgb_to_oklab(rgb_u8.astype(np.float32) / 255.0)
    if method == "l0":
        return l0_smooth(lab, _lambda_l0(scale))
    if method == "rtv":
        return rtv_smooth(lab, scale)
    raise ValueError(f"未知 texture_method: {method}")


def _light_base(lab: np.ndarray) -> np.ndarray:
    """半径 1 的引导滤波，只抹微噪，卷发、五官和文字都留着。"""
    src = np.ascontiguousarray(lab, dtype=np.float32)
    guide = np.ascontiguousarray(src[..., 0], dtype=np.float32)
    try:
        out = cv2.ximgproc.guidedFilter(guide, src, radius=1, eps=4e-4)
        if out is not None and out.shape == src.shape:
            return np.ascontiguousarray(out, dtype=np.float32)
    except cv2.error:
        pass
    return cv2.GaussianBlur(src, (0, 0), 0.8)


def _u8_from_float(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def _bilateral(rgb_u8: np.ndarray, texture_scale: float) -> np.ndarray:
    sigma_s = _spatial_sigma(texture_scale)
    # uint8 的 sigmaColor 模拟 PS 特殊模糊的阈值半径。
    sigma_c = 18.0 + texture_scale * 8.0
    out = cv2.bilateralFilter(rgb_u8, d=0, sigmaColor=float(sigma_c), sigmaSpace=float(sigma_s))
    return out


def _rolling_guidance(rgb_u8: np.ndarray, texture_scale: float) -> np.ndarray:
    sigma_s = _spatial_sigma(texture_scale)
    sigma_c = 12.0 + texture_scale * 6.5
    try:
        out = cv2.ximgproc.rollingGuidanceFilter(
            rgb_u8,
            None,
            -1,
            float(sigma_c),
            float(sigma_s),
            4,
        )
        if out is not None and out.shape == rgb_u8.shape:
            return out
    except cv2.error:
        pass
    return _rolling_guidance_guided(rgb_u8, sigma_s, texture_scale)


def _rolling_guidance_guided(rgb_u8: np.ndarray, sigma_s: float, texture_scale: float) -> np.ndarray:
    """cv2.ximgproc.guidedFilter 的 4 次滚动引导，作为 RGF 的数值稳定后备。"""
    src = rgb_u8.astype(np.float32) / 255.0
    radius = max(1, int(round(sigma_s * 2.0)))
    eps = (0.012 + 0.004 * float(texture_scale)) ** 2
    guide = cv2.GaussianBlur(src, (0, 0), sigma_s)
    for _ in range(4):
        guide = cv2.ximgproc.guidedFilter(guide, src, radius, float(eps))
    return _u8_from_float(guide)


# 在长边 512 上跑 κ=2.6 的 3 通道联合 FFT L0，再用原图做联合引导滤波上采样，兼顾 <280ms 速度与 1px 刀切硬边。
_L0_WORK_MAX = 512


def l0_smooth(image: np.ndarray, lam: float, kappa: float = 2.6, beta_max: float = 1e5) -> np.ndarray:
    """Xu 2011 L0 梯度最小化，3 通道联合硬阈值 + 半二次分裂 2D FFT。

    beta 从 2λ 起按 κ=2.6 增至 1e5。联合判据 sum(gx^2+gy^2) <= λ/β 置零。
    反射填充把周期振铃留在裁掉的边带里；若经过降采样求解，则用 guidedFilter 恢复原分辨率锐利边缘。
    """
    src = np.ascontiguousarray(image, dtype=np.float32)
    if src.ndim == 2:
        src = src[..., None]
    height, width = src.shape[:2]
    longest = max(height, width)
    if longest > _L0_WORK_MAX:
        scale = _L0_WORK_MAX / float(longest)
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        work = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
        solved = _l0_fft(work, lam, kappa, beta_max)
        up = cv2.resize(solved, (width, height), interpolation=cv2.INTER_LINEAR)
        # 用全分辨率原图 L 通道做轻半径引导上采样，消除线性放大的 2~3px 斜坡糊边
        guide = np.ascontiguousarray(src[..., 0], dtype=np.float32)
        sharp = cv2.ximgproc.guidedFilter(guide, up, radius=2, eps=float(max(lam * 0.08, 1e-4)))
        return np.ascontiguousarray(sharp, dtype=np.float32)
    return _l0_fft(src, lam, kappa, beta_max)


def _l0_fft(src: np.ndarray, lam: float, kappa: float, beta_max: float) -> np.ndarray:
    pad = 24
    padded = np.pad(src, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w, channels = padded.shape
    kx = np.arange(w // 2 + 1, dtype=np.float64)
    ky = np.arange(h, dtype=np.float64)
    fdx = (np.exp(2j * np.pi * kx / w) - 1.0).astype(np.complex64)[None, :]
    fdy = (np.exp(2j * np.pi * ky / h) - 1.0).astype(np.complex64)[:, None]
    den = (np.abs(fdx) ** 2 + np.abs(fdy) ** 2).astype(np.float32)
    fdx_conj = np.conj(fdx)
    fdy_conj = np.conj(fdy)

    s = padded
    with fft.set_workers(_FFT_WORKERS):
        fi = fft.rfft2(s, axes=(0, 1))
        beta = float(2.0 * lam)
        lam = float(lam)
        for _ in range(32):
            if beta >= beta_max:
                break
            gx, gy = _circular_grad(s)
            mag2 = gx * gx + gy * gy
            if channels > 1:
                mag2 = mag2.sum(axis=2)
            keep = (mag2 > (lam / beta)).astype(np.float32)
            if channels > 1:
                keep = keep[..., None]
            gx *= keep
            gy *= keep
            fh = fft.rfft2(gx, axes=(0, 1))
            fv = fft.rfft2(gy, axes=(0, 1))
            beta32 = np.float32(beta)
            numer = fi + beta32 * (fdx_conj[..., None] * fh + fdy_conj[..., None] * fv)
            denom = (np.float32(1.0) + beta32 * den)[..., None]
            fs = numer / denom
            s = fft.irfft2(fs, s=(h, w), axes=(0, 1)).astype(np.float32, copy=False)
            beta *= kappa
    cropped = s[pad : pad + src.shape[0], pad : pad + src.shape[1]]
    return np.ascontiguousarray(cropped, dtype=np.float32)


def _circular_grad(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.empty_like(s)
    gy = np.empty_like(s)
    gx[:, :-1] = s[:, 1:] - s[:, :-1]
    gx[:, -1] = s[:, 0] - s[:, -1]
    gy[:-1] = s[1:] - s[:-1]
    gy[-1] = s[0] - s[-1]
    return gx, gy


def rtv_smooth(image: np.ndarray, texture_scale: float, iterations: int = 2) -> np.ndarray:
    """快速相对总变分近似。

    D = Gσ*|∂I|，L = |Gσ*∂I|。周期网格、格纹、半调与锁子甲在窗口内正负对消，
    L/D 很小，权重 w = 1/(L/D+ε) 变大，这些区域被高斯抹平；孤立硬边 L≈D，权重大
    约等于 1，不参与模糊。2~3 次迭代，全程向量化。
    """
    t = float(np.clip(texture_scale, 0.0, 10.0)) / 10.0
    # 盒子窗口就是 RTV 原文的局部窗，比同半径高斯快一个数量级。
    radius = 2 + int(round(t * 22.0))
    eps = np.float32(1e-3)
    current = np.ascontiguousarray(image, dtype=np.float32)
    w_struct = np.float32(1.0 / (1.0 + 1e-3))
    for _ in range(int(iterations)):
        dx = cv2.Sobel(current, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        dy = cv2.Sobel(current, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
        dx *= np.float32(1.0 / 8.0)
        dy *= np.float32(1.0 / 8.0)
        d_x = _box(np.abs(dx), radius)
        d_y = _box(np.abs(dy), radius)
        l_x = np.abs(_box(dx, radius))
        l_y = np.abs(_box(dy, radius))
        w_x = 1.0 / (l_x / (d_x + eps) + eps)
        w_y = 1.0 / (l_y / (d_y + eps) + eps)
        weight = 0.5 * (w_x + w_y)
        if weight.ndim == 3:
            weight = weight.mean(axis=2)
        hi = _pct(weight, 98.0)
        excess = (weight - w_struct) / (hi - float(w_struct) + 1e-6)
        amount = np.clip((excess - 0.12) / 0.88, 0.0, 1.0).astype(np.float32)
        amount = cv2.GaussianBlur(amount, (0, 0), 0.8, borderType=cv2.BORDER_REFLECT)
        blurred = _box(current, radius)
        if amount.ndim == 2 and current.ndim == 3:
            amount = amount[..., None]
        current = current * (1.0 - amount) + blurred * amount
    return np.ascontiguousarray(current, dtype=np.float32)


def _box(image: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * max(int(radius), 1) + 1
    return cv2.blur(image, (k, k), borderType=cv2.BORDER_REFLECT)


def _pct(arr: np.ndarray, q: float) -> float:
    flat = arr.ravel()
    step = max(1, flat.size // 250_000)
    return float(np.percentile(flat[::step], q))
