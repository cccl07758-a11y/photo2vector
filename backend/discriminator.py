"""256px 纯数学图像体检：只读像素，输出策略、连续参数和纹理蒙版。

判别量全部来自 OKLab 明度/色度的积分统计、加窗频谱和尺度比，
不读取文件名、不做样本哈希。映射是平滑权重在四组物理工况之间的插值，
再向 UNIVERSAL 锚点收回一成，避免落到崩坏极值。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from color_space import M1, M2, srgb_to_linear
from presets import UNIVERSAL, merge_params

# 256 级 sRGB 码值的精确线性化表，避免每次探针再做幂运算。
_SRGB_LUT = srgb_to_linear(np.arange(256, dtype=np.float32) / np.float32(255.0)).astype(np.float32)

_PROBE = 256
_MAD_SCALE = 6.0 * 0.67448975
_LAPLACE = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)

# 四组工况锚点。数值按「插值后再混入 10% UNIVERSAL」反推，使基准样张落在目标区间。
# grid：粗周期网格；pattern：细周期 + 软阴影；paint：非周期厚涂；natural：自然光比。
_GRID = {
    "texture_scale": 9.52,
    "contrast_norm": 0.336,
    "quietness": 0.804,
    "ao_suppression": 0.858,
    "chroma_weight": 0.933,
    "line_width": 1.82,
    "coh_power": 1.36,
    "flow_coherence": 0.582,
    "thresh_low": 0.226,
    "thresh_high": 0.451,
    "min_component_size": 70.0,
    "min_span": 22.4,
    "spatial_strength": 0.872,
    "sigma_norm": 16.0,
}
_PATTERN = {
    "texture_scale": 7.80,
    "contrast_norm": 0.480,
    "quietness": 0.527,
    "ao_suppression": 1.002,
    "chroma_weight": 1.00,
    "line_width": 1.79,
    "coh_power": 1.55,
    "flow_coherence": 0.493,
    "thresh_low": 0.182,
    "thresh_high": 0.362,
    "min_component_size": 40.0,
    "min_span": 15.8,
    "spatial_strength": 0.850,
    "sigma_norm": 16.0,
}
_PAINT = {
    "texture_scale": 4.50,
    "contrast_norm": 0.720,
    "quietness": 0.745,
    "ao_suppression": 0.940,
    "chroma_weight": 1.156,
    "line_width": 1.74,
    "coh_power": 1.52,
    "flow_coherence": 0.649,
    "thresh_low": 0.195,
    "thresh_high": 0.405,
    "min_component_size": 56.0,
    "min_span": 18.5,
    "spatial_strength": 0.783,
    "sigma_norm": 16.0,
}
_NATURAL = {
    "texture_scale": 5.80,
    "contrast_norm": 0.752,
    "quietness": 0.821,
    "ao_suppression": 0.636,
    "chroma_weight": 1.211,
    "line_width": 1.79,
    "coh_power": 1.41,
    "flow_coherence": 0.582,
    "thresh_low": 0.193,
    "thresh_high": 0.384,
    "min_component_size": 56.0,
    "min_span": 18.0,
    "spatial_strength": 0.806,
    "sigma_norm": 18.2,
}
_BLEND_KEYS = tuple(_GRID)
_ANCHOR = 0.10


def diagnose_image(rgb_u8: np.ndarray) -> dict:
    """对任意 RGB 图做 256px 体检。返回报告、推荐参数和 w_tex_256。"""
    started = time.perf_counter()
    lab, L = _probe_oklab(rgb_u8)
    dx, dy = _diff(L)
    texture = _multiscale_texture(dx, dy)
    spectrum = _fft_periodicity(L)
    noise_sigma = _noise_sigma(L)
    light = _exposure_stats(L)
    ao = _soft_ao(L)
    chroma_frac = _chroma_edge(lab, dx, dy)
    w_tex = _texture_weight(L, texture)

    periodicity = _periodicity_score(spectrum, texture)
    roughness = _roughness_score(texture, periodicity)
    hdr = _hdr_score(light, periodicity)
    soft_ao = _soft_ao_score(ao)
    chroma = _smoothstep(0.0022, 0.0172, chroma_frac)
    method = _route(periodicity, roughness, light["loc_cv"])
    params = _map_params(
        method=method,
        periodicity=periodicity,
        roughness=roughness,
        hdr=hdr,
        soft_ao=soft_ao,
        chroma=chroma,
        spectrum=spectrum,
        texture=texture,
        light=light,
    )
    label, badge = _strategy_text(method, soft_ao, noise_sigma, hdr, chroma)
    reasons = _reasons(
        method=method,
        periodicity=periodicity,
        roughness=roughness,
        hdr=hdr,
        soft_ao=soft_ao,
        chroma=chroma,
        noise_sigma=noise_sigma,
        spectrum=spectrum,
        texture=texture,
        light=light,
        ao=ao,
        chroma_frac=chroma_frac,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    scores = {
        "periodicity": _round(periodicity, 4),
        "roughness": _round(roughness, 4),
        "hdr_unevenness": _round(hdr, 4),
        "soft_ao": _round(soft_ao, 4),
        "chroma_edge": _round(chroma, 4),
        "noise_sigma": float(noise_sigma),
        "diagnose_ms": round(elapsed, 2),
    }
    metrics = {
        "T_act": {str(k): _round(v, 4) for k, v in texture["T_act"].items()},
        "sigma_tex_256": _round(texture["sigma_tex_256"], 3),
        "late_growth": _round(texture["late_growth"], 4),
        "ring_kurtosis": _round(spectrum["kurtosis"], 3),
        "Pi": _round(spectrum["Pi"], 3),
        "conc20": _round(spectrum["conc20"], 4),
        "spike_energy": _round(spectrum["spike_energy"], 4),
        "dominant_period_px": _round(spectrum["dominant_period_px"], 2),
        "loc_cv": _round(light["loc_cv"], 4),
        "span": _round(light["span"], 4),
        "dark_ratio": _round(light["dark_ratio"], 4),
        "median_L": _round(light["median_L"], 4),
        "edge_density": _round(texture["edge_density"], 4),
        "ao_net": _round(ao["net"], 4),
        "gentle_frac": _round(ao["gentle_frac"], 4),
        "chroma_frac": _round(chroma_frac, 4),
        "grid_mix": _round(params.pop("_grid_mix"), 4),
    }
    return {
        "strategy_label": label,
        "strategy_badge": badge,
        "reasons": reasons,
        "scores": scores,
        "diagnose_ms": scores["diagnose_ms"],
        "texture_method": method,
        "metrics": metrics,
        "recommended_params": params,
        "w_tex_256": w_tex,
    }


def public_diagnosis(diag: dict) -> dict:
    """去掉蒙版数组，留下可 JSON 化的体检报告。"""
    return {
        "strategy_label": diag["strategy_label"],
        "strategy_badge": diag["strategy_badge"],
        "reasons": list(diag["reasons"]),
        "scores": {key: _json_num(val) for key, val in diag["scores"].items()},
        "diagnose_ms": _json_num(diag["diagnose_ms"]),
        "texture_method": diag["texture_method"],
        "metrics": {key: _json_num(val) for key, val in diag["metrics"].items()},
        "recommended_params": {key: _json_num(val) for key, val in diag["recommended_params"].items()},
    }


def _probe_oklab(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb_u8)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    elif rgb.shape[2] > 3:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest > _PROBE:
        scale = _PROBE / float(longest)
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    lab = _oklab_u8(rgb)
    L = np.ascontiguousarray(lab[..., 0], dtype=np.float32)
    return lab, L


def _oklab_u8(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB → OKLab。线性段走 256 项查表，与 srgb_to_oklab 在码值上一致。"""
    lin = _SRGB_LUT[rgb_u8]
    lms = lin @ M1.T
    lms_c = np.cbrt(np.maximum(lms, 0.0)).astype(np.float32)
    return (lms_c @ M2.T).astype(np.float32)


def _quantile(values: np.ndarray, q: float) -> float:
    """子采样后的分位。256px 上和全量分位只差噪声级，但避免反复排序整图。"""
    flat = np.asarray(values).ravel()
    step = 1 if flat.size <= 8192 else flat.size // 8192
    sample = flat[::step]
    if sample.size == 0:
        return 0.0
    pos = (sample.size - 1) * (float(q) / 100.0)
    k = int(pos)
    if k <= 0:
        k = 0
    elif k >= sample.size:
        k = sample.size - 1
    return float(np.partition(sample, k)[k])


def _diff(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.empty_like(image)
    dy = np.empty_like(image)
    dx[:, 0] = 0
    dx[:, 1:] = image[:, 1:] - image[:, :-1]
    dy[0, :] = 0
    dy[1:, :] = image[1:, :] - image[:-1, :]
    return dx, dy


def _box(image: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * max(int(radius), 1) + 1
    return cv2.blur(image, (k, k), borderType=cv2.BORDER_REFLECT)


def _multiscale_texture(dx: np.ndarray, dy: np.ndarray) -> dict:
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    t_act: dict[int, float] = {}
    coherence_2 = oscillation_2 = magnitude_2 = None
    coherence_4 = oscillation_4 = magnitude_4 = None
    for radius in (2, 4, 8, 16):
        coherence, oscillation, magnitude = _coherence_abs(abs_dx, abs_dy, dx, dy, radius)
        threshold = _quantile(magnitude, 58)
        active = magnitude > threshold
        if np.any(active):
            t_act[radius] = float(oscillation[active].mean())
        else:
            t_act[radius] = float(oscillation.mean())
        if radius == 2:
            coherence_2, oscillation_2, magnitude_2 = coherence, oscillation, magnitude
        elif radius == 4:
            coherence_4, oscillation_4, magnitude_4 = coherence, oscillation, magnitude
    values = np.array([t_act[r] for r in (2, 4, 8, 16)], dtype=np.float64)
    delta = np.diff(values)
    weights = np.clip(delta, 0.0, None)
    centers = np.array([3.0, 6.0, 12.0], dtype=np.float64)
    sigma = float(np.sum(weights * centers) / (np.sum(weights) + 1e-8))
    late = float(delta[-1] / (delta[0] + 1e-6))
    active_2 = magnitude_2 > _quantile(magnitude_2, 58)
    if np.any(active_2):
        edge_density = float((coherence_2[active_2] < 0.45).mean())
    else:
        edge_density = float((coherence_2 < 0.45).mean())
    return {
        "T_act": t_act,
        "sigma_tex_256": sigma,
        "late_growth": late,
        "edge_density": edge_density,
        "C_fine": np.clip(coherence_2, 0.0, 1.0).astype(np.float32),
        "T_use": np.clip(oscillation_4, 0.0, 1.0).astype(np.float32),
        "D_use": magnitude_4.astype(np.float32),
        "D_fine": magnitude_2.astype(np.float32),
    }


def _coherence_abs(
    abs_dx: np.ndarray,
    abs_dy: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_x = _box(abs_dx, radius)
    window_y = _box(abs_dy, radius)
    coherent_x = np.abs(_box(dx, radius))
    coherent_y = np.abs(_box(dy, radius))
    coherence = 0.5 * (
        coherent_x / (window_x + np.float32(1e-6)) + coherent_y / (window_y + np.float32(1e-6))
    )
    return coherence, np.float32(1.0) - coherence, window_x + window_y


def _fft_periodicity(L: np.ndarray) -> dict:
    height, width = L.shape
    hann = np.hanning(height).astype(np.float32)[:, None] * np.hanning(width).astype(np.float32)[None, :]
    field = (L - np.float32(L.mean())) * hann
    spec = np.fft.rfft2(field)
    power = spec.real.astype(np.float64) ** 2 + spec.imag.astype(np.float64) ** 2
    freq_y = np.fft.fftfreq(height).astype(np.float64)[:, None]
    freq_x = np.fft.rfftfreq(width).astype(np.float64)[None, :]
    cycles = np.sqrt(freq_y * freq_y + freq_x * freq_x)
    ref = float(max(height, width))
    period = np.divide(1.0, cycles, out=np.full_like(cycles, 1e9), where=cycles > 1e-12)
    radius = np.maximum(np.rint(cycles * ref).astype(np.int32), 0)
    band = (period >= 4.0) & (period <= 48.0) & (radius > 0)
    rad = radius[band].ravel()
    pw = power[band].ravel()
    if rad.size < 8:
        return {
            "kurtosis": 0.0,
            "Pi": 1.0,
            "conc20": 0.0,
            "spike_energy": 0.0,
            "dominant_period_px": 8.0,
        }
    max_r = int(rad.max())
    sum_energy = np.bincount(rad, weights=pw, minlength=max_r + 1)
    counts = np.bincount(rad, minlength=max_r + 1).astype(np.float64)
    valid = counts > 0
    mean = np.zeros(max_r + 1, dtype=np.float64)
    mean[valid] = sum_energy[valid] / np.maximum(counts[valid], 1.0)
    radii = np.nonzero(valid)[0].astype(np.float64)
    ring = mean[radii.astype(np.int32)]
    rel, slope, intercept = _detrend_rings(radii, ring)
    center = float(np.median(rel)) + 1e-12
    variance = float(rel.var()) + 1e-20
    kurtosis = float(np.mean((rel - rel.mean()) ** 4) / (variance ** 2))
    peakiness = float(rel.max() / center)
    trend = np.exp(slope * np.log(np.arange(max_r + 1, dtype=np.float64) + 1e-6) + intercept)
    white = pw / np.maximum(trend[rad], 1e-20)
    top = min(20, white.size)
    conc20 = float(np.partition(white, -top)[-top:].sum() / (white.sum() + 1e-12))
    median_bin = float(np.median(white)) + 1e-12
    spiked = white > (3.0 * median_bin)
    spike_energy = float(white[spiked].sum() / (white.sum() + 1e-12)) if np.any(spiked) else 0.0
    return {
        "kurtosis": kurtosis,
        "Pi": peakiness,
        "conc20": conc20,
        "spike_energy": spike_energy,
        "dominant_period_px": _dominant_period(rel, radii, ref),
    }


def _detrend_rings(radii: np.ndarray, ring: np.ndarray) -> tuple[np.ndarray, float, float]:
    """用环能量的下包络拟合 1/f，避免谐波峰把趋势抬高。"""
    log_r = np.log(radii)
    log_e = np.log(ring + 1e-20)
    keep = np.ones(radii.shape[0], dtype=bool)
    slope, intercept = _linfit(log_r, log_e)
    for _ in range(4):
        residual = log_e - (slope * log_r + intercept)
        if keep.sum() < 6:
            break
        keep = residual <= np.percentile(residual, 55)
        if int(keep.sum()) < 6:
            keep = np.ones(radii.shape[0], dtype=bool)
            break
        slope, intercept = _linfit(log_r[keep], log_e[keep])
    trend = np.exp(slope * log_r + intercept)
    return ring / np.maximum(trend, 1e-20), float(slope), float(intercept)


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def _dominant_period(rel: np.ndarray, radii: np.ndarray, ref: float) -> float:
    if rel.size < 3:
        return float(ref / max(float(radii[int(np.argmax(rel))]), 1.0))
    middle = rel[1:-1]
    peaks = np.nonzero((middle >= rel[:-2]) & (middle >= rel[2:]) & (middle > 1.05))[0] + 1
    if peaks.size == 0:
        return float(ref / float(radii[int(np.argmax(rel))]))
    order = peaks[np.argsort(rel[peaks])[::-1]]
    best = float(rel[order[0]])
    keep = order[rel[order] >= best * 0.92]
    return float(np.max(ref / radii[keep]))


def _noise_sigma(L: np.ndarray) -> float:
    response = cv2.filter2D(L, cv2.CV_32F, _LAPLACE, borderType=cv2.BORDER_REFLECT)
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    weak = grad < _quantile(grad, 30)
    if not np.any(weak):
        weak = np.ones(grad.shape, dtype=bool)
    mad = float(np.median(np.abs(response[weak])))
    return mad / _MAD_SCALE


def _exposure_stats(L: np.ndarray) -> dict:
    p01, p99 = _quantile(L, 1), _quantile(L, 99)
    mu = _box(L, 12)
    mu2 = _box(L * L, 12)
    sigma = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    loc_cv = float(sigma.std() / (float(sigma.mean()) + 1e-6))
    return {
        "span": float(p99 - p01),
        "dark_ratio": float((L < 0.16).mean()),
        "median_L": _quantile(L, 50),
        "loc_cv": loc_cv,
    }


def _soft_ao(L: np.ndarray) -> dict:
    """先剥掉 σ=12 的漫射照明，再比较粗细尺度梯度。宽缓斜坡 R 高，刀切边 R 低。"""
    residual = L - cv2.GaussianBlur(L, (0, 0), 12)
    coarse = _grad_mag(cv2.GaussianBlur(residual, (0, 0), 4.0))
    fine_blur = cv2.GaussianBlur(residual, (0, 0), 0.8)
    grad_x = cv2.Sobel(fine_blur, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    grad_y = cv2.Sobel(fine_blur, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    fine = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    ratio = (np.float32(4.0) * coarse) / (np.float32(0.8) * fine + np.float32(1e-6))
    coherence = _tensor_coherence(grad_x, grad_y, radius=2)
    low, high = _quantile(fine, 45), _quantile(fine, 88)
    edge = (coherence > 0.40) & (fine >= low) & (fine <= high)
    if np.any(edge):
        values = ratio[edge]
        hi = float((values > 1.15).mean())
        lo = float((values < 0.85).mean())
    else:
        hi = lo = 0.0
    amplitude = cv2.GaussianBlur(np.abs(residual), (0, 0), 3)
    gentle = (amplitude > _quantile(amplitude, 60)) & (fine < _quantile(fine, 50))
    return {"net": hi - lo, "gentle_frac": float(gentle.mean()), "hi": hi, "lo": lo}


def _grad_mag(image: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def _tensor_coherence(grad_x: np.ndarray, grad_y: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    jxx = cv2.blur(grad_x * grad_x, (k, k))
    jyy = cv2.blur(grad_y * grad_y, (k, k))
    jxy = cv2.blur(grad_x * grad_y, (k, k))
    gap = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    return gap / (jxx + jyy + np.float32(1e-6))


def _chroma_edge(lab: np.ndarray, dx_L: np.ndarray, dy_L: np.ndarray) -> float:
    a = lab[..., 1]
    b = lab[..., 2]
    dx_a, dy_a = _diff(a)
    dx_b, dy_b = _diff(b)
    grad_c = np.sqrt(dx_a * dx_a + dy_a * dy_a + dx_b * dx_b + dy_b * dy_b)
    grad_L = np.sqrt(dx_L * dx_L + dy_L * dy_L)
    jxx = _box(dx_a * dx_a + dx_b * dx_b, 2)
    jyy = _box(dy_a * dy_a + dy_b * dy_b, 2)
    jxy = _box(dx_a * dy_a + dx_b * dy_b, 2)
    gap = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    coherence = gap / (jxx + jyy + np.float32(1e-6))
    gate = (grad_c > _quantile(grad_c, 80)) & (grad_L < _quantile(grad_L, 55)) & (coherence > 0.35)
    return float(gate.mean())


def _texture_weight(L: np.ndarray, texture: dict) -> np.ndarray:
    """纹理区趋近 1，单次阶跃和天空趋近 0，再用引导滤波收掉光晕。

    细尺度相干度压低文字、镜框、剑刃这类单次阶跃；平坦门控关掉天空。
    只压最强的一档梯度，避免把格纹和笔触整片当成硬边保护起来。
    归一化用高分位而不是尖峰，否则卷发会把砖墙和水泥相对压暗。
    """
    coherence = texture["C_fine"]
    oscillation = np.clip(texture["T_use"], 0.0, 1.0)
    activity = texture["D_use"]
    fine = texture["D_fine"]
    low, high = _quantile(activity, 12), _quantile(activity, 34)
    if high <= low + 1e-8:
        gate = np.ones_like(activity, dtype=np.float32)
    else:
        gate = _smoothstep_array(float(low), float(high), activity)
    protect = np.power(np.clip(1.0 - coherence, 0.0, 1.0), 0.55).astype(np.float32)
    edge_lo, edge_hi = _quantile(fine, 86), _quantile(fine, 97)
    if edge_hi <= edge_lo + 1e-8:
        edge = np.zeros_like(fine, dtype=np.float32)
    else:
        edge = _smoothstep_array(float(edge_lo), float(edge_hi), fine)
    weight = np.clip(oscillation * protect * gate * (1.0 - edge), 0.0, 1.0).astype(np.float32)
    scale = _quantile(weight, 80)
    if scale > 0.12:
        weight = np.clip(weight / np.float32(scale), 0.0, 1.0)
    guide = np.ascontiguousarray(L, dtype=np.float32)
    source = np.ascontiguousarray(weight, dtype=np.float32)
    smoothed = cv2.ximgproc.guidedFilter(guide, source, radius=2, eps=1e-3)
    opened = _smoothstep_array(0.38, 0.74, np.clip(smoothed, 0.0, 1.0))
    return np.clip(opened, 0.0, 1.0).astype(np.float32)


def _periodicity_score(spectrum: dict, texture: dict) -> float:
    jump = texture["T_act"][4] - texture["T_act"][2]
    score = (
        0.36 * _smoothstep(1.85, 8.6, spectrum["kurtosis"])
        + 0.30 * _smoothstep(1.22, 2.50, spectrum["Pi"])
        + 0.18 * _smoothstep(0.36, 0.53, spectrum["spike_energy"])
        + 0.06 * _smoothstep(0.028, 0.042, spectrum["conc20"])
        + 0.10 * _smoothstep(0.09, 0.21, jump)
    )
    return float(np.clip(score, 0.0, 1.0))


def _roughness_score(texture: dict, periodicity: float) -> float:
    fine = 0.55 * texture["T_act"][2] + 0.45 * texture["T_act"][4]
    aperiodic = 1.0 - 0.75 * periodicity
    density = 0.50 + 0.70 * texture["edge_density"]
    return float(np.clip((fine * aperiodic * density) / 0.62, 0.0, 1.0))


def _hdr_score(light: dict, periodicity: float) -> float:
    gated = light["loc_cv"] * (1.0 - 0.65 * periodicity)
    key = float(np.clip((0.55 - light["median_L"]) / 0.32, 0.0, 1.0))
    score = (
        0.08
        + 0.58 * _smoothstep(0.30, 0.84, gated)
        + 0.34 * key
        + 0.10 * _smoothstep(0.55, 0.88, light["span"])
        + 0.08 * _smoothstep(0.0, 0.06, light["dark_ratio"])
    )
    return float(np.clip(score, 0.0, 1.0))


def _soft_ao_score(ao: dict) -> float:
    gentle = _smoothstep(0.050, 0.084, ao["gentle_frac"])
    net = _smoothstep(-0.55, -0.12, ao["net"])
    return float(np.clip(0.78 * gentle + 0.22 * net, 0.0, 1.0))


def _route(periodicity: float, roughness: float, loc_cv: float) -> str:
    if periodicity >= 0.52:
        return "rtv"
    if roughness >= 0.50 and loc_cv < 0.66:
        return "l0"
    return "rgf"


def _map_params(
    method: str,
    periodicity: float,
    roughness: float,
    hdr: float,
    soft_ao: float,
    chroma: float,
    spectrum: dict,
    texture: dict,
    light: dict,
) -> dict:
    period_n = _smoothstep(8.0, 15.5, spectrum["dominant_period_px"])
    late_n = _smoothstep(0.42, 0.92, texture["late_growth"])
    grid_mix = float(np.clip(0.64 * period_n + 0.36 * late_n, 0.0, 1.0))
    # 周期权重在路由阈值 0.52 附近饱和，避免「判成 RTV、参数却还停在自然场景」。
    w_periodic = _smoothstep(0.40, 0.55, periodicity)
    loc_gate = 1.0 - _smoothstep(0.64, 0.74, light["loc_cv"])
    w_paint = _smoothstep(0.46, 0.64, roughness) * (1.0 - w_periodic) * loc_gate
    w_natural = max(0.0, 1.0 - w_periodic - w_paint)
    total = w_periodic + w_paint + w_natural
    w_periodic /= total
    w_paint /= total
    w_natural /= total
    blended: dict[str, float] = {}
    for key in _BLEND_KEYS:
        periodic = grid_mix * _GRID[key] + (1.0 - grid_mix) * _PATTERN[key]
        mixed = w_periodic * periodic + w_paint * _PAINT[key] + w_natural * _NATURAL[key]
        anchored = (1.0 - _ANCHOR) * mixed + _ANCHOR * float(UNIVERSAL[key])
        blended[key] = anchored
    # 工况内部仍随测量值连续挪动，幅度被限制在锚点附近。
    blended["contrast_norm"] += 0.04 * (hdr - 0.5) * (1.0 - w_periodic)
    blended["chroma_weight"] += 0.02 * (chroma - 0.5)
    blended["ao_suppression"] += 0.035 * (soft_ao - 0.5)
    blended["texture_scale"] += 0.35 * (period_n - 0.35) * w_periodic
    blended["spatial_strength"] += 0.04 * w_periodic + 0.02 * roughness * (1.0 - w_periodic)
    _ = method, roughness
    mapped = dict(UNIVERSAL)
    mapped.update(blended)
    mapped["texture_method"] = method
    mapped["spatial_adaptive"] = True
    mapped.pop("noise", None)
    params = merge_params(mapped)
    params["_grid_mix"] = grid_mix
    return params


def _strategy_text(method: str, soft_ao: float, noise_sigma: float, hdr: float, chroma: float) -> tuple[str, str]:
    if method == "rtv" and soft_ao >= 0.62 and noise_sigma < 1.1e-3:
        return "RTV 去格纹 + 强抑软阴影", "RTV 去格纹"
    if method == "rtv":
        return "RTV 周期消网 + 分区护主体", "RTV 周期消网"
    if method == "l0":
        if hdr >= 0.45:
            return "L0 厚涂烫平 + 暗部提亮", "L0 厚涂去笔触"
        return "L0 厚涂烫平 + 硬边保留", "L0 厚涂去笔触"
    if hdr >= 0.45 and chroma >= 0.45:
        return "RGF 强光均衡 + 撞色增强", "RGF 强光均衡"
    return "RGF 保边去纹理 + 主体保护", "RGF 保边"


def _reasons(
    method: str,
    periodicity: float,
    roughness: float,
    hdr: float,
    soft_ao: float,
    chroma: float,
    noise_sigma: float,
    spectrum: dict,
    texture: dict,
    light: dict,
    ao: dict,
    chroma_frac: float,
) -> list[str]:
    period = spectrum["dominant_period_px"]
    if method == "rtv":
        head = (
            f"中频环峰度 {spectrum['kurtosis']:.1f}、谱尖峰 {spectrum['Pi']:.2f}，"
            f"周期度 {periodicity:.2f}，主导周期约 {period:.0f}px。"
        )
    elif method == "l0":
        head = (
            f"活跃区振荡 T2={texture['T_act'][2]:.2f}、T4={texture['T_act'][4]:.2f}，"
            f"周期度仅 {periodicity:.2f}，肌理度 {roughness:.2f}。"
        )
    else:
        head = (
            f"周期度 {periodicity:.2f}、环峰度 {spectrum['kurtosis']:.1f}，"
            f"没有稳定的中频谐波，改走保边滚动引导。"
        )
    gated = light["loc_cv"] * (1.0 - 0.65 * periodicity)
    light_line = (
        f"明度跨度 {light['span']:.2f}，局部反差 CV {light['loc_cv']:.2f}"
        f"（周期门控后 {gated:.2f}），光比不均度 {hdr:.2f}。"
    )
    tail = (
        f"软影净占比 {ao['net']:+.2f}、宽缓斜坡 {ao['gentle_frac']:.3f}，"
        f"底噪 σ={noise_sigma:.1e}，撞色占比 {chroma_frac:.1%}，撞色度 {chroma:.2f}。"
    )
    return [head, light_line, tail]


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 == edge0:
        return 0.0
    t = (float(value) - edge0) / (edge1 - edge0)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def _smoothstep_array(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    t = np.clip((values - np.float32(edge0)) / np.float32(edge1 - edge0), 0.0, 1.0)
    return (t * t * (np.float32(3.0) - np.float32(2.0) * t)).astype(np.float32)


def _round(value: float, digits: int) -> float:
    return round(float(value), digits)


def _json_num(value):
    if isinstance(value, dict):
        return {str(key): _json_num(item) for key, item in value.items()}
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
