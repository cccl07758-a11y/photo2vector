"""暗部 / 中调 / 高光三层虚拟曝光提线。

曝光在去纹理之前做：把挤在暗部或高光里的色阶拉开，L0 / RGF 才不会把
盔甲和受光皮肤当成弱笔触抹掉。三层软墨线按亮度窗留在各自的优势区，
合成只改权重时直接命中缓存。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from color_space import oklab_to_srgb, srgb_to_oklab
from line_extractors import extract_soft
from postprocess import finalize
from smoothing import remove_texture
from tensor_norm import structure_tensor

BANDS = ("shadow", "midtone", "highlight")
# 864 是质量上限。再往上三层混合提线会超过 1.1s，768 仍能把线稿线性放大回请求尺寸。
WORK_SIDE = 768

_SHADOW_RGB = np.array([0x1D, 0x4E, 0xD8], dtype=np.float32)
_MID_RGB = np.array([0x18, 0x18, 0x1B], dtype=np.float32)
_HIGH_RGB = np.array([0xDB, 0x27, 0x77], dtype=np.float32)

_PACK_CACHE: OrderedDict[str, "BandPack"] = OrderedDict()
_LINE_CACHE: OrderedDict[str, "LinePack"] = OrderedDict()
_COMPOSE_CACHE: OrderedDict[str, dict] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_PACK_LIMIT = 3
_LINE_LIMIT = 4
_COMPOSE_LIMIT = 12


@dataclass
class BandPack:
    key: str
    raw: dict[str, np.ndarray]
    gated: dict[str, np.ndarray]
    exposure: dict[str, np.ndarray]
    out_size: tuple[int, int]


@dataclass
class LinePack:
    gated_ink: dict[str, np.ndarray]
    rgb: dict[str, np.ndarray]
    color_index: np.ndarray
    color_alpha: dict[str, np.ndarray]


@dataclass
class MultiBandResult:
    final_rgb: np.ndarray
    rgba: np.ndarray
    images: dict[str, np.ndarray]
    keys: dict[str, str]
    stats: dict
    pack_ms: float
    compose_ms: float
    pack_hit: bool
    ink: np.ndarray


def run_multiband(
    rgb_u8: np.ndarray,
    params: dict,
    w_tex_256: np.ndarray | None,
    token: str,
) -> MultiBandResult:
    """三层提线。`params` 需已经过 merge_params。输出尺寸与单层管线的 max_side 一致。"""
    out = _resize_max(rgb_u8, int(params["max_side"]))
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    pack_key = _pack_key(token, out.shape, params)
    t0 = time.perf_counter()
    with _CACHE_LOCK:
        pack = _PACK_CACHE.get(pack_key)
        if pack is not None:
            _PACK_CACHE.move_to_end(pack_key)
    pack_hit = pack is not None
    if pack is None:
        pack = _build_pack(out, params, w_tex_256, pack_key)
        with _CACHE_LOCK:
            _PACK_CACHE[pack_key] = pack
            _PACK_CACHE.move_to_end(pack_key)
            while len(_PACK_CACHE) > _PACK_LIMIT:
                _PACK_CACHE.popitem(last=False)
    pack_ms = 0.0 if pack_hit else (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    composed = _compose(pack, params)
    compose_ms = (time.perf_counter() - t1) * 1000.0
    return MultiBandResult(
        final_rgb=composed["final_rgb"],
        rgba=composed["rgba"],
        images=composed["images"],
        keys=composed["keys"],
        stats=composed["stats"],
        pack_ms=pack_ms,
        compose_ms=compose_ms,
        pack_hit=pack_hit,
        ink=composed["final_ink"],
    )


def expose_oklab(lab: np.ndarray, band: str, params: dict) -> np.ndarray:
    """自适应分位数三档虚拟曝光。输入输出都是 OKLab float32。

    关键改进：不再假设所有图片中位数都在 0.5（海滩中位数 0.75、骑士中位数 0.31），
    而是根据当前画面的真实明度分位数把暗部带、中调带、高光带各自撑满 [0.05, 0.95] 动态范围！
    """
    src = np.asarray(lab, dtype=np.float32)
    luma = np.clip(src[..., 0], 0.0, 1.0)
    chroma_a = src[..., 1]
    chroma_b = src[..., 2]
    p01, p22, p46, p54, p78, p995 = np.percentile(
        luma[::2, ::2].ravel(), [1.0, 22.0, 46.0, 54.0, 78.0, 99.5]
    )
    out = np.empty_like(src)

    if band == "shadow":
        push = float(np.clip(params["shadow_push"], 0.0, 1.0))
        lo = float(p01)
        hi = max(float(p46), lo + 0.16)
        # 将暗部主区间 [lo, hi] 线性+有理曲线撑开至 [0.04, 0.92]
        norm_t = np.clip((luma - np.float32(lo)) / np.float32(hi - lo + 1e-5), 0.0, 1.35)
        k = np.float32(0.65 - 0.42 * push)
        stretched = np.clip((norm_t * (np.float32(1.0) + k)) / (norm_t + k + np.float32(1e-6)), 0.0, 1.0)
        lifted = (1.0 - push) * luma + push * (np.float32(0.04) + np.float32(0.91) * stretched)
        in_band = (1.0 - _smoothstep(hi - 0.05, hi + 0.12, luma)).astype(np.float32)
        gain = (1.0 + 0.75 * push * in_band).astype(np.float32)
        out[..., 0] = np.clip(lifted, 0.0, 1.0)
        out[..., 1] = chroma_a * gain
        out[..., 2] = chroma_b * gain
        return out

    if band == "highlight":
        pull = float(np.clip(params["highlight_pull"], 0.0, 1.0))
        hi = max(float(p995), 0.35)
        lo = min(float(p54), hi - 0.14)
        # 将强光挤压区间 [lo, hi]（例如海滩女孩 [0.74, 0.96]）撑满至 [0.06, 0.95]！
        norm_t = np.clip((luma - np.float32(lo)) / np.float32(hi - lo + 1e-5), -0.25, 1.0)
        u = np.clip(np.float32(1.0) - norm_t, 0.0, 1.25)
        k = np.float32(0.62 - 0.40 * pull)
        pulled_u = (u * (np.float32(1.0) + k)) / (u + k + np.float32(1e-6))
        darkened = np.clip(np.float32(1.0) - np.clip(pulled_u, 0.0, 1.0), 0.0, 1.0)
        mapped = (1.0 - pull) * luma + pull * (np.float32(0.06) + np.float32(0.90) * darkened)
        in_band = _smoothstep(lo - 0.10, lo + 0.05, luma)
        gain = (1.0 + 0.90 * pull * in_band).astype(np.float32)
        out[..., 0] = np.clip(mapped, 0.0, 1.0)
        out[..., 1] = chroma_a * gain
        out[..., 2] = chroma_b * gain
        return out

    lo_m = float(p22)
    hi_m = max(float(p78), lo_m + 0.22)
    norm_m = np.clip((luma - np.float32(lo_m)) / np.float32(hi_m - lo_m + 1e-5), 0.0, 1.0)
    contrast = float(params.get("mid_contrast", 1.15))
    blend_m = np.clip((contrast - 1.0) / 0.45, 0.0, 1.0)
    out[..., 0] = np.clip((1.0 - 0.55 * blend_m) * luma + (0.55 * blend_m) * (0.05 + 0.90 * norm_m), 0.0, 1.0)
    out[..., 1] = chroma_a * np.float32(1.12)
    out[..., 2] = chroma_b * np.float32(1.12)
    return out


def luminance_weights(luma: np.ndarray, params: dict) -> dict[str, np.ndarray]:
    """自适应分位数平顶重叠窗：保证任意亮度处 max(W_s, W_m, W_h) == 1.0，绝不在过渡带削弱边缘！"""
    base = cv2.GaussianBlur(
        np.ascontiguousarray(luma, dtype=np.float32),
        (0, 0),
        4.0,
        borderType=cv2.BORDER_REFLECT,
    )
    p35, p66 = np.percentile(base[::2, ::2].ravel(), [35.0, 66.0])
    user_s = float(params.get("shadow_hi", 0.42))
    user_h = float(params.get("highlight_lo", 0.58))
    # 70% 跟随当前画面真实三分位，30% 保留参数微调锚点
    q_s = float(np.clip(0.70 * p35 + 0.30 * user_s, 0.15, 0.72))
    q_h = float(np.clip(0.70 * p66 + 0.30 * user_h, q_s + 0.08, 0.90))
    overlap = max(float(params.get("band_overlap", 0.14)), 0.06)

    # 平顶窗：[0, q_s] 内 shadow 恒为 1.0，向 q_s + overlap 平滑衰减；
    # [q_s, q_h] 内 mid 恒为 1.0，两侧与 shadow/highlight 的 1.0 平台完整重叠！
    # [q_h, 1.0] 内 highlight 恒为 1.0！
    shadow = (1.0 - _smoothstep(q_s, q_s + overlap, base)).astype(np.float32)
    highlight = _smoothstep(q_h - overlap, q_h, base).astype(np.float32)
    mid_rise = _smoothstep(q_s - overlap, q_s, base)
    mid_fall = 1.0 - _smoothstep(q_h, q_h + overlap, base)
    mid = np.clip(mid_rise * mid_fall, 0.0, 1.0).astype(np.float32)
    return {"shadow": shadow, "midtone": mid, "highlight": highlight}


def _build_pack(
    rgb_out: np.ndarray,
    params: dict,
    w_tex_256: np.ndarray | None,
    pack_key: str,
) -> BandPack:
    work_side = min(int(params["max_side"]), WORK_SIDE)
    rgb_work = _resize_max(rgb_out, work_side)
    base_lab = srgb_to_oklab(rgb_work.astype(np.float32) / np.float32(255.0))
    out_size = (rgb_out.shape[1], rgb_out.shape[0])

    exposed_labs = {}
    for band in BANDS:
        exposed = expose_oklab(base_lab, band, params)
        exposed_labs[band] = np.nan_to_num(exposed, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    # 先算平顶亮度窗，把每层优势区作为结构张量与软墨线的独立标定 ROI，防止带外强边缘压扁本层！
    windows = luminance_weights(base_lab[..., 0], params)
    rois = {band: (windows[band] > 0.25) for band in BANDS}

    preview, labs = _smooth_all(exposed_labs, params, w_tex_256)

    def _tensor(band: str):
        return band, structure_tensor(labs[band], params, roi_mask=rois[band])

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="band-tensor") as pool:
        fields = dict(pool.map(_tensor, BANDS))

    # remap 必须单线程才不会写出 NaN，但三层可以各占一个核同时跑。
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)

    def _extract(band: str):
        soft = extract_soft(
            params["extract_mode"],
            fields[band],
            params,
            preview[band],
            parallel_safe=True,
        )
        soft = np.clip(np.nan_to_num(soft, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        return band, soft.astype(np.float32)

    try:
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="band-extract") as pool:
            softs = dict(pool.map(_extract, BANDS))
    finally:
        cv2.setNumThreads(previous_threads)

    raw = {band: softs[band] for band in BANDS}
    gated = {
        band: np.clip(softs[band] * windows[band], 0.0, 1.0).astype(np.float32)
        for band in BANDS
    }
    exposure = {band: _resize_rgb(preview[band], out_size) for band in BANDS}
    return BandPack(key=pack_key, raw=raw, gated=gated, exposure=exposure, out_size=out_size)


def _smooth_all(exposed_labs: dict[str, np.ndarray], params: dict, w_tex_256):
    """中调跑主去纹理算子；暗部与高光辅层用带空间纹理蒙版 W_tex 的多轮保边引导滤波，避免大阈值双边滤波糊掉受光手臂！"""
    method = str(params["texture_method"])
    scale = float(params["texture_scale"])
    preview: dict[str, np.ndarray] = {}
    labs: dict[str, np.ndarray] = {}
    if method in {"", "none"} or scale <= 1e-6:
        for band in BANDS:
            preview[band], labs[band] = _lab_to_u8(exposed_labs[band]), exposed_labs[band]
        return preview, labs

    spatial_on = bool(params.get("spatial_adaptive", True))
    spatial_str = float(params.get("spatial_strength", 0.8))

    def _aux_band(band: str):
        return band, _guided_band_smooth(
            exposed_labs[band],
            scale,
            w_tex_256=w_tex_256 if spatial_on else None,
            spatial_strength=spatial_str,
        )

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="band-smooth") as pool:
        mid_fut = pool.submit(_smooth_band, "midtone", exposed_labs["midtone"], params, w_tex_256)
        sh_fut = pool.submit(_aux_band, "shadow")
        hi_fut = pool.submit(_aux_band, "highlight")
        preview["midtone"], labs["midtone"] = mid_fut.result()
        for fut in (sh_fut, hi_fut):
            b_name, (b_rgb, b_lab) = fut.result()
            preview[b_name], labs[b_name] = b_rgb, b_lab
    return preview, labs


def _guided_band_smooth(
    exposed_lab: np.ndarray,
    scale: float,
    w_tex_256: np.ndarray | None = None,
    spatial_strength: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """针对已拉开动态范围的暗部/高光层做 2 轮滚动引导滤波，并用 W_tex 保护主体边缘。"""
    t = float(np.clip(scale, 0.0, 10.0)) / 10.0
    src = np.ascontiguousarray(exposed_lab, dtype=np.float32)
    radius = int(np.clip(round(2 + 5.0 * t), 2, 7))
    # 适配分位数拉伸后的振幅：3 轮滚动引导抹平被同步放大的暗部石墙笔触与微颗粒，保住大阶跃硬边
    eps = float((0.026 + 0.056 * t) ** 2)
    guide = cv2.GaussianBlur(src[..., 0], (0, 0), max(1.2, radius * 0.55), borderType=cv2.BORDER_REFLECT)
    heavy = src
    try:
        for _ in range(3):
            heavy = cv2.ximgproc.guidedFilter(np.ascontiguousarray(guide, dtype=np.float32), src, radius, eps)
            guide = heavy[..., 0]
        light = cv2.ximgproc.guidedFilter(np.ascontiguousarray(src[..., 0], dtype=np.float32), src, 1, 3e-4)
    except cv2.error:
        heavy = cv2.GaussianBlur(src, (0, 0), 1.4)
        light = src

    if w_tex_256 is not None and spatial_strength > 1e-4:
        h, w = src.shape[:2]
        w_full = cv2.resize(np.asarray(w_tex_256, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        w_full = np.clip(w_full, 0.0, 1.0)
        heavy_w = np.clip(1.0 - spatial_strength * (1.0 - w_full), 0.0, 1.0)[..., None]
        out = np.ascontiguousarray(light + heavy_w * (heavy - light), dtype=np.float32)
    else:
        out = np.ascontiguousarray(heavy, dtype=np.float32)
    return _lab_to_u8(out), out


def _smooth_band(band: str, exposed_lab: np.ndarray, params: dict, w_tex_256):
    method = str(params["texture_method"])
    scale = float(params["texture_scale"])
    # 辅层不走整段 L0 / RTV。曝光已经把弱阶跃拉开，引导滤波或短滚动引导足够保边。
    if band != "midtone" and method in {"l0", "rgf", "rtv"} and scale > 1e-6:
        if method == "l0":
            return _fast_l0(exposed_lab, scale)
        rgb = _bilateral_u8(exposed_lab, scale)
        lab = srgb_to_oklab(rgb.astype(np.float32) / np.float32(255.0))
        return rgb, lab
    exposed_rgb = _lab_to_u8(exposed_lab)
    return remove_texture(
        exposed_rgb,
        method,
        scale,
        w_tex_256=w_tex_256,
        spatial_adaptive=bool(params["spatial_adaptive"]),
        spatial_strength=float(params["spatial_strength"]),
    )


def _fast_l0(exposed_lab: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """L0 的高速近似：曝光后的 OKLab 走一次引导滤波，保住拉开的阶跃、压住笔触。"""
    t = float(np.clip(scale, 0.0, 10.0)) / 10.0
    radius = int(np.clip(round(2 + 4.0 * t), 2, 6))
    eps = float((0.015 + 0.030 * t) ** 2)
    src = np.ascontiguousarray(exposed_lab, dtype=np.float32)
    guide = np.ascontiguousarray(src[..., 0], dtype=np.float32)
    try:
        smooth = cv2.ximgproc.guidedFilter(guide, src, radius, eps)
    except cv2.error:
        smooth = cv2.GaussianBlur(src, (0, 0), 1.2)
    smooth = np.ascontiguousarray(smooth, dtype=np.float32)
    return _lab_to_u8(smooth), smooth


def _resize_rgba(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    if image.shape[1] == size_wh[0] and image.shape[0] == size_wh[1]:
        return image
    return cv2.resize(image, size_wh, interpolation=cv2.INTER_LINEAR)


def _compose(pack: BandPack, params: dict) -> dict:
    light_key = pack.key + _light_sig(params) + _thresh_sig(params)
    with _CACHE_LOCK:
        cached = _COMPOSE_CACHE.get(light_key)
        if cached is not None:
            _COMPOSE_CACHE.move_to_end(light_key)
            return cached

    ws = float(params["shadow_weight"])
    wm = float(params["mid_weight"])
    wh = float(params["highlight_weight"])
    mode = str(params.get("band_blend_mode", "weighted_max"))
    combined = _blend(pack.gated["shadow"], pack.gated["midtone"], pack.gated["highlight"], ws, wm, wh, mode)
    lines = _lines_for(pack, params)
    weights = {"shadow": ws, "midtone": wm, "highlight": wh}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="band-compose") as pool:
        final_fut = pool.submit(finalize, combined, params)
        color_fut = pool.submit(_color_from_prepared, lines, weights, combined.shape)
        final_ink, info = final_fut.result()
        color = color_fut.result()
    out_size = pack.out_size
    final_rgb = _resize_rgb(_ink_rgb(final_ink), out_size)
    rgba = _resize_rgba(_ink_rgba(final_ink), out_size)
    color = _resize_rgb(color, out_size)
    images = dict(lines.rgb)
    images["shadow_exposure"] = pack.exposure["shadow"]
    images["mid_exposure"] = pack.exposure["midtone"]
    images["highlight_exposure"] = pack.exposure["highlight"]
    images["color_bands"] = color
    keys = _image_keys(pack.key, params)
    payload = {
        "final_rgb": final_rgb,
        "rgba": rgba,
        "final_ink": np.ascontiguousarray(final_ink, dtype=np.float32),
        "images": images,
        "keys": keys,
        "stats": {
            "components": int(info["components"]),
            "line_ratio": float(info["line_ratio"]),
            "soft_p50": _pct(combined, 50.0),
            "soft_p99": _pct(combined, 99.0),
            "band_ratio": {band: _pct_mean(lines.gated_ink[band]) for band in BANDS},
        },
    }
    with _CACHE_LOCK:
        _COMPOSE_CACHE[light_key] = payload
        _COMPOSE_CACHE.move_to_end(light_key)
        while len(_COMPOSE_CACHE) > _COMPOSE_LIMIT:
            _COMPOSE_CACHE.popitem(last=False)
    return payload


def _blend(shadow, mid, highlight, ws, wm, wh, mode: str) -> np.ndarray:
    if mode == "soft_sum":
        a = np.clip(shadow * np.float32(ws), 0.0, 1.0)
        b = np.clip(mid * np.float32(wm), 0.0, 1.0)
        c = np.clip(highlight * np.float32(wh), 0.0, 1.0)
        return (1.0 - (1.0 - a) * (1.0 - b) * (1.0 - c)).astype(np.float32)
    return np.clip(
        np.maximum.reduce(
            (
                shadow * np.float32(ws),
                mid * np.float32(wm),
                highlight * np.float32(wh),
            )
        ),
        0.0,
        1.0,
    ).astype(np.float32)


def _lines_for(pack: BandPack, params: dict) -> LinePack:
    key = pack.key + _thresh_sig(params)
    with _CACHE_LOCK:
        found = _LINE_CACHE.get(key)
        if found is not None:
            _LINE_CACHE.move_to_end(key)
            return found

    def _one(item):
        band, kind, ink = item
        final, _info = finalize(ink, params)
        return band, kind, final

    jobs = []
    for band in BANDS:
        jobs.append((band, "gated", pack.gated[band]))
        jobs.append((band, "raw", pack.raw[band]))
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="band-line") as pool:
        done = list(pool.map(_one, jobs))

    gated_ink = {}
    rgb = {}
    for band, kind, ink in done:
        if kind == "gated":
            gated_ink[band] = ink
            rgb[f"{band}_line"] = _resize_rgb(_ink_rgb(ink), pack.out_size)
        else:
            rgb[f"{band}_raw_line"] = _resize_rgb(_ink_rgb(ink), pack.out_size)
    # 中调键名在前端用 mid_*。内部带名是 midtone。
    rgb["mid_line"] = rgb.pop("midtone_line")
    rgb["mid_raw_line"] = rgb.pop("midtone_raw_line")
    color_index, color_alpha = _prepare_color(gated_ink)
    pack_lines = LinePack(
        gated_ink=gated_ink,
        rgb=rgb,
        color_index=color_index,
        color_alpha=color_alpha,
    )
    with _CACHE_LOCK:
        _LINE_CACHE[key] = pack_lines
        _LINE_CACHE.move_to_end(key)
        while len(_LINE_CACHE) > _LINE_LIMIT:
            _LINE_CACHE.popitem(last=False)
    return pack_lines


def _prepare_color(gated_ink: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """把三层净网线的非白像素收成索引，权重滑块只在这些点上混色。"""
    peak = np.maximum(
        np.maximum(gated_ink["shadow"], gated_ink["midtone"]),
        gated_ink["highlight"],
    )
    index = np.flatnonzero(peak > np.float32(0.02))
    alphas = {band: np.ascontiguousarray(gated_ink[band].reshape(-1)[index]) for band in BANDS}
    return index, alphas


def _color_from_prepared(lines: LinePack, weights: dict[str, float], shape: tuple[int, int]) -> np.ndarray:
    """白底上叠三色：靛蓝暗部、炭黑中调、品红高光。重叠处按墨量混色。"""
    colors = {"shadow": _SHADOW_RGB, "midtone": _MID_RGB, "highlight": _HIGH_RGB}
    out = np.full(shape + (3,), 255, np.uint8)
    index = lines.color_index
    if index.size == 0:
        return out
    acc = np.zeros((index.size, 3), np.float32)
    cover = np.zeros(index.size, np.float32)
    for band, color in colors.items():
        alpha = np.clip(lines.color_alpha[band] * np.float32(max(weights[band], 0.0)), 0.0, 1.0)
        acc += color * alpha[:, None]
        cover += alpha
    cover = np.clip(cover, 0.0, 1.0)
    mix = acc / np.maximum(cover[:, None], np.float32(1e-4))
    painted = np.float32(255.0) * (1.0 - cover)[:, None] + mix * cover[:, None]
    painted = np.clip(np.round(painted), 0, 255).astype(np.uint8)
    # 权重为 0 的像素 cover=0，混色结果是 0，要改回纸白。
    painted[cover <= np.float32(1e-4)] = 255
    out.reshape(-1, 3)[index] = painted
    return out


def _image_keys(pack_key: str, params: dict) -> dict[str, str]:
    thresh = _thresh_sig(params)
    light = _light_sig(params)
    keys = {
        "shadow_exposure": pack_key + "_sx",
        "mid_exposure": pack_key + "_mx",
        "highlight_exposure": pack_key + "_hx",
        "shadow_line": pack_key + thresh + "_sl",
        "mid_line": pack_key + thresh + "_ml",
        "highlight_line": pack_key + thresh + "_hl",
        "shadow_raw_line": pack_key + thresh + "_sr",
        "mid_raw_line": pack_key + thresh + "_mr",
        "highlight_raw_line": pack_key + thresh + "_hr",
        "color_bands": pack_key + thresh + light + "_cb",
        "final_line": pack_key + thresh + light + "_fl",
        "final_rgba": pack_key + thresh + light + "_fa",
    }
    return keys


def _pack_key(token: str, shape, params: dict) -> str:
    payload = {
        "token": token,
        "shape": list(shape),
        "work": min(int(params["max_side"]), WORK_SIDE),
        "texture_method": params["texture_method"],
        "texture_scale": round(float(params["texture_scale"]), 5),
        "spatial_adaptive": bool(params["spatial_adaptive"]),
        "spatial_strength": round(float(params["spatial_strength"]), 5),
        "chroma_weight": round(float(params["chroma_weight"]), 5),
        "sigma_g": round(float(params["sigma_g"]), 5),
        "sigma_norm": round(float(params["sigma_norm"]), 5),
        "quietness": round(float(params["quietness"]), 5),
        "contrast_norm": round(float(params["contrast_norm"]), 5),
        "extract_mode": params["extract_mode"],
        "line_width": round(float(params["line_width"]), 5),
        "coh_power": round(float(params["coh_power"]), 5),
        "ao_suppression": round(float(params["ao_suppression"]), 5),
        "flow_coherence": round(float(params["flow_coherence"]), 5),
        "fdog_sigma": round(float(params["fdog_sigma"]), 5),
        "fdog_rho": round(float(params["fdog_rho"]), 5),
        "fdog_eps": round(float(params["fdog_eps"]), 5),
        "fdog_phi": round(float(params["fdog_phi"]), 5),
        "eta": round(float(params["eta"]), 5),
        "etf_iters": int(params["etf_iters"]),
        "etf_radius": int(params["etf_radius"]),
        "shadow_push": round(float(params["shadow_push"]), 5),
        "highlight_pull": round(float(params["highlight_pull"]), 5),
        "mid_contrast": round(float(params.get("mid_contrast", 1.15)), 5),
        "shadow_hi": round(float(params.get("shadow_hi", 0.42)), 5),
        "highlight_lo": round(float(params.get("highlight_lo", 0.58)), 5),
        "band_overlap": round(float(params.get("band_overlap", 0.14)), 5),
    }
    return _digest(payload)


def _thresh_sig(params: dict) -> str:
    return _digest(
        {
            "threshold_mode": params["threshold_mode"],
            "thresh_low": round(float(params["thresh_low"]), 5),
            "thresh_high": round(float(params["thresh_high"]), 5),
            "min_component_size": int(params["min_component_size"]),
            "min_span": round(float(params["min_span"]), 4),
            "thin_1px": bool(params["thin_1px"]),
        }
    )


def _light_sig(params: dict) -> str:
    return _digest(
        {
            "mode": params.get("band_blend_mode", "weighted_max"),
            "s": round(float(params["shadow_weight"]), 5),
            "m": round(float(params["mid_weight"]), 5),
            "h": round(float(params["highlight_weight"]), 5),
        }
    )


def _digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    span = np.float32(edge1 - edge0)
    t = np.clip((x - np.float32(edge0)) / span, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _resize_max(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side or max_side <= 0:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _lab_to_u8(lab: np.ndarray) -> np.ndarray:
    rgb = np.clip(oklab_to_srgb(lab), 0.0, 1.0)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def _ink_rgb(ink: np.ndarray) -> np.ndarray:
    gray = cv2.convertScaleAbs((1.0 - ink) * np.float32(255.0))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _ink_rgba(ink: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.round(ink * 255.0), 0, 255).astype(np.uint8)
    rgba = np.zeros((ink.shape[0], ink.shape[1], 4), np.uint8)
    rgba[..., 3] = alpha
    return rgba


def _resize_float(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    if image.shape[1] == size_wh[0] and image.shape[0] == size_wh[1]:
        return np.ascontiguousarray(image, dtype=np.float32)
    return cv2.resize(image, size_wh, interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    if image.shape[1] == size_wh[0] and image.shape[0] == size_wh[1]:
        return image
    return cv2.resize(image, size_wh, interpolation=cv2.INTER_LINEAR)


def _pct(arr: np.ndarray, q: float) -> float:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    step = max(1, flat.size // 250_000)
    return round(float(np.percentile(flat[::step], q)), 4)


def _pct_mean(ink: np.ndarray) -> float:
    return round(float(np.mean(ink >= 0.35)), 4)
