"""阶段一全链路：去纹理 → 结构张量 → 提线 → 净网，带分段内存缓存。"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from discriminator import diagnose_image, public_diagnosis
from line_extractors import extract_soft
from postprocess import finalize
from presets import merge_params
from smoothing import remove_texture
from tensor_norm import structure_tensor


@dataclass
class Phase1Result:
    original: np.ndarray
    smoothed: np.ndarray
    energy: np.ndarray
    soft: np.ndarray
    final: np.ndarray
    rgba: np.ndarray
    timings_ms: dict
    stats: dict
    cache_hits: dict
    keys: dict
    w_tex: np.ndarray
    diagnosis: dict
    single_line: np.ndarray | None = None
    band_images: dict | None = None
    ink: np.ndarray | None = None
    oklab: np.ndarray | None = None
    energy_norm: np.ndarray | None = None
    vector_scale: tuple[float, float] = (1.0, 1.0)


_POST_CACHE: OrderedDict[str, tuple] = OrderedDict()
_POST_LOCK = threading.Lock()
_POST_LIMIT = 8
_SINGLE_MEMO: OrderedDict[str, Phase1Result] = OrderedDict()
_SINGLE_LOCK = threading.Lock()
_SINGLE_LIMIT = 4
_BAND_ONLY = ("shadow_weight", "mid_weight", "highlight_weight", "band_blend_mode")


class StageCache:
    """按阶段缓存 numpy 结果。拖动阈值时纹理、张量和软墨线都直接复用。"""

    def __init__(self, limit: int = 3):
        self.limit = limit
        self.buckets = {
            "a": OrderedDict(),
            "b": OrderedDict(),
            "c": OrderedDict(),
        }

    def get(self, bucket: str, key: str):
        found = self.buckets[bucket].get(key)
        if found is not None:
            self.buckets[bucket].move_to_end(key)
        return found

    def put(self, bucket: str, key: str, value):
        store = self.buckets[bucket]
        store[key] = value
        store.move_to_end(key)
        while len(store) > self.limit:
            store.popitem(last=False)


_DIAG_CACHE: OrderedDict[str, dict] = OrderedDict()
_DIAG_LOCK = threading.Lock()
_DIAG_LIMIT = 8


def get_diagnosis(rgb_u8: np.ndarray, token: str) -> dict:
    """同一张图的体检按 token 缓存。重复请求直接复用，不再跑探针。"""
    with _DIAG_LOCK:
        found = _DIAG_CACHE.get(token)
        if found is not None:
            _DIAG_CACHE.move_to_end(token)
            return found
    diag = diagnose_image(rgb_u8)
    with _DIAG_LOCK:
        _DIAG_CACHE[token] = diag
        _DIAG_CACHE.move_to_end(token)
        while len(_DIAG_CACHE) > _DIAG_LIMIT:
            _DIAG_CACHE.popitem(last=False)
    return diag


def load_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_max(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side or max_side <= 0:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _finalize_cached(soft_ink: np.ndarray, params: dict, key: str):
    with _POST_LOCK:
        found = _POST_CACHE.get(key)
        if found is not None:
            _POST_CACHE.move_to_end(key)
            return found
    final_ink, info = finalize(soft_ink, params)
    payload = (final_ink, info, ink_to_rgb(final_ink), ink_to_rgba(final_ink))
    with _POST_LOCK:
        _POST_CACHE[key] = payload
        _POST_CACHE.move_to_end(key)
        while len(_POST_CACHE) > _POST_LIMIT:
            _POST_CACHE.popitem(last=False)
    return payload


def run_phase1(
    rgb_u8: np.ndarray,
    params: dict,
    cache: StageCache | None = None,
    token: str = "img",
    diagnosis: dict | None = None,
) -> Phase1Result:
    params = merge_params(params)
    diag = diagnosis if diagnosis is not None else get_diagnosis(rgb_u8, token)
    if str(params.get("band_blend_mode", "single")) == "single":
        result = _run_core(rgb_u8, params, cache, token, diag)
        result.single_line = result.final
        result.keys["single_line"] = result.keys["final_line"]
        return result

    from multiband import run_multiband

    memo_key = _single_memo_key(token, params)
    started = time.perf_counter()
    with _SINGLE_LOCK:
        base = _SINGLE_MEMO.get(memo_key)
        if base is not None:
            _SINGLE_MEMO.move_to_end(memo_key)
    if base is None:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="phase") as pool:
            single_fut = pool.submit(_run_core, rgb_u8, params, cache, token, diag)
            band_fut = pool.submit(run_multiband, rgb_u8, params, diag.get("w_tex_256"), token)
            base = single_fut.result()
            multi = band_fut.result()
        base.single_line = base.final
        base.keys = dict(base.keys)
        base.keys["single_line"] = base.keys["final_line"]
        with _SINGLE_LOCK:
            _SINGLE_MEMO[memo_key] = base
            _SINGLE_MEMO.move_to_end(memo_key)
            while len(_SINGLE_MEMO) > _SINGLE_LIMIT:
                _SINGLE_MEMO.popitem(last=False)
        cached_single = False
    else:
        multi = run_multiband(rgb_u8, params, diag.get("w_tex_256"), token)
        cached_single = True
    result = _clone_result(base)
    _apply_multiband(result, multi, started, cached_single)
    return result


def _single_memo_key(token: str, params: dict) -> str:
    payload = {key: value for key, value in params.items() if key not in _BAND_ONLY}
    payload["token"] = token
    return _hash(payload)


def _clone_result(src: Phase1Result) -> Phase1Result:
    dst = copy.copy(src)
    dst.keys = dict(src.keys)
    dst.stats = dict(src.stats)
    dst.timings_ms = dict(src.timings_ms)
    dst.cache_hits = dict(src.cache_hits)
    return dst


def _apply_multiband(result: Phase1Result, multi, started: float, cached_single: bool) -> None:
    result.final = multi.final_rgb
    result.rgba = multi.rgba
    result.band_images = multi.images
    result.keys["final_line"] = multi.keys["final_line"]
    result.keys["final_rgba"] = multi.keys["final_rgba"]
    for name, key in multi.keys.items():
        result.keys[name] = key
    result.stats["components"] = int(multi.stats["components"])
    result.stats["line_ratio"] = float(multi.stats["line_ratio"])
    result.stats["soft_p50"] = float(multi.stats["soft_p50"])
    result.stats["soft_p99"] = float(multi.stats["soft_p99"])
    result.stats["band_ratio"] = multi.stats["band_ratio"]
    result.timings_ms["bands"] = round(float(multi.pack_ms), 2)
    result.timings_ms["compose"] = round(float(multi.compose_ms), 2)
    result.timings_ms["total"] = round((time.perf_counter() - started) * 1000.0, 2)
    result.cache_hits["bands"] = bool(multi.pack_hit)
    result.ink = multi.ink
    if result.oklab is not None and multi.ink is not None and result.oklab.shape[:2] != multi.ink.shape[:2]:
        result.oklab = cv2.resize(
            result.oklab,
            (multi.ink.shape[1], multi.ink.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if (
        result.energy_norm is not None
        and result.oklab is not None
        and result.energy_norm.shape[:2] != result.oklab.shape[:2]
    ):
        result.energy_norm = cv2.resize(
            result.energy_norm,
            (result.oklab.shape[1], result.oklab.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if multi.ink is not None and multi.ink.size:
        result.vector_scale = (
            float(result.original.shape[1]) / float(multi.ink.shape[1]),
            float(result.original.shape[0]) / float(multi.ink.shape[0]),
        )
    if cached_single:
        result.cache_hits["texture"] = True
        result.cache_hits["tensor"] = True
        result.cache_hits["soft_ink"] = True
        result.timings_ms["texture"] = 0.0
        result.timings_ms["tensor"] = 0.0
        result.timings_ms["extract"] = 0.0
        result.timings_ms["post"] = 0.0


def _run_core(
    rgb_u8: np.ndarray,
    params: dict,
    cache: StageCache | None = None,
    token: str = "img",
    diagnosis: dict | None = None,
) -> Phase1Result:
    params = merge_params(params)
    diag = diagnosis if diagnosis is not None else get_diagnosis(rgb_u8, token)
    started = time.perf_counter()
    original = resize_max(rgb_u8, int(params["max_side"]))
    if original.dtype != np.uint8:
        original = np.clip(original, 0, 255).astype(np.uint8)

    hits = {"texture": False, "tensor": False, "soft_ink": False, "bands": False}
    a_key = _hash(
        {
            "token": token,
            "shape": list(original.shape),
            "texture_method": params["texture_method"],
            "texture_scale": round(float(params["texture_scale"]), 5),
            "max_side": int(params["max_side"]),
            "spatial_adaptive": bool(params["spatial_adaptive"]),
            "spatial_strength": round(float(params["spatial_strength"]), 5),
        }
    )
    t0 = time.perf_counter()
    cached_a = cache.get("a", a_key) if cache else None
    if cached_a is None:
        smoothed, oklab = remove_texture(
            original,
            params["texture_method"],
            float(params["texture_scale"]),
            w_tex_256=diag["w_tex_256"],
            spatial_adaptive=bool(params["spatial_adaptive"]),
            spatial_strength=float(params["spatial_strength"]),
        )
        if cache:
            cache.put("a", a_key, (smoothed, oklab))
    else:
        smoothed, oklab = cached_a
        hits["texture"] = True
    texture_ms = (time.perf_counter() - t0) * 1000.0

    b_key = _hash(
        {
            "a": a_key,
            "chroma_weight": round(float(params["chroma_weight"]), 5),
            "sigma_g": round(float(params["sigma_g"]), 5),
            "sigma_norm": round(float(params["sigma_norm"]), 5),
            "quietness": round(float(params["quietness"]), 5),
            "contrast_norm": round(float(params["contrast_norm"]), 5),
        }
    )
    t0 = time.perf_counter()
    cached_b = cache.get("b", b_key) if cache else None
    if cached_b is None:
        fields = structure_tensor(oklab, params)
        energy = colorize_energy(fields.energy_norm)
        if cache:
            cache.put("b", b_key, (fields, energy))
    else:
        fields, energy = cached_b
        hits["tensor"] = True
    tensor_ms = (time.perf_counter() - t0) * 1000.0

    c_key = _hash({"b": b_key, **_extract_signature(params)})
    t0 = time.perf_counter()
    cached_c = cache.get("c", c_key) if cache else None
    if cached_c is None:
        soft_ink = extract_soft(params["extract_mode"], fields, params, smoothed)
        soft_rgb = ink_to_rgb(soft_ink)
        if cache:
            cache.put("c", c_key, (soft_ink, soft_rgb))
    else:
        soft_ink, soft_rgb = cached_c
        hits["soft_ink"] = True
    extract_ms = (time.perf_counter() - t0) * 1000.0

    d_key = _hash(
        {
            "c": c_key,
            "threshold_mode": params["threshold_mode"],
            "thresh_low": round(float(params["thresh_low"]), 5),
            "thresh_high": round(float(params["thresh_high"]), 5),
            "min_component_size": int(params["min_component_size"]),
            "min_span": round(float(params["min_span"]), 4),
            "thin_1px": bool(params["thin_1px"]),
        }
    )
    t0 = time.perf_counter()
    final_ink, info, final_rgb, rgba = _finalize_cached(soft_ink, params, d_key)
    post_ms = (time.perf_counter() - t0) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0
    stats = {
        "width": int(original.shape[1]),
        "height": int(original.shape[0]),
        "components": int(info["components"]),
        "line_ratio": float(info["line_ratio"]),
        "soft_p50": _pct(soft_ink, 50.0),
        "soft_p99": _pct(soft_ink, 99.0),
    }
    timings = {
        "texture": round(texture_ms, 2),
        "tensor": round(tensor_ms, 2),
        "extract": round(extract_ms, 2),
        "post": round(post_ms, 2),
        "total": round(total_ms, 2),
        "diagnose": float(diag["diagnose_ms"]),
    }
    keys = {
        "original": _hash({"token": token, "shape": list(original.shape), "kind": "original"}),
        "smoothed": a_key,
        "energy": b_key,
        "soft_ink": c_key,
        "final_line": d_key + "_line",
        "final_rgba": d_key + "_rgba",
        "w_tex": _hash({"token": token, "kind": "w_tex"}),
    }
    w_full = cv2.resize(
        np.asarray(diag["w_tex_256"], dtype=np.float32),
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    w_rgb = colorize_wtex(np.clip(w_full, 0.0, 1.0))
    return Phase1Result(
        original=original,
        smoothed=smoothed,
        energy=energy,
        soft=soft_rgb,
        final=final_rgb,
        rgba=rgba,
        timings_ms=timings,
        stats=stats,
        cache_hits=hits,
        keys=keys,
        w_tex=w_rgb,
        diagnosis=public_diagnosis(diag),
        ink=final_ink,
        oklab=oklab,
        energy_norm=np.ascontiguousarray(fields.energy_norm, dtype=np.float32),
        vector_scale=(1.0, 1.0),
    )


def ink_to_rgb(ink: np.ndarray) -> np.ndarray:
    gray = np.clip(np.round((1.0 - ink) * 255.0), 0, 255).astype(np.uint8)
    return np.dstack([gray, gray, gray])


def ink_to_rgba(ink: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.round(ink * 255.0), 0, 255).astype(np.uint8)
    rgba = np.zeros((ink.shape[0], ink.shape[1], 4), np.uint8)
    rgba[..., 3] = alpha
    return rgba


def colorize_wtex(weight: np.ndarray) -> np.ndarray:
    """暗色是受保护的结构，亮色是要抹平的纹理。"""
    x = np.clip(weight, 0.0, 1.0).astype(np.float32)
    stops = np.array(
        [
            [8, 10, 18],
            [32, 64, 138],
            [214, 148, 62],
            [255, 248, 236],
        ],
        dtype=np.float32,
    )
    scaled = x * (len(stops) - 1)
    index = np.clip(np.floor(scaled).astype(np.int32), 0, len(stops) - 2)
    frac = (scaled - index.astype(np.float32))[..., None]
    rgb = stops[index] * (1.0 - frac) + stops[index + 1] * frac
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def colorize_energy(energy: np.ndarray) -> np.ndarray:
    x = np.clip(energy, 0.0, 1.0).astype(np.float32)
    stops = np.array(
        [
            [10, 12, 18],
            [42, 22, 92],
            [186, 46, 98],
            [242, 156, 48],
            [255, 246, 230],
        ],
        dtype=np.float32,
    )
    scaled = x * (len(stops) - 1)
    index = np.clip(np.floor(scaled).astype(np.int32), 0, len(stops) - 2)
    frac = (scaled - index.astype(np.float32))[..., None]
    rgb = stops[index] * (1.0 - frac) + stops[index + 1] * frac
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def _extract_signature(params: dict) -> dict:
    keys = (
        "extract_mode",
        "line_width",
        "coh_power",
        "ao_suppression",
        "flow_coherence",
        "fdog_sigma",
        "fdog_rho",
        "fdog_eps",
        "fdog_phi",
        "eta",
        "etf_iters",
        "etf_radius",
        "contrast_norm",
        "chroma_weight",
        "sigma_norm",
    )
    out = {}
    for key in keys:
        value = params[key]
        if isinstance(value, float):
            out[key] = round(value, 5)
        else:
            out[key] = value
    return out


def _hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:20]


def _pct(arr: np.ndarray, q: float) -> float:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    step = max(1, flat.size // 250_000)
    return round(float(np.percentile(flat[::step], q)), 4)
