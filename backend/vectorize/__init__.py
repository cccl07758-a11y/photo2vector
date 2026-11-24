"""阶段二：紧凑骨架图上的断线焊接与极简贝塞尔矢量化。"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import cv2
import numpy as np

from vectorize.export import build_output
from vectorize.fit import fit_stroke
from vectorize.skeleton import extract_skeleton_graph, prune_barbs
from vectorize.strokes import collapse_highlights, score_strokes, trace_strokes, weld_gaps

DEFAULTS = {
    "fit_tol": 1.8,
    "line_snap_scale": 0.85,
    "gap_max_px": 22.0,
    "detail_keep": 0.92,
    "min_stroke_len": 12.0,
    "stroke_scale": 1.0,
    "cusp_angle_deg": 42.0,
    "max_turn_deg": 45.0,
    "highlight_collapse": True,
    "render_style": "center",
    "show_secondary": True,
    "prune_ratio": 1.85,
    "max_barb_len": 11.0,
    "mask_threshold": 0.32,
}

_RANGES = {
    "fit_tol": (0.6, 6.0),
    "line_snap_scale": (0.35, 2.6),
    "gap_max_px": (0.0, 40.0),
    "detail_keep": (0.0, 1.0),
    "min_stroke_len": (2.0, 80.0),
    "stroke_scale": (0.4, 3.0),
    "cusp_angle_deg": (20.0, 80.0),
    "max_turn_deg": (15.0, 80.0),
    "prune_ratio": (0.8, 4.0),
    "max_barb_len": (4.0, 30.0),
    "mask_threshold": (0.05, 0.9),
}

_LOCK = threading.Lock()
_TOPO: OrderedDict[tuple, tuple] = OrderedDict()
_FIT: OrderedDict[tuple, list] = OrderedDict()
_LIMIT = 6


def parse_params(raw: dict | None) -> dict:
    params = dict(DEFAULTS)
    for key, value in dict(raw or {}).items():
        if key not in params:
            continue
        params[key] = _coerce(key, value, params[key])
    if params["render_style"] not in {"center", "taper"}:
        params["render_style"] = "center"
    return params


def vectorize(
    ink: np.ndarray,
    oklab: np.ndarray,
    params: dict | None = None,
    display_size: tuple[int, int] | None = None,
    source_key: str = "",
) -> dict:
    started = time.perf_counter()
    user = parse_params(params)
    ink = np.nan_to_num(np.asarray(ink, dtype=np.float32), nan=0.0)
    lab = _match_lab(oklab, ink.shape)
    height, width = ink.shape[:2]
    disp_w, disp_h = display_size or (width, height)
    sx = float(disp_w) / max(width, 1)
    sy = float(disp_h) / max(height, 1)
    px = max((sx + sy) * 0.5, 1e-6)
    internal = dict(user)
    for key in ("fit_tol", "gap_max_px", "min_stroke_len", "max_barb_len"):
        internal[key] = float(user[key]) / px
    internal["ortho_min"] = 20.0 / px
    strokes, collapsed = _topology(ink, lab, internal, source_key)
    visible = _select(strokes, float(user["detail_keep"]), bool(user["show_secondary"]))
    fitted = _fit_paths(visible, internal, source_key)
    elapsed = (time.perf_counter() - started) * 1000.0
    output = build_output(fitted, sx, sy, int(disp_w), int(disp_h), user, collapsed, elapsed)
    output["params"] = user
    output["timings_ms"] = {"stage2": output["vector_stats"]["stage2_ms"]}
    return output


def _match_lab(oklab: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    lab = np.nan_to_num(np.asarray(oklab, dtype=np.float32), nan=0.0)
    if lab.ndim != 3:
        lab = np.zeros((shape[0], shape[1], 3), np.float32)
    if lab.shape[0] != shape[0] or lab.shape[1] != shape[1]:
        lab = cv2.resize(lab, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(lab, dtype=np.float32)


def _topology(ink, lab, internal, source_key: str):
    key = (
        source_key,
        round(float(internal["gap_max_px"]), 3),
        round(float(internal["min_stroke_len"]), 3),
        round(float(internal["max_turn_deg"]), 2),
        round(float(internal["prune_ratio"]), 3),
        round(float(internal["max_barb_len"]), 3),
        round(float(internal["mask_threshold"]), 3),
        bool(internal["highlight_collapse"]),
    )
    if source_key:
        with _LOCK:
            found = _TOPO.get(key)
            if found is not None:
                _TOPO.move_to_end(key)
                return found
    graph = extract_skeleton_graph(ink, float(internal["mask_threshold"]))
    prune_barbs(graph, internal)
    strokes = trace_strokes(graph, float(internal["max_turn_deg"]))
    strokes, _welds = weld_gaps(strokes, lab, float(internal["gap_max_px"]))
    strokes, collapsed = collapse_highlights(strokes, lab, bool(internal["highlight_collapse"]))
    score_strokes(strokes, lab)
    payload = (strokes, int(collapsed))
    if source_key:
        with _LOCK:
            _TOPO[key] = payload
            _TOPO.move_to_end(key)
            while len(_TOPO) > _LIMIT:
                _TOPO.popitem(last=False)
    return payload


def _select(strokes, detail_keep: float, show_secondary: bool):
    floor = (1.0 - float(np.clip(detail_keep, 0.0, 1.0))) * 0.50
    kept = []
    for stroke in strokes:
        if stroke.tier != "primary" and float(stroke.score) < floor:
            continue
        if stroke.tier != "primary" and not show_secondary:
            continue
        kept.append(stroke)
    return kept


def _fit_paths(strokes, internal, source_key: str):
    # 拟合缓存按拓扑后的笔画身份，细节滑块只过滤、不重拟合。
    ident = tuple(id(stroke) for stroke in strokes)
    key = (
        source_key,
        ident,
        round(float(internal["fit_tol"]), 3),
        round(float(internal["line_snap_scale"]), 3),
        round(float(internal["cusp_angle_deg"]), 2),
        round(float(internal["ortho_min"]), 3),
    )
    if source_key:
        with _LOCK:
            found = _FIT.get(key)
            if found is not None:
                _FIT.move_to_end(key)
                return found
    fitted = []
    for stroke in strokes:
        path = fit_stroke(stroke, internal)
        if path is not None:
            fitted.append(path)
    if source_key:
        with _LOCK:
            _FIT[key] = fitted
            _FIT.move_to_end(key)
            while len(_FIT) > _LIMIT:
                _FIT.popitem(last=False)
    return fitted


def _coerce(key: str, value, fallback):
    try:
        if key in {"highlight_collapse", "show_secondary"}:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if key == "render_style":
            text = str(value)
            if text in {"taper", "envelope", "brush"}:
                return "taper"
            return "center"
        lo, hi = _RANGES[key]
        return float(min(hi, max(lo, float(value))))
    except (TypeError, ValueError):
        return fallback
