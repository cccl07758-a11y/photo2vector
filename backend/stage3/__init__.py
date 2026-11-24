"""阶段三 + 阶段四：主体分区、虚实分流、调色盘与叠纸矢量插画。"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import cv2
import numpy as np

from stage3.edge_routing import dam_polylines, measure, select, stroke_hex
from stage3.paperstack import assemble, build_shapes, rasterize
from stage3.rag_palette import build as build_partition
from stage3.rag_palette import flat_rgb
from stage3.saliency import compute as compute_saliency

DEFAULTS = {
    "palette_k": 14,
    "bg_simplify": 0.095,
    "fg_detail": 0.032,
    "shape_fit_tol": 2.8,
    "solid_line_gate": 0.28,
    "line_color_mode": "colored",
    "show_bg_shapes": True,
    "show_fg_shapes": True,
    "show_strokes": True,
}

_RANGES = {
    "palette_k": (6, 22),
    "bg_simplify": (0.03, 0.22),
    "fg_detail": (0.008, 0.09),
    "shape_fit_tol": (0.6, 6.0),
    "solid_line_gate": (0.0, 0.75),
}

_LOCK = threading.Lock()
_SALIENCY: OrderedDict[tuple, np.ndarray] = OrderedDict()
_ROUTE: OrderedDict[tuple, list] = OrderedDict()
_REGION: OrderedDict[tuple, dict] = OrderedDict()
_GEOM: OrderedDict[tuple, tuple] = OrderedDict()
_LIMIT = 4


def parse_params(raw: dict | None) -> dict:
    params = dict(DEFAULTS)
    for key, value in dict(raw or {}).items():
        if key not in params:
            continue
        params[key] = _coerce(key, value, params[key])
    if params["line_color_mode"] not in {"colored", "black", "none"}:
        params["line_color_mode"] = "colored"
    params["palette_k"] = int(round(params["palette_k"]))
    return params


def illustrate(
    oklab: np.ndarray,
    paths: list[dict],
    params: dict | None = None,
    *,
    display_size: tuple[int, int] | None = None,
    source_key: str = "",
    paths_key: str = "",
    w_tex: np.ndarray | None = None,
    energy_norm: np.ndarray | None = None,
) -> dict:
    started = time.perf_counter()
    user = parse_params(params)
    lab_src = np.nan_to_num(np.asarray(oklab, dtype=np.float32), nan=0.0)
    src_h, src_w = lab_src.shape[:2]
    if display_size is None:
        disp_w, disp_h = src_w, src_h
    else:
        disp_w, disp_h = int(display_size[0]), int(display_size[1])
    work_w, work_h = _work_size(src_w, src_h)
    if (work_w, work_h) != (src_w, src_h):
        lab = cv2.resize(lab_src, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        lab = lab_src
    hits = {"saliency": False, "routing": False, "regions": False, "geometry": False}
    t0 = time.perf_counter()
    saliency, hits["saliency"] = _saliency(lab, w_tex, energy_norm, source_key, (work_w, work_h))
    saliency_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    measured, hits["routing"] = _routing(paths, lab, saliency, (disp_w, disp_h), source_key, paths_key)
    routing_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    partition, hits["regions"] = _regions(lab, saliency, measured, user, source_key, paths_key, (work_w, work_h))
    region_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    shapes, ground, shape_stats, hits["geometry"] = _geometry(
        partition, user, (disp_w, disp_h), source_key, paths_key, (work_w, work_h)
    )
    geometry_ms = (time.perf_counter() - t0) * 1000.0
    kept = select(measured, float(user["solid_line_gate"]), (disp_w, disp_h))
    mode = user["line_color_mode"]
    for index, stroke in enumerate(kept):
        stroke["index"] = index
        stroke["mode"] = mode
        stroke["hex"] = stroke_hex(stroke["darker_lab"], mode)
    visible = kept if mode != "none" and user["show_strokes"] else []
    svg = assemble(shapes, kept, user, (disp_w, disp_h), ground)
    paint_shapes = shapes if user["show_bg_shapes"] or user["show_fg_shapes"] else []
    if not user["show_bg_shapes"] or not user["show_fg_shapes"]:
        paint_shapes = [
            shape
            for shape in shapes
            if (shape["group"] == "bg" and user["show_bg_shapes"]) or (shape["group"] != "bg" and user["show_fg_shapes"])
        ]
    shapes_rgb, shape_coverage = rasterize(shapes, [], (disp_w, disp_h), ground, draw_strokes=False)
    final_rgb, _covered = rasterize(
        paint_shapes,
        visible,
        (disp_w, disp_h),
        ground,
        draw_shapes=bool(paint_shapes) or True,
        draw_strokes=bool(visible),
    )
    if not paint_shapes and not user["show_bg_shapes"]:
        final_rgb, _covered = rasterize([], visible, (disp_w, disp_h), ground, draw_shapes=False, draw_strokes=bool(visible), paper=True)
    stroke_rgb, _ignored = rasterize([], visible, (disp_w, disp_h), ground, draw_shapes=False, draw_strokes=True, paper=True)
    heat = _heatmap(saliency, (disp_w, disp_h))
    flat = cv2.resize(flat_rgb(partition), (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    elapsed = (time.perf_counter() - started) * 1000.0
    total_paths = max(len(measured), 1)
    retained = len(kept) / total_paths if measured else 0.0
    # 底色矩形托住画布边缘；形状并集覆盖率单独统计，避免把托底误当成缝。
    visual_coverage = 1.0 if ground else shape_coverage
    stats = {
        "shape_count": shape_stats["shape_count"],
        "palette_hex": [item["hex"] for item in partition["palette"]],
        "shape_anchors": shape_stats["shape_anchors"],
        "stroke_paths": 0 if mode == "none" else len(kept),
        "retained_stroke_ratio": round(float(retained), 4),
        "coverage_ratio": round(float(visual_coverage), 4),
        "shape_coverage": round(float(shape_coverage), 4),
        "stage3_ms": round(elapsed, 2),
        "primary_strokes": sum(1 for item in kept if item["paint_tier"] == "primary"),
        "secondary_strokes": sum(1 for item in kept if item["paint_tier"] != "primary"),
    }
    return {
        "svg_text": svg,
        "palette": partition["palette"],
        "illustration_stats": stats,
        "shapes": shapes,
        "previews": {
            "saliency": heat,
            "flat": flat,
            "strokes": stroke_rgb,
            "shapes": shapes_rgb,
            "final": final_rgb,
        },
        "params": user,
        "cache": hits,
        "timings_ms": {
            "saliency": round(saliency_ms, 2),
            "routing": round(routing_ms, 2),
            "palette": round(region_ms, 2),
            "geometry": round(geometry_ms, 2),
            "stage3": round(elapsed, 2),
        },
        "size": {"width": disp_w, "height": disp_h},
    }


def _work_size(width: int, height: int, longest: int = 768) -> tuple[int, int]:
    long_side = max(width, height)
    if long_side <= longest:
        return width, height
    scale = longest / float(long_side)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _saliency(lab, w_tex, energy, source_key: str, work_size: tuple[int, int]):
    key = (source_key, work_size)
    if source_key:
        with _LOCK:
            found = _SALIENCY.get(key)
            if found is not None:
                _SALIENCY.move_to_end(key)
                return found, True
    field = compute_saliency(lab, w_tex, energy)
    if source_key:
        with _LOCK:
            _SALIENCY[key] = field
            _SALIENCY.move_to_end(key)
            while len(_SALIENCY) > _LIMIT:
                _SALIENCY.popitem(last=False)
    return field, False


def _routing(paths, lab, saliency, display_size, source_key: str, paths_key: str):
    key = (source_key, paths_key, lab.shape)
    if source_key and paths_key:
        with _LOCK:
            found = _ROUTE.get(key)
            if found is not None:
                _ROUTE.move_to_end(key)
                return found, True
    measured = measure(paths, lab, saliency, display_size)
    if source_key and paths_key:
        with _LOCK:
            _ROUTE[key] = measured
            _ROUTE.move_to_end(key)
            while len(_ROUTE) > _LIMIT:
                _ROUTE.popitem(last=False)
    return measured, False


def _regions(lab, saliency, measured, params, source_key: str, paths_key: str, work_size: tuple[int, int]):
    key = (
        source_key,
        paths_key,
        work_size,
        round(float(params["bg_simplify"]), 4),
        round(float(params["fg_detail"]), 4),
        int(params["palette_k"]),
    )
    if source_key and paths_key:
        with _LOCK:
            found = _REGION.get(key)
            if found is not None:
                _REGION.move_to_end(key)
                return found, True
    partition = build_partition(lab, saliency, dam_polylines(measured), params)
    if source_key and paths_key:
        with _LOCK:
            _REGION[key] = partition
            _REGION.move_to_end(key)
            while len(_REGION) > _LIMIT:
                _REGION.popitem(last=False)
    return partition, False


def _geometry(partition, params, display_size, source_key: str, paths_key: str, work_size: tuple[int, int]):
    key = (
        source_key,
        paths_key,
        work_size,
        display_size,
        round(float(params["bg_simplify"]), 4),
        round(float(params["fg_detail"]), 4),
        int(params["palette_k"]),
        round(float(params["shape_fit_tol"]), 3),
    )
    if source_key and paths_key:
        with _LOCK:
            found = _GEOM.get(key)
            if found is not None:
                _GEOM.move_to_end(key)
                return (*found, True)
    shapes, ground, stats = build_shapes(partition, float(params["shape_fit_tol"]), display_size)
    if source_key and paths_key:
        with _LOCK:
            _GEOM[key] = (shapes, ground, stats)
            _GEOM.move_to_end(key)
            while len(_GEOM) > _LIMIT:
                _GEOM.popitem(last=False)
    return shapes, ground, stats, False


def _heatmap(saliency: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    field = saliency
    if field.shape[1] != width or field.shape[0] != height:
        field = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)
    gray = np.clip(np.round(field * 255.0), 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _coerce(key: str, value, fallback):
    try:
        if key in {"show_bg_shapes", "show_fg_shapes", "show_strokes"}:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if key == "line_color_mode":
            text = str(value).strip().lower()
            return text if text in {"colored", "black", "none"} else fallback
        if key == "palette_k":
            lo, hi = _RANGES[key]
            return int(min(hi, max(lo, round(float(value)))))
        lo, hi = _RANGES[key]
        return float(min(hi, max(lo, float(value))))
    except (TypeError, ValueError):
        return fallback
