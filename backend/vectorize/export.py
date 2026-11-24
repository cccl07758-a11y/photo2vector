"""标准 SVG 与 Paper.js 显微镜用的矢量数据。"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from vectorize.geom import eval_cubic


def build_output(
    paths: list[dict],
    sx: float,
    sy: float,
    width: int,
    height: int,
    params: dict,
    collapsed: int,
    elapsed_ms: float,
) -> dict:
    px = (sx + sy) * 0.5
    visible = [_scale_path(path, sx, sy, px, float(params["stroke_scale"]), index) for index, path in enumerate(paths)]
    svg = _svg(visible, width, height, params)
    stats = _stats(visible, collapsed, elapsed_ms)
    return {
        "svg_text": svg,
        "vector_data": {"width": int(width), "height": int(height), "paths": visible},
        "vector_stats": stats,
    }


def _scale_path(path: dict, sx: float, sy: float, px: float, stroke_scale: float, index: int) -> dict:
    commands = [_scale_cmd(cmd, sx, sy) for cmd in path["commands"]]
    anchors = [_scale_anchor(anchor, sx, sy) for anchor in path["anchors"]]
    bridges = [_scale_bridge(bridge, sx, sy) for bridge in path["bridges"]]
    full = 2.0 * float(path["median_radius"]) * px
    if full < 1.7:
        base = 1.4
    elif full < 2.7:
        base = 2.2
    else:
        base = 3.2
    return {
        "id": index,
        "tier": path["tier"],
        "score": round(float(path["score"]), 4),
        "stroke_width": round(base * stroke_scale, 3),
        "closed": bool(path["closed"]),
        "commands": commands,
        "anchors": anchors,
        "welded_bridges": bridges,
        "width_profile": [float(value) * px for value in path["width_profile"]],
    }


def _scale_cmd(cmd: dict, sx: float, sy: float) -> dict:
    out = {"op": cmd["op"]}
    if "p" in cmd:
        out["p"] = _pt(cmd["p"], sx, sy)
    if "h1" in cmd:
        out["h1"] = _pt(cmd["h1"], sx, sy)
    if "h2" in cmd:
        out["h2"] = _pt(cmd["h2"], sx, sy)
    return out


def _scale_anchor(anchor: dict, sx: float, sy: float) -> dict:
    out = {"p": _pt(anchor["p"], sx, sy), "kind": anchor["kind"]}
    if "h_in" in anchor:
        out["h_in"] = _pt(anchor["h_in"], sx, sy)
    if "h_out" in anchor:
        out["h_out"] = _pt(anchor["h_out"], sx, sy)
    return out


def _scale_bridge(bridge: dict, sx: float, sy: float) -> dict:
    return {
        "p0": _pt(bridge["p0"], sx, sy),
        "h1": _pt(bridge["h1"], sx, sy),
        "h2": _pt(bridge["h2"], sx, sy),
        "p1": _pt(bridge["p1"], sx, sy),
        "kind": bridge.get("kind", "gap"),
    }


def _pt(point, sx: float, sy: float) -> list[float]:
    return [round(float(point[0]) * sx, 2), round(float(point[1]) * sy, 2)]


def _svg(paths: list[dict], width: int, height: int, params: dict) -> str:
    tapered = params["render_style"] == "taper"
    show_secondary = bool(params["show_secondary"])
    primary = [path for path in paths if path["tier"] == "primary"]
    secondary = [path for path in paths if path["tier"] != "primary"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<g id="strokes-primary" display="{"none" if tapered else "inline"}">',
        *[_stroke_path(path) for path in primary],
        "</g>",
        f'<g id="strokes-secondary" display="{"none" if tapered or not show_secondary else "inline"}">',
        *[_stroke_path(path) for path in secondary],
        "</g>",
        f'<g id="tapered-envelopes" display="{"inline" if tapered else "none"}">',
    ]
    if tapered:
        shown = primary + (secondary if show_secondary else [])
        for path in shown:
            env = _envelope(path)
            if env:
                lines.append(env)
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines)


def _stroke_path(path: dict) -> str:
    width = path["stroke_width"]
    return (
        f'<path id="s{path["id"]}" d="{_d(path["commands"])}" fill="none" stroke="#141417" '
        f'stroke-width="{_num(width)}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def _d(commands: list[dict]) -> str:
    parts = []
    for cmd in commands:
        op = cmd["op"]
        if op == "M":
            parts.append(f'M {_pair(cmd["p"])}')
        elif op == "L":
            parts.append(f'L {_pair(cmd["p"])}')
        elif op == "C":
            parts.append(f'C {_pair(cmd["h1"])} {_pair(cmd["h2"])} {_pair(cmd["p"])}')
        elif op == "Z":
            parts.append("Z")
    return " ".join(parts)


def _envelope(path: dict) -> str | None:
    samples = _sample_centerline(path["commands"], 2.0)
    if len(samples) < 4:
        return None
    profile = np.asarray(path["width_profile"], dtype=np.float32)
    if len(profile) < 2:
        profile = np.full(len(samples), float(profile[0] if len(profile) else path["stroke_width"]), np.float32)
    else:
        slots = np.linspace(0.0, 1.0, len(samples), dtype=np.float32)
        base = np.linspace(0.0, 1.0, len(profile), dtype=np.float32)
        profile = np.interp(slots, base, profile).astype(np.float32)
    if len(profile) >= 5:
        profile = gaussian_filter1d(profile, 1.15).astype(np.float32)
    if not path["closed"]:
        taper = int(np.clip(round(len(samples) * 0.12), 4, 20))
        ramp = np.linspace(0.06, 1.0, taper, dtype=np.float32) ** 1.05
        profile = profile.copy()
        profile[:taper] *= ramp
        profile[-taper:] *= ramp[::-1]
    center = samples
    if len(center) >= 5:
        center = gaussian_filter1d(samples, 0.75, axis=0).astype(np.float32)
        if not path["closed"]:
            center[0] = samples[0]
            center[-1] = samples[-1]
    tang = np.gradient(center, axis=0)
    tang = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    normal = np.stack((-tang[:, 1], tang[:, 0]), axis=1)
    half = (profile * 0.5)[:, None]
    left = center + normal * half
    right = center - normal * half
    if path["closed"]:
        poly = np.vstack((left[:-1], right[:-1][::-1]))
    else:
        poly = np.vstack((left, right[::-1]))
    body = " ".join(f"{_num(point[0])} {_num(point[1])}" for point in poly)
    return f'<path d="M {body} Z" fill="#141417"/>'


def _sample_centerline(commands: list[dict], step: float) -> np.ndarray:
    points: list[np.ndarray] = []
    cursor = None
    for cmd in commands:
        if cmd["op"] == "M":
            cursor = np.asarray(cmd["p"], dtype=np.float32)
            points.append(cursor)
        elif cmd["op"] == "L" and cursor is not None:
            end = np.asarray(cmd["p"], dtype=np.float32)
            dist = float(np.linalg.norm(end - cursor))
            count = max(1, int(np.ceil(dist / step)))
            for index in range(1, count + 1):
                points.append(cursor + (end - cursor) * (index / count))
            cursor = end
        elif cmd["op"] == "C" and cursor is not None:
            end = np.asarray(cmd["p"], dtype=np.float32)
            samples = eval_cubic(
                np.linspace(0.0, 1.0, max(4, int(np.linalg.norm(end - cursor) / step) + 1), dtype=np.float32),
                cursor,
                np.asarray(cmd["h1"], dtype=np.float32),
                np.asarray(cmd["h2"], dtype=np.float32),
                end,
            )
            points.extend(samples[1:])
            cursor = end
        elif cmd["op"] == "Z" and points:
            points.append(np.asarray(points[0], dtype=np.float32))
    if len(points) < 2:
        return np.zeros((0, 2), np.float32)
    return np.stack(points).astype(np.float32)


def _stats(paths: list[dict], collapsed: int, elapsed_ms: float) -> dict:
    primary = sum(path["tier"] == "primary" for path in paths)
    anchors = cusps = smooths = lines = cubics = welds = 0
    for path in paths:
        for anchor in path["anchors"]:
            anchors += 1
            if anchor["kind"] == "cusp":
                cusps += 1
            elif anchor["kind"] == "smooth":
                smooths += 1
        for cmd in path["commands"]:
            if cmd["op"] == "L":
                lines += 1
            elif cmd["op"] == "C":
                cubics += 1
        welds += len(path["welded_bridges"])
    total = len(paths)
    denom = lines + cubics
    return {
        "total_paths": total,
        "primary_paths": primary,
        "secondary_paths": total - primary,
        "total_anchors": anchors,
        "cusp_anchors": cusps,
        "smooth_anchors": smooths,
        "avg_anchors_per_path": round(anchors / total, 2) if total else 0.0,
        "line_segments": lines,
        "cubic_segments": cubics,
        "straight_ratio": round(lines / denom, 4) if denom else 0.0,
        "welded_count": welds,
        "collapsed_double_lines": int(collapsed),
        "stage2_ms": round(float(elapsed_ms), 2),
    }


def _pair(point) -> str:
    return f"{_num(point[0])} {_num(point[1])}"


def _num(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    if text in {"-0", ""}:
        return "0"
    return text
