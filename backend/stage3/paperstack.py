"""全画布托底、父剪影实心并集，以及单向向内 Under-lap 的闭合贝塞尔。"""

from __future__ import annotations

import math

import cv2
import numpy as np

from stage3.edge_routing import _flatten, stroke_hex
from vectorize.fit import _fit_cubic


GROUPS = {
    "bg": "bg-shapes",
    "base": "fg-base-silhouettes",
    "detail": "fg-detail-shapes",
}


def build_shapes(partition: dict, shape_fit_tol: float, display_size: tuple[int, int]) -> tuple[list[dict], str, dict]:
    labels = partition["labels"]
    height, width = labels.shape
    disp_w, disp_h = display_size
    area = partition["area"]
    sal = partition["saliency"]
    chroma = partition["chroma"]
    parent, minx, miny, maxx, maxy = _parents(labels, area)
    depth = _depths(parent)
    total = float(max(height * width, 1))
    comps = []
    for index in range(len(area)):
        if area[index] <= 0 or maxx[index] < 0:
            continue
        touches = bool(minx[index] <= 0 or miny[index] <= 0 or maxx[index] >= width - 1 or maxy[index] >= height - 1)
        comps.append(
            {
                "id": index,
                "area": float(area[index]),
                "saliency": float(sal[index]),
                "chroma": float(chroma[index]),
                "fill": partition["hex"][index],
                "parent": int(parent[index]),
                "depth": int(depth[index]),
                "touches": touches,
                "is_fg": bool(sal[index] >= 0.24 or (chroma[index] > 0.075 and sal[index] >= 0.14)),
            }
        )
    _assign_groups(comps)
    ordered = sorted(comps, key=lambda item: (_band(item), item["depth"], -item["area"], item["chroma"], item["id"]))
    for z, comp in enumerate(ordered, start=1):
        comp["z"] = z
    by_id = {comp["id"]: comp for comp in comps}
    children: dict[int, list[int]] = {comp["id"]: [] for comp in comps}
    for comp in comps:
        parent_id = comp["parent"]
        if parent_id in by_id and parent_id != comp["id"]:
            children[parent_id].append(comp["id"])
    descendants = {comp["id"]: _collect(comp["id"], children) for comp in comps}
    z_lut = np.zeros(int(labels.max()) + 1, np.int16)
    for comp in comps:
        z_lut[comp["id"]] = comp["z"]
    z_map = z_lut[labels]
    scale = 0.5 * ((width / max(disp_w, 1)) + (height / max(disp_h, 1)))
    radius = max(2, int(math.ceil((1.25 * float(shape_fit_tol) + 1.0) * max(scale, 1e-6))))
    shapes = []
    anchor_total = 0
    for comp in ordered:
        mask_ids = [comp["id"], *descendants[comp["id"]]]
        contours = _tucked_contours(
            labels,
            z_map,
            mask_ids,
            comp["z"],
            radius,
            (minx, miny, maxx, maxy),
        )
        for contour in contours:
            commands, anchors = _vectorize_contour(contour, shape_fit_tol, (disp_w / width, disp_h / height))
            if not commands:
                continue
            shapes.append(
                {
                    "id": len(shapes),
                    "region": comp["id"],
                    "group": comp["group"],
                    "fill": comp["fill"],
                    "z": comp["z"],
                    "commands": commands,
                    "anchors": anchors,
                }
            )
            anchor_total += len(anchors)
    ground = _ground_color(comps)
    stats = {
        "shape_count": {
            "bg": sum(1 for shape in shapes if shape["group"] == "bg"),
            "fg": sum(1 for shape in shapes if shape["group"] != "bg"),
        },
        "shape_anchors": anchor_total,
    }
    return shapes, ground, stats


def assemble(shapes: list[dict], strokes: list[dict], params: dict, size: tuple[int, int], ground: str) -> str:
    width, height = size
    buckets = {key: [] for key in GROUPS}
    for shape in shapes:
        buckets.get(shape["group"], buckets["detail"]).append(shape)
    show_bg = bool(params.get("show_bg_shapes", True))
    show_fg = bool(params.get("show_fg_shapes", True))
    show_strokes = bool(params.get("show_strokes", True)) and params.get("line_color_mode") != "none"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<g id="bg-shapes" display="{_display(show_bg)}">',
        f'<rect id="canvas-ground" x="0" y="0" width="{width}" height="{height}" fill="{ground}"/>',
    ]
    for shape in buckets["bg"]:
        lines.append(_fill_path(shape))
    lines.append("</g>")
    lines.append(f'<g id="fg-base-silhouettes" display="{_display(show_fg)}">')
    for shape in buckets["base"]:
        lines.append(_fill_path(shape))
    lines.append("</g>")
    lines.append(f'<g id="fg-detail-shapes" display="{_display(show_fg)}">')
    for shape in buckets["detail"]:
        lines.append(_fill_path(shape))
    lines.append("</g>")
    secondary = [item for item in strokes if item["paint_tier"] != "primary"]
    primary = [item for item in strokes if item["paint_tier"] == "primary"]
    lines.append(f'<g id="strokes-secondary" display="{_display(show_strokes)}">')
    if params.get("line_color_mode") != "none":
        lines.extend(_stroke_path(item, params["line_color_mode"]) for item in secondary)
    lines.append("</g>")
    lines.append(f'<g id="strokes-primary" display="{_display(show_strokes)}">')
    if params.get("line_color_mode") != "none":
        lines.extend(_stroke_path(item, params["line_color_mode"]) for item in primary)
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines)


def rasterize(
    shapes: list[dict],
    strokes: list[dict],
    size: tuple[int, int],
    ground: str,
    *,
    draw_shapes: bool = True,
    draw_strokes: bool = True,
    paper: bool = False,
) -> tuple[np.ndarray, float]:
    width, height = size
    canvas = np.empty((height, width, 3), np.uint8)
    canvas[:] = (246, 243, 236) if paper else _hex_rgb(ground)
    cover = np.zeros((height, width), np.uint8)
    if draw_shapes:
        for shape in shapes:
            color = _hex_rgb(shape["fill"])
            for poly in _polygons(shape["commands"], 1.35):
                if len(poly) < 3:
                    continue
                cnt = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(canvas, [cnt], color.tolist())
                cv2.fillPoly(cover, [cnt], 1)
    if draw_strokes:
        for stroke in strokes:
            color = _hex_rgb(stroke.get("hex") or stroke_hex(stroke["darker_lab"], stroke.get("mode", "colored")))
            width_px = max(1, int(round(stroke["paint_width"])))
            for poly in _polygons(stroke["commands"], 2.0):
                if len(poly) < 2:
                    continue
                cnt = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [cnt], bool(stroke["closed"]), color.tolist(), width_px, cv2.LINE_AA)
    coverage = float(cover.mean()) if cover.size else 1.0
    return canvas, coverage


def _assign_groups(comps: list[dict]) -> None:
    by_id = {comp["id"]: comp for comp in comps}
    total = max(sum(item["area"] for item in comps), 1.0)
    for comp in comps:
        comp["group"] = "detail"
    for comp in sorted(comps, key=lambda item: item["depth"]):
        if not comp["is_fg"]:
            comp["group"] = "detail" if _ancestor_fg(comp, by_id) else "bg"
            continue
        parent = by_id.get(comp["parent"])
        ratio = comp["area"] / total
        if ratio >= 0.016 and comp["chroma"] < 0.09 and comp["depth"] <= 3 and (parent is None or parent["group"] != "detail"):
            comp["group"] = "base"
    if not any(comp["group"] == "base" for comp in comps):
        fg = sorted((comp for comp in comps if comp["is_fg"]), key=lambda item: -item["area"])
        for comp in fg[:3]:
            if comp["area"] >= 0.01 * total:
                comp["group"] = "base"


def _ancestor_fg(comp: dict, by_id: dict) -> bool:
    seen = set()
    parent_id = comp["parent"]
    while parent_id in by_id and parent_id not in seen:
        seen.add(parent_id)
        parent = by_id[parent_id]
        if parent["is_fg"]:
            return True
        parent_id = parent["parent"]
    return False


def _band(comp: dict) -> int:
    return {"bg": 0, "base": 1, "detail": 2}.get(comp["group"], 2)


def _ground_color(comps: list[dict]) -> str:
    if not comps:
        return "#f4f1ea"
    bg = [comp for comp in comps if comp["group"] == "bg"] or comps
    touch = [comp for comp in bg if comp["touches"]] or bg
    winner = max(touch, key=lambda comp: comp["area"])
    return winner["fill"]


def _parents(labels: np.ndarray, area: np.ndarray):
    count = int(labels.max()) + 1
    height, width = labels.shape
    parent = np.full(count, -1, np.int32)
    minx = np.full(count, width, np.int32)
    maxx = np.full(count, -1, np.int32)
    miny = np.full(count, height, np.int32)
    maxy = np.full(count, -1, np.int32)
    ys, xs = np.indices((height, width))
    flat = labels.ravel()
    np.minimum.at(minx, flat, xs.ravel())
    np.maximum.at(maxx, flat, xs.ravel())
    np.minimum.at(miny, flat, ys.ravel())
    np.maximum.at(maxy, flat, ys.ravel())
    kernel = np.ones((3, 3), np.uint8)
    for index in range(count):
        if index >= len(area) or area[index] < 24 or maxx[index] < 0:
            continue
        y0 = max(int(miny[index]) - 2, 0)
        y1 = min(int(maxy[index]) + 3, height)
        x0 = max(int(minx[index]) - 2, 0)
        x1 = min(int(maxx[index]) + 3, width)
        crop = labels[y0:y1, x0:x1]
        mask = crop == index
        dilated = cv2.dilate(mask.astype(np.uint8), kernel)
        ring = (dilated > 0) & ~mask
        border_hits = 0
        if y0 == 0:
            border_hits += int(mask[0].sum())
        if y1 == height:
            border_hits += int(mask[-1].sum())
        if x0 == 0:
            border_hits += int(mask[:, 0].sum())
        if x1 == width:
            border_hits += int(mask[:, -1].sum())
        if not np.any(ring):
            continue
        counts = np.bincount(crop[ring], minlength=count)
        counts[index] = 0
        total = int(counts.sum()) + border_hits
        if total <= 0:
            continue
        best = int(np.argmax(counts))
        if counts[best] / total >= 0.72:
            parent[index] = best
    for index in range(count):
        other = int(parent[index])
        if other < 0 or other >= count:
            continue
        if int(parent[other]) == index:
            if area[index] <= area[other]:
                parent[other] = -1
            else:
                parent[index] = -1
    return parent, minx, miny, maxx, maxy


def _depths(parent: np.ndarray) -> list[int]:
    memo: dict[int, int] = {}

    def walk(index: int, stack: set[int]) -> int:
        if index in memo:
            return memo[index]
        other = int(parent[index]) if index < len(parent) else -1
        if other < 0 or other in stack:
            if other in stack:
                parent[index] = -1
            memo[index] = 0
            return 0
        stack.add(index)
        memo[index] = walk(other, stack) + 1
        stack.remove(index)
        return memo[index]

    return [walk(index, set()) for index in range(len(parent))]


def _collect(index: int, children: dict[int, list[int]]) -> list[int]:
    found = []
    for child in children.get(index, []):
        found.append(child)
        found.extend(_collect(child, children))
    return found


def _tucked_contours(labels, z_map, ids, z: int, radius: int, boxes):
    minx, miny, maxx, maxy = boxes
    height, width = labels.shape
    ys = [int(miny[item]) for item in ids if item < len(miny)]
    xs = [int(minx[item]) for item in ids if item < len(minx)]
    ye = [int(maxy[item]) for item in ids if item < len(maxy)]
    xe = [int(maxx[item]) for item in ids if item < len(maxx)]
    if not ys:
        return []
    pad = radius + 3
    y0 = max(min(ys) - pad, 0)
    y1 = min(max(ye) + pad + 1, height)
    x0 = max(min(xs) - pad, 0)
    x1 = min(max(xe) + pad + 1, width)
    table = np.zeros(int(labels.max()) + 1, np.uint8)
    for item in ids:
        if 0 <= item < len(table):
            table[item] = 1
    crop_labels = labels[y0:y1, x0:x1]
    solid = table[crop_labels] > 0
    if not np.any(solid):
        return []
    solid = _fill_holes(solid)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(solid.astype(np.uint8), kernel) > 0
    higher = z_map[y0:y1, x0:x1] > z
    # 只往更高层的肚子底下垫边；外轮廓对着更低层时一像素都不外扩。
    tucked = solid | (dilated & higher)
    closed = cv2.morphologyEx(tucked.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    allow = tucked | (z_map[y0:y1, x0:x1] >= z)
    closed = closed & allow
    if y0 == 0:
        closed[0, :] = closed[0, :] | closed[min(1, closed.shape[0] - 1), :]
    if y1 == height:
        closed[-1, :] = closed[-1, :] | closed[max(0, closed.shape[0] - 2), :]
    if x0 == 0:
        closed[:, 0] = closed[:, 0] | closed[:, min(1, closed.shape[1] - 1)]
    if x1 == width:
        closed[:, -1] = closed[:, -1] | closed[:, max(0, closed.shape[1] - 2)]
    contours, _hier = cv2.findContours(closed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found = []
    for contour in contours:
        if cv2.contourArea(contour) < 12.0:
            continue
        pts = contour.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += x0
        pts[:, 1] += y0
        found.append(pts)
    return found


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = mask.astype(np.uint8)
    flood = np.where(padded == 0, np.uint8(255), np.uint8(0))
    cv2.floodFill(flood, np.zeros((height + 4, width + 4), np.uint8), (0, 0), 0)
    holes = flood[1:-1, 1:-1] == 255
    return mask | holes


def _vectorize_contour(pts: np.ndarray, fit_tol: float, scale: tuple[float, float]):
    from scipy.ndimage import gaussian_filter1d

    pts = pts.copy()
    pts[:, 0] *= scale[0]
    pts[:, 1] *= scale[1]
    pts = _dedup_closed(pts, 0.65)
    if len(pts) < 4:
        return _polygon(pts), []
    # 先消除 SLIC 超像素边缘的波浪锯齿，同时锁定真正的折角尖点
    raw_approx = cv2.approxPolyDP(np.vstack((pts, pts[:1])).astype(np.float32), max(float(fit_tol) * 0.9, 1.5), True).reshape(-1, 2)
    cusp_seed = raw_approx[_polygon_cusps(raw_approx, 55.0)] if len(raw_approx) >= 3 else np.zeros((0, 2), np.float32)
    if len(pts) >= 8:
        sm = gaussian_filter1d(pts, sigma=1.65, axis=0, mode="wrap").astype(np.float32)
        if len(cusp_seed):
            for c_pt in cusp_seed:
                d = np.hypot(pts[:, 0] - c_pt[0], pts[:, 1] - c_pt[1])
                idx = int(np.argmin(d))
                if float(d[idx]) <= 4.0:
                    sm[idx] = pts[idx]
        pts = sm
    pts = _resample_closed(pts, 1.5, 220)
    commands, anchors = _fit_fast(pts, float(fit_tol))
    if not commands:
        commands, anchors = _polygon(pts), []
    return _plain_commands(commands), _plain_anchors(anchors)


def _fit_fast(pts: np.ndarray, tol: float):
    """尖角锁定 G0，长直边强制 L，有机轮廓生成丝滑 G1 三次贝塞尔 C（消灭折线多边形棱角感）。"""
    ring = np.vstack((pts, pts[:1])).astype(np.float32)
    approx = cv2.approxPolyDP(ring, max(tol * 1.35, 2.0), True).reshape(-1, 2)
    if len(approx) < 3:
        return [], []
    cusp_flags = _polygon_cusps(approx, 54.0)
    cusp_pts = approx[cusp_flags] if len(approx) else np.zeros((0, 2), np.float32)
    indices = _nearest_forward(pts, approx)
    count = len(indices)

    # 预计算控制多边形各顶点的平滑切向（Catmull-Rom 方向），供短有机弧段兜底使用，杜绝退化成生硬折线
    vert_tangents = []
    for k in range(count):
        prev_v = approx[(k - 1) % count]
        next_v = approx[(k + 1) % count]
        vec = next_v - prev_v
        nrm = float(np.hypot(vec[0], vec[1]))
        vert_tangents.append((vec / nrm).astype(np.float32) if nrm > 1e-5 else np.array([1.0, 0.0], np.float32))

    segs = []
    for step in range(count):
        nxt_step = (step + 1) % count
        i0 = indices[step]
        i1 = indices[nxt_step]
        chunk = _slice_closed(pts, i0, i1)
        if len(chunk) < 2:
            continue
        p0 = chunk[0].astype(np.float32)
        p1 = chunk[-1].astype(np.float32)
        chord_vec = p1 - p0
        chord_len = float(np.hypot(chord_vec[0], chord_vec[1]))
        locked_start = bool(cusp_flags[step])
        locked = bool(cusp_flags[nxt_step])
        dev = _line_dev(chunk)

        # 只有真正足够长且笔直的边（或两端都是硬尖角的直边）才输出直线 L；有机短弧一律走平滑三次贝塞尔 C！
        is_true_line = (chord_len >= 16.0 and dev <= max(0.75, tol * 0.26)) or (
            locked_start and locked and dev <= max(1.0, tol * 0.45)
        )
        if is_true_line or chord_len < 2.2:
            segs.append(("L", p0, p1, None, None, locked))
            continue

        fitted = _fit_cubic(chunk, 1) if len(chunk) >= 5 else None
        if fitted is not None and fitted["dev"] <= tol * 1.45:
            segs.append(("C", p0, p1, fitted["h1"], fitted["h2"], locked))
        else:
            # 平滑 Catmull-Rom 样条兜底，彻底消除多边形折角感
            t_start = (chord_vec / max(chord_len, 1e-5)) if locked_start else vert_tangents[step]
            t_end = (chord_vec / max(chord_len, 1e-5)) if locked else vert_tangents[nxt_step]
            arm = np.float32(chord_len * 0.32)
            h1 = p0 + t_start * arm
            h2 = p1 - t_end * arm
            segs.append(("C", p0, p1, h1.astype(np.float32), h2.astype(np.float32), locked))
    if len(segs) < 3:
        return [], []
    _align_fast(segs)
    commands = [{"op": "M", "p": [float(segs[0][1][0]), float(segs[0][1][1])]}]
    anchors = []
    for index, seg in enumerate(segs):
        kind, p0, p1, h1, h2, locked = seg
        nxt = segs[(index + 1) % len(segs)]
        if kind == "C" and h1 is not None and h2 is not None:
            end = segs[0][1] if index == len(segs) - 1 else p1
            commands.append(
                {
                    "op": "C",
                    "p": [float(end[0]), float(end[1])],
                    "h1": [float(h1[0]), float(h1[1])],
                    "h2": [float(h2[0]), float(h2[1])],
                }
            )
        else:
            end = segs[0][1] if index == len(segs) - 1 else p1
            commands.append({"op": "L", "p": [float(end[0]), float(end[1])]})
        if index == 0:
            anchors.append(_anchor_point(p0, "cusp" if _near_cusp(p0, cusp_pts) else "smooth", None, h1 if kind == "C" else None))
        if index < len(segs) - 1:
            anchors.append(
                _anchor_point(
                    p1,
                    "cusp" if locked else "smooth",
                    h2 if kind == "C" else None,
                    nxt[3] if nxt[0] == "C" else None,
                )
            )
    commands.append({"op": "Z"})
    return commands, anchors


def _polygon_cusps(approx: np.ndarray, angle: float = 52.0) -> np.ndarray:
    count = len(approx)
    flags = np.zeros(count, np.bool_)
    if count < 3:
        return flags
    for index in range(count):
        prev = approx[(index - 1) % count] - approx[index]
        nxt = approx[(index + 1) % count] - approx[index]
        denom = float(np.hypot(prev[0], prev[1]) * np.hypot(nxt[0], nxt[1]))
        if denom < 1e-6:
            continue
        cos = float(np.clip(np.dot(prev, nxt) / denom, -1.0, 1.0))
        if math.degrees(math.acos(cos)) < 180.0 - angle:
            flags[index] = True
    return flags


def _nearest_forward(pts: np.ndarray, approx: np.ndarray) -> list[int]:
    indices = []
    start = 0
    count = len(pts)
    for vertex in approx:
        best = start
        best_d = 1.0e18
        for step in range(count):
            index = (start + step) % count
            dist = float(np.hypot(*(pts[index] - vertex)))
            if dist < best_d:
                best_d = dist
                best = index
            elif step > 6 and dist > best_d + 8.0:
                break
        indices.append(best)
        start = (best + 1) % count
    return indices


def _slice_closed(pts: np.ndarray, start: int, end: int) -> np.ndarray:
    if end > start:
        return pts[start : end + 1]
    return np.vstack((pts[start:], pts[: end + 1]))


def _line_dev(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    chord = pts[-1] - pts[0]
    length = float(np.hypot(chord[0], chord[1]))
    rel = pts - pts[0]
    if length < 1e-6:
        return float(np.max(np.hypot(rel[:, 0], rel[:, 1])))
    return float(np.max(np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / length))


def _near_cusp(point: np.ndarray, cusps: np.ndarray) -> bool:
    if len(cusps) == 0:
        return False
    dist = np.hypot(cusps[:, 0] - point[0], cusps[:, 1] - point[1])
    return bool(np.min(dist) <= 6.0)


def _align_fast(segs: list) -> None:
    for index, seg in enumerate(segs):
        nxt = segs[(index + 1) % len(segs)]
        if seg[5] or seg[0] != "C" or nxt[0] != "C" or seg[4] is None or nxt[3] is None:
            continue
        anchor = np.asarray(seg[2], np.float32)
        incoming = anchor - np.asarray(seg[4], np.float32)
        outgoing = np.asarray(nxt[3], np.float32) - anchor
        ni = float(np.hypot(incoming[0], incoming[1]))
        no = float(np.hypot(outgoing[0], outgoing[1]))
        if ni < 1e-4 or no < 1e-4:
            continue
        direction = incoming / ni + outgoing / no
        norm = float(np.hypot(direction[0], direction[1]))
        if norm < 0.35:
            continue
        direction /= norm
        segs[index] = (seg[0], seg[1], seg[2], seg[3], anchor - direction * ni, seg[5])
        segs[(index + 1) % len(segs)] = (nxt[0], nxt[1], nxt[2], anchor + direction * no, nxt[4], nxt[5])


def _anchor_point(point, kind: str, h_in, h_out) -> dict:
    item = {"p": [float(point[0]), float(point[1])], "kind": kind}
    if h_in is not None:
        item["h_in"] = [float(h_in[0]), float(h_in[1])]
    if h_out is not None:
        item["h_out"] = [float(h_out[0]), float(h_out[1])]
    return item


def _dedup_closed(pts: np.ndarray, tol: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    kept = [0]
    for index in range(1, len(pts)):
        if float(np.hypot(*(pts[index] - pts[kept[-1]]))) >= tol:
            kept.append(index)
    out = pts[np.array(kept, dtype=np.int32)]
    if len(out) >= 2 and float(np.hypot(*(out[0] - out[-1]))) < tol:
        out = out[:-1]
    return out


def _resample_closed(pts: np.ndarray, step: float, limit: int) -> np.ndarray:
    if len(pts) < 4:
        return pts
    ring = np.vstack((pts, pts[:1]))
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    total = float(seg.sum())
    if total < step * 4:
        return pts
    count = int(np.clip(math.ceil(total / step), 12, limit))
    target = np.linspace(0.0, total, count, endpoint=False)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    samples = np.empty((count, 2), np.float32)
    cursor = 0
    for index, dist in enumerate(target):
        while cursor < len(seg) - 1 and cumulative[cursor + 1] < dist:
            cursor += 1
        span = max(float(seg[cursor]), 1e-6)
        t = (dist - float(cumulative[cursor])) / span
        samples[index] = ring[cursor] * (1.0 - t) + ring[cursor + 1] * t
    return samples


def _polygon(pts: np.ndarray):
    if len(pts) < 3:
        return []
    approx = cv2.approxPolyDP(pts.astype(np.float32), 1.8, True).reshape(-1, 2)
    if len(approx) < 3:
        approx = pts
    commands = [{"op": "M", "p": [float(approx[0, 0]), float(approx[0, 1])]}]
    for point in approx[1:]:
        commands.append({"op": "L", "p": [float(point[0]), float(point[1])]})
    commands.append({"op": "Z"})
    return commands


def _plain_commands(commands: list[dict]) -> list[dict]:
    plain = []
    for cmd in commands:
        item = {"op": cmd["op"]}
        for key in ("p", "h1", "h2"):
            if cmd.get(key) is None:
                continue
            item[key] = [round(float(cmd[key][0]), 2), round(float(cmd[key][1]), 2)]
        plain.append(item)
    return plain


def _plain_anchors(anchors: list[dict]) -> list[dict]:
    plain = []
    for anchor in anchors:
        item = {"p": [round(float(anchor["p"][0]), 2), round(float(anchor["p"][1]), 2)], "kind": anchor.get("kind", "smooth")}
        for key in ("h_in", "h_out"):
            if anchor.get(key) is not None:
                item[key] = [round(float(anchor[key][0]), 2), round(float(anchor[key][1]), 2)]
        plain.append(item)
    return plain


def _fill_path(shape: dict) -> str:
    return f'<path id="sh{shape["id"]}" d="{_d(shape["commands"])}" fill="{shape["fill"]}"/>'


def _stroke_path(stroke: dict, mode: str) -> str:
    color = stroke_hex(stroke["darker_lab"], mode)
    opacity = "" if stroke["opacity"] >= 0.995 else f' stroke-opacity="{stroke["opacity"]:.2f}"'
    return (
        f'<path id="ln{stroke["index"]}" d="{_d(stroke["commands"])}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke["paint_width"]:.2f}" stroke-linecap="round" stroke-linejoin="round"{opacity}/>'
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


def _pair(point) -> str:
    return f"{float(point[0]):.2f} {float(point[1]):.2f}"


def _display(visible: bool) -> str:
    return "inline" if visible else "none"


def _polygons(commands: list[dict], step: float) -> list[np.ndarray]:
    if not commands:
        return []
    pts = _flatten(commands, step)
    if len(pts) < 2:
        return []
    splits = [0]
    cursor = 0
    # _flatten 已经连成一条序列；闭合路径直接整段填充。
    _ = splits, cursor
    return [pts]


def _hex_rgb(value: str) -> np.ndarray:
    text = value.lstrip("#")
    if len(text) != 6:
        return np.array([244, 241, 234], np.uint8)
    return np.array([int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)], np.uint8)
