"""拐角硬切、直线优先，以及自底向上删锚点的三次贝塞尔拟合。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from vectorize.geom import (
    angle_between,
    arc_length,
    chord_params,
    dedup_chain,
    dot,
    eval_cubic,
    eval_cubic_derivs,
    max_deviation,
    principal_axis,
    resample_polyline,
    unit,
)
from vectorize.strokes import Stroke


@dataclass
class Seg:
    i0: int
    i1: int
    kind: str
    p0: np.ndarray
    p1: np.ndarray
    h1: np.ndarray | None
    h2: np.ndarray | None
    lock_end: bool
    dev: float


def fit_stroke(stroke: Stroke, params: dict) -> dict | None:
    pts = np.asarray(stroke.pts, dtype=np.float32).copy()
    radii = np.asarray(stroke.radii, dtype=np.float32).copy()
    cusp = np.asarray(stroke.cusp, dtype=bool).copy()
    if len(pts) < 2:
        return None
    closed = bool(stroke.closed)
    if closed and len(pts) >= 3 and float(np.hypot(*(pts[0] - pts[-1]))) < 1.25:
        pts, radii, cusp = pts[:-1], radii[:-1], cusp[:-1]
    pts, radii, cusp = dedup_chain(pts, radii, cusp)
    if len(pts) < 2:
        return None
    profile = _width_profile(pts, radii)
    median_radius = float(np.median(radii)) if len(radii) else 1.0
    params = dict(params)
    params["r_med"] = median_radius
    if not closed and len(pts) >= 2:
        line_tol = max(0.85, 0.42 * median_radius) * float(params["line_snap_scale"]) * 1.15
        if max_deviation(pts, pts[0], pts[-1]) <= line_tol:
            commands = [{"op": "M", "p": _xy(pts[0])}, {"op": "L", "p": _xy(pts[-1])}]
            anchors = [_anchor(pts[0], "endpoint", None, None), _anchor(pts[-1], "endpoint", None, None)]
            return {
                "tier": stroke.tier,
                "score": float(stroke.score),
                "closed": False,
                "commands": commands,
                "anchors": anchors,
                "bridges": [dict(item) for item in stroke.bridges],
                "width_profile": profile,
                "median_radius": median_radius,
            }
    if not closed and len(pts) <= 42:
        fast = _fit_short(pts, cusp, params, stroke, profile, median_radius, closed)
        if fast is not None:
            return fast
    cusp = _detect_cusps(pts, radii, cusp, closed, float(params["cusp_angle_deg"]))
    smoothed = _smooth(pts, cusp, closed)
    if closed:
        commands, anchors = _fit_closed(smoothed, cusp, params)
    else:
        commands, anchors = _fit_open(smoothed, cusp, params)
    if not commands:
        return None
    return {
        "tier": stroke.tier,
        "score": float(stroke.score),
        "closed": closed and any(item["op"] == "Z" for item in commands),
        "commands": commands,
        "anchors": anchors,
        "bridges": [dict(item) for item in stroke.bridges],
        "width_profile": profile,
        "median_radius": median_radius,
    }


def _fit_short(pts, cusp, params, stroke, profile, median_radius, closed):
    """短笔画不再走整段拐点搜索。直线优先，放不进容差再拆一次。"""
    commands, anchors = _short_commands(pts, params, bool(np.any(cusp)), 0)
    if not commands:
        return None
    return {
        "tier": stroke.tier,
        "score": float(stroke.score),
        "closed": False,
        "commands": commands,
        "anchors": anchors,
        "bridges": [dict(item) for item in stroke.bridges],
        "width_profile": profile,
        "median_radius": median_radius,
    }


def _short_commands(pts, params, locked: bool, depth: int):
    p0 = pts[0]
    p1 = pts[-1]
    dev = max_deviation(pts, p0, p1)
    line_tol = max(0.85, 0.42 * float(params.get("r_med", 1.2))) * float(params["line_snap_scale"])
    if len(pts) <= 3 or dev <= line_tol * 1.15:
        return [{"op": "M", "p": _xy(p0)}, {"op": "L", "p": _xy(p1)}], [
            _anchor(p0, "endpoint", None, None),
            _anchor(p1, "endpoint", None, None),
        ]
    if len(pts) >= 4:
        cubic = _fit_cubic(pts, 1)
        if cubic is not None and cubic["dev"] <= float(params["fit_tol"]) * 1.35:
            seg = _seg_from_fit(0, len(pts) - 1, {**cubic, "kind": "C", "p0": p0, "p1": p1}, False)
            return _commands([seg], False)
    if depth >= 2 or len(pts) < 8:
        anchors = [_anchor(p0, "endpoint", None, None), _anchor(p1, "endpoint", None, None)]
        return [{"op": "M", "p": _xy(p0)}, {"op": "L", "p": _xy(p1)}], anchors
    rel = pts - p0
    chord = p1 - p0
    length = float(np.hypot(chord[0], chord[1]))
    dist = np.hypot(rel[:, 0], rel[:, 1]) if length < 1e-6 else np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / max(length, 1e-6)
    pivot = int(np.clip(int(np.argmax(dist)), 2, len(pts) - 3))
    left_c, left_a = _short_commands(pts[: pivot + 1], params, False, depth + 1)
    right_c, right_a = _short_commands(pts[pivot:], params, locked, depth + 1)
    commands = [left_c[0]]
    commands.extend(item for item in left_c[1:] if item["op"] != "M")
    commands.extend(item for item in right_c[1:] if item["op"] != "M")
    anchors = [left_a[0]]
    if len(left_a) > 2:
        anchors.extend(left_a[1:-1])
    joint = dict(left_a[-1])
    joint["kind"] = "cusp" if float(dist[pivot]) > 0.55 * float(params["cusp_angle_deg"]) else "smooth"
    if right_a[-1].get("h_in") and joint["kind"] == "smooth":
        pass
    if len(right_a) > 1 and "h_out" in right_a[0]:
        joint["h_out"] = right_a[0]["h_out"]
    if "h_in" in left_a[-1]:
        joint["h_in"] = left_a[-1]["h_in"]
    anchors.append(joint)
    anchors.extend(right_a[1:])
    anchors[0]["kind"] = "endpoint"
    anchors[-1]["kind"] = "endpoint"
    return commands, anchors


def _width_profile(pts: np.ndarray, radii: np.ndarray) -> list[float]:
    if len(radii) == 0:
        return [1.2]
    step = max(1, len(radii) // 24)
    return [float(value) * 2.0 for value in radii[::step]]


def _detect_cusps(pts, radii, cusp, closed: bool, angle_deg: float) -> np.ndarray:
    cusp = np.asarray(cusp, dtype=bool).copy()
    count = len(pts)
    r_med = float(np.median(radii)) if len(radii) else 1.0
    window = int(np.clip(round(2.0 * r_med + 2.0), 3, 8))
    if count < window * 2 + 2:
        if not closed:
            cusp[0] = False
            cusp[-1] = False
        return cusp
    turn, cross = _turn_and_cross(pts, window, closed)
    picked = []
    for index in range(count):
        if not closed and (index < window or index >= count - window):
            continue
        if turn[index] <= angle_deg:
            continue
        prev_i = (index - 1) % count
        next_i = (index + 1) % count
        if turn[index] >= turn[prev_i] and turn[index] >= turn[next_i]:
            picked.append(index)
    for index in _suppress(picked, turn, window, count, closed):
        cusp[index] = True
    if not closed and count:
        cusp[0] = False
        cusp[-1] = False
    return cusp


def _suppress(indices: list[int], turn: np.ndarray, gap: int, count: int, closed: bool) -> list[int]:
    kept: list[int] = []
    for index in sorted(indices, key=lambda item: -float(turn[item])):
        if all(_circular_gap(index, other, count, closed) >= gap for other in kept):
            kept.append(index)
    return kept


def _circular_gap(left: int, right: int, count: int, closed: bool) -> int:
    delta = abs(left - right)
    if closed:
        return min(delta, count - delta)
    return delta


def _turn_and_cross(pts: np.ndarray, window: int, closed: bool):
    count = len(pts)
    if closed:
        pad = np.vstack((pts[-window:], pts, pts[:window]))
        start = pad[window : window + count]
        before = pad[:count]
        after = pad[2 * window : 2 * window + count]
        vec_a = start - before
        vec_b = after - start
    else:
        vec_a = pts[window : count - window] - pts[: count - 2 * window]
        vec_b = pts[2 * window :] - pts[window : count - window]
    turn = np.zeros(count, np.float32)
    cross = np.zeros(count, np.float32)
    angles = _angles(vec_a, vec_b)
    signed = vec_a[:, 0] * vec_b[:, 1] - vec_a[:, 1] * vec_b[:, 0]
    if closed:
        turn[:] = angles
        cross[:] = signed
    else:
        turn[window : count - window] = angles
        cross[window : count - window] = signed
    return turn, cross


def _angles(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    na = np.maximum(np.linalg.norm(vec_a, axis=1), 1e-6)
    nb = np.maximum(np.linalg.norm(vec_b, axis=1), 1e-6)
    cos = np.clip(np.sum(vec_a * vec_b, axis=1) / (na * nb), -1.0, 1.0)
    return np.degrees(np.arccos(cos)).astype(np.float32)


def _smooth(pts: np.ndarray, cusp: np.ndarray, closed: bool) -> np.ndarray:
    if len(pts) < 5:
        return pts.copy()
    smoothed = gaussian_filter1d(pts, 0.9, axis=0, mode="wrap" if closed else "nearest")
    smoothed = smoothed.astype(np.float32)
    if not closed:
        smoothed[0] = pts[0]
        smoothed[-1] = pts[-1]
    smoothed[cusp] = pts[cusp]
    return smoothed


def _fit_open(pts: np.ndarray, cusp: np.ndarray, params: dict):
    cuts = [0]
    cuts.extend(int(index) for index in np.flatnonzero(cusp))
    cuts.append(len(pts) - 1)
    cuts = sorted(set(cuts))
    segs: list[Seg] = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        segs.extend(_segments_along(pts, left, right, params, bool(cusp[right]) and right != len(pts) - 1))
    segs = _merge(segs, pts, params)
    _g1(segs, closed=False)
    return _commands(segs, closed=False)


def _fit_closed(pts: np.ndarray, cusp: np.ndarray, params: dict):
    if len(pts) < 4:
        return [], []
    count = len(pts)
    work = np.vstack((pts, pts[:1])).astype(np.float32)
    flags = np.concatenate((cusp, cusp[:1]))
    cuts = [0]
    cuts.extend(int(index) for index in np.flatnonzero(cusp))
    if len(cuts) == 1:
        cuts.append(max(1, count // 2))
    cuts.append(count)
    cuts = sorted({index for index in cuts if 0 <= index <= count})
    segs: list[Seg] = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        lock = bool(flags[min(right, len(flags) - 1)])
        segs.extend(_segments_along(work, left, right, params, lock))
    segs = _merge(segs, work, params)
    if len(segs) >= 2 and not segs[-1].lock_end:
        segs = _merge_wrap(segs, work, params)
    _g1(segs, closed=True)
    return _commands(segs, closed=True)


def _segments_along(pts: np.ndarray, start: int, end: int, params: dict, lock_end: bool) -> list[Seg]:
    if end <= start:
        return []
    sub_pts = pts[start : end + 1]
    no_cusp = np.zeros(len(sub_pts), dtype=bool)
    knots = _feature_knots(sub_pts, no_cusp, closed=False)
    segs: list[Seg] = []
    for left, right in zip(knots[:-1], knots[1:]):
        if right <= left:
            continue
        segs.extend(_subdivide(pts, start + left, start + right, params, lock_end and start + right == end, 0))
    return segs


def _rdp_indices(pts: np.ndarray, tol: float) -> list[int]:
    count = len(pts)
    if count < 3:
        return [0, max(0, count - 1)]
    keep = np.zeros(count, np.uint8)
    keep[0] = 1
    keep[-1] = 1
    stack = [(0, count - 1)]
    while stack:
        left, right = stack.pop()
        if right <= left + 1:
            continue
        chunk = pts[left : right + 1]
        start = chunk[0]
        chord = chunk[-1] - start
        length = float(np.hypot(chord[0], chord[1]))
        rel = chunk - start
        if length < 1e-6:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            dist = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / length
        pivot = int(np.argmax(dist))
        if float(dist[pivot]) > tol and 0 < pivot < len(chunk) - 1:
            keep[left + pivot] = 1
            stack.append((left, left + pivot))
            stack.append((left + pivot, right))
    return np.flatnonzero(keep).astype(int).tolist()


def _feature_knots(pts: np.ndarray, cusp: np.ndarray, closed: bool) -> list[int]:
    count = len(pts)
    r_span = max(count, 1)
    window = int(np.clip(round(6), 3, 8))
    if count >= 8:
        turn, cross = _turn_and_cross(pts, min(window, max(3, count // 6)), closed)
        window = min(window, max(3, count // 6))
    else:
        turn = np.zeros(count, np.float32)
        cross = np.zeros(count, np.float32)
        window = 3
    knots = {0, count - 1} if not closed else {0}
    knots.update(int(index) for index in np.flatnonzero(cusp))
    limit = count if closed else count - window
    start = 0 if closed else window
    for index in range(start, limit):
        prev_i = (index - 1) % count
        next_i = (index + 1) % count
        if cross[prev_i] * cross[next_i] < 0 and turn[index] < 28.0:
            if abs(float(cross[prev_i])) > 1.4 and abs(float(cross[next_i])) > 1.4:
                knots.add(index)
        if 12.0 < turn[index] <= 80.0 and turn[index] >= turn[prev_i] and turn[index] >= turn[next_i]:
            knots.add(index)
    ordered = sorted(knots)
    if not ordered:
        return [0, count - 1]
    filtered = [ordered[0]]
    spacing = max(3, window)
    for index in ordered[1:]:
        if index - filtered[-1] < spacing and not cusp[index % count]:
            continue
        if index - filtered[-1] < spacing and cusp[index % count] and cusp[filtered[-1] % count]:
            if turn[index % count] > turn[filtered[-1] % count]:
                filtered[-1] = index
            continue
        filtered.append(index)
    if not closed:
        if filtered[0] != 0:
            filtered.insert(0, 0)
        if filtered[-1] != count - 1:
            filtered.append(count - 1)
    _ = r_span
    return filtered


def _initial_segments(pts, knots, cusp, params, closed: bool) -> list[Seg]:
    segs: list[Seg] = []
    usable = [int(k) for k in knots if 0 <= int(k) < len(pts)]
    if closed and usable[-1] != len(pts) - 1:
        usable.append(len(pts) - 1)
    for left, right in zip(usable[:-1], usable[1:]):
        if right <= left:
            continue
        lock = bool(cusp[right]) if right < len(cusp) else False
        if closed and right == len(pts) - 1:
            lock = bool(cusp[0])
        segs.extend(_subdivide(pts, left, right, params, lock, 0))
    return segs


def _subdivide(pts, left: int, right: int, params: dict, lock: bool, depth: int) -> list[Seg]:
    chunk = pts[left : right + 1]
    if (
        len(chunk) >= 8
        and depth < 3
        and float(np.hypot(*(chunk[0] - chunk[-1]))) < 1.2
        and arc_length(chunk) > 8.0
    ):
        cut = left + max(2, (right - left) // 2)
        if left + 2 < cut < right - 2:
            return _subdivide(pts, left, cut, params, False, depth + 1) + _subdivide(
                pts, cut, right, params, lock, depth + 1
            )
    fitted = _fit_segment(chunk, params)
    short = len(chunk) < 8 or arc_length(chunk) < 6.0
    if fitted["kind"] == "L" or fitted["dev"] <= params["fit_tol"] or short or depth >= 3:
        return [_seg_from_fit(left, right, fitted, lock)]
    errors = fitted["errors"]
    if errors is None or len(errors) < 6:
        return [_seg_from_fit(left, right, fitted, lock)]
    pivot = int(np.argmax(errors))
    pivot = int(np.clip(pivot, 3, len(chunk) - 4))
    cut = left + pivot
    if cut <= left + 2 or cut >= right - 2:
        return [_seg_from_fit(left, right, fitted, lock)]
    return _subdivide(pts, left, cut, params, False, depth + 1) + _subdivide(
        pts, cut, right, params, lock, depth + 1
    )


def _seg_from_fit(left: int, right: int, fitted: dict, lock: bool) -> Seg:
    return Seg(
        left,
        right,
        fitted["kind"],
        np.asarray(fitted["p0"], np.float32),
        np.asarray(fitted["p1"], np.float32),
        None if fitted["h1"] is None else np.asarray(fitted["h1"], np.float32),
        None if fitted["h2"] is None else np.asarray(fitted["h2"], np.float32),
        lock,
        float(fitted["dev"]),
    )


def _merge(segs: list[Seg], pts: np.ndarray, params: dict) -> list[Seg]:
    failed: set[tuple[int, int]] = set()
    guard = 0
    while guard < 2000 and len(segs) >= 2:
        guard += 1
        merged = False
        index = 0
        while index < len(segs) - 1:
            left = segs[index]
            right = segs[index + 1]
            pair_key = (left.i0, right.i1)
            if left.lock_end or right.i1 <= left.i0 or pair_key in failed:
                index += 1
                continue
            chunk = pts[left.i0 : right.i1 + 1]
            if _closedish(chunk):
                failed.add(pair_key)
                index += 1
                continue
            fitted = _fit_segment(chunk, params)
            if fitted["dev"] <= params["fit_tol"] + 1e-6:
                segs[index : index + 2] = [_seg_from_fit(left.i0, right.i1, fitted, right.lock_end)]
                merged = True
            else:
                failed.add(pair_key)
                index += 1
        if not merged:
            break
    return segs


def _merge_wrap(segs: list[Seg], work: np.ndarray, params: dict) -> list[Seg]:
    first, last = segs[0], segs[-1]
    if last.i0 >= len(work) - 1 or first.i1 <= 0:
        return segs
    chunk = np.vstack((work[last.i0 :], work[1 : first.i1 + 1]))
    if len(chunk) < 4 or _closedish(chunk):
        return segs
    fitted = _fit_segment(chunk, params)
    if fitted["dev"] > params["fit_tol"] + 1e-6:
        return segs
    joined = _seg_from_fit(last.i0, first.i1, fitted, first.lock_end)
    joined.p0 = np.asarray(fitted["p0"], np.float32)
    joined.p1 = np.asarray(fitted["p1"], np.float32)
    middle = segs[1:-1]
    return middle + [joined]


def _g1(segs: list[Seg], closed: bool) -> None:
    if len(segs) < 2 and not (closed and len(segs) == 1):
        return
    pairs = [(segs[index], segs[index + 1]) for index in range(len(segs) - 1) if not segs[index].lock_end]
    if closed and segs and not segs[-1].lock_end and len(segs) >= 2:
        pairs.append((segs[-1], segs[0]))
    for prev, nxt in pairs:
        _align(prev, nxt)
    if closed and len(segs) == 1 and segs[0].kind == "C" and not segs[0].lock_end:
        _align_loop(segs[0])


def _align(prev: Seg, nxt: Seg) -> None:
    anchor = prev.p1.astype(np.float32)
    nxt.p0 = anchor.copy()
    if prev.kind == "L" and nxt.kind == "L":
        return
    if prev.kind == "L" and nxt.kind == "C" and nxt.h1 is not None:
        direction = unit(anchor - prev.p0)
        nxt.h1 = anchor + direction * float(np.linalg.norm(nxt.h1 - anchor))
        return
    if prev.kind == "C" and nxt.kind == "L" and prev.h2 is not None:
        direction = unit(nxt.p1 - anchor)
        prev.h2 = anchor - direction * float(np.linalg.norm(anchor - prev.h2))
        return
    if prev.h2 is None or nxt.h1 is None:
        return
    incoming = unit(anchor - prev.h2)
    outgoing = unit(nxt.h1 - anchor)
    direction = unit(incoming + outgoing)
    if float(np.linalg.norm(direction)) < 0.5:
        direction = outgoing
    prev.h2 = anchor - direction * float(np.linalg.norm(anchor - prev.h2))
    nxt.h1 = anchor + direction * float(np.linalg.norm(nxt.h1 - anchor))


def _align_loop(seg: Seg) -> None:
    if seg.h1 is None or seg.h2 is None:
        return
    anchor = seg.p0
    incoming = unit(anchor - seg.h2)
    outgoing = unit(seg.h1 - anchor)
    direction = unit(incoming + outgoing)
    if float(np.linalg.norm(direction)) < 0.5:
        return
    seg.h2 = anchor - direction * float(np.linalg.norm(anchor - seg.h2))
    seg.h1 = anchor + direction * float(np.linalg.norm(seg.h1 - anchor))
    seg.p1 = anchor.copy()


def _commands(segs: list[Seg], closed: bool):
    if not segs:
        return [], []
    if not closed:
        anchors = [_anchor(segs[0].p0, "endpoint", None, segs[0].h1 if segs[0].kind == "C" else None)]
        commands = [{"op": "M", "p": _xy(segs[0].p0)}]
        for index, seg in enumerate(segs):
            commands.append(_command(seg))
            last = index == len(segs) - 1
            kind = "endpoint" if last else ("cusp" if seg.lock_end else "smooth")
            h_out = None
            if not last and segs[index + 1].kind == "C":
                h_out = segs[index + 1].h1
            h_in = seg.h2 if seg.kind == "C" else None
            anchors.append(_anchor(seg.p1, kind, h_in, h_out))
        return commands, anchors
    start = segs[0].p0
    start_kind = "cusp" if segs[-1].lock_end else "smooth"
    h_out = segs[0].h1 if segs[0].kind == "C" else None
    h_in = segs[-1].h2 if segs[-1].kind == "C" else None
    anchors = [_anchor(start, start_kind, h_in, h_out)]
    commands = [{"op": "M", "p": _xy(start)}]
    for index, seg in enumerate(segs):
        end_point = start if index == len(segs) - 1 else seg.p1
        commands.append(_command(seg, end_point))
        if index == len(segs) - 1:
            continue
        kind = "cusp" if seg.lock_end else "smooth"
        nxt = segs[index + 1]
        anchors.append(
            _anchor(
                seg.p1,
                kind,
                seg.h2 if seg.kind == "C" else None,
                nxt.h1 if nxt.kind == "C" else None,
            )
        )
    commands.append({"op": "Z"})
    return commands, anchors


def _anchor(point, kind: str, h_in, h_out) -> dict:
    item = {"p": _xy(point), "kind": kind}
    if h_in is not None:
        item["h_in"] = _xy(h_in)
    if h_out is not None:
        item["h_out"] = _xy(h_out)
    return item


def _command(seg: Seg, end=None) -> dict:
    point = seg.p1 if end is None else end
    if seg.kind == "C" and seg.h1 is not None and seg.h2 is not None:
        return {"op": "C", "p": _xy(point), "h1": _xy(seg.h1), "h2": _xy(seg.h2)}
    return {"op": "L", "p": _xy(point)}


def _xy(point) -> list[float]:
    arr = np.asarray(point, dtype=np.float32).reshape(-1)
    return [float(arr[0]), float(arr[1])]


def _closedish(chunk: np.ndarray) -> bool:
    if len(chunk) < 4:
        return False
    return float(np.hypot(*(chunk[0] - chunk[-1]))) < 1.5 and arc_length(chunk) > 8.0


def _fit_segment(pts: np.ndarray, params: dict) -> dict:
    pts = np.asarray(pts, dtype=np.float32)
    p0 = pts[0].astype(np.float32)
    p1 = pts[-1].astype(np.float32)
    r_med = float(params.get("r_med", 1.2))
    line_tol = max(0.85, 0.42 * r_med) * float(params["line_snap_scale"])
    line_dev = max_deviation(pts, p0, p1)
    length = float(np.hypot(*(p1 - p0)))
    if len(pts) <= 3 or length < 4.5 or line_dev <= line_tol * 1.15:
        sp0, sp1 = _maybe_ortho(p0, p1, length, float(params["ortho_min"]))
        return {"kind": "L", "p0": sp0, "p1": sp1, "h1": None, "h2": None, "dev": line_dev, "errors": None}
    passes = 1 if line_dev <= float(params["fit_tol"]) * 2.2 else 3
    cubic = _fit_cubic(pts, passes)
    if cubic is None:
        return {"kind": "L", "p0": p0, "p1": p1, "h1": None, "h2": None, "dev": line_dev, "errors": None}
    chord = p1 - p0
    arch = max_deviation(eval_cubic(np.linspace(0.0, 1.0, 16, dtype=np.float32), p0, cubic["h1"], cubic["h2"], p1), p0, p1)
    a1 = angle_between(cubic["h1"] - p0, chord)
    a2 = angle_between(p1 - cubic["h2"], chord)
    if a1 < 7.0 and a2 < 7.0 and arch <= 1.15 * line_tol:
        sp0, sp1 = _maybe_ortho(p0, p1, length, float(params["ortho_min"]))
        return {"kind": "L", "p0": sp0, "p1": sp1, "h1": None, "h2": None, "dev": max(line_dev, arch), "errors": None}
    return {
        "kind": "C",
        "p0": p0,
        "p1": p1,
        "h1": cubic["h1"],
        "h2": cubic["h2"],
        "dev": cubic["dev"],
        "errors": cubic["errors"],
    }


def _maybe_ortho(p0, p1, length: float, ortho_min: float):
    if length <= ortho_min:
        return p0, p1
    delta = np.asarray(p1, np.float32) - np.asarray(p0, np.float32)
    ang = abs(float(np.degrees(np.arctan2(delta[1], delta[0])))) % 180.0
    to_h = min(ang, abs(180.0 - ang))
    to_v = abs(ang - 90.0)
    if min(to_h, to_v) >= 2.5:
        return p0, p1
    if to_h <= to_v:
        y = float(p0[1] + p1[1]) * 0.5
        return np.array([p0[0], y], np.float32), np.array([p1[0], y], np.float32)
    x = float(p0[0] + p1[0]) * 0.5
    return np.array([x, p0[1]], np.float32), np.array([x, p1[1]], np.float32)


def _fit_cubic(pts: np.ndarray, passes: int = 3):
    p0 = pts[0].astype(np.float32)
    p3 = pts[-1].astype(np.float32)
    span = float(np.hypot(p3[0] - p0[0], p3[1] - p0[1]))
    if span < 1e-4 or len(pts) < 4:
        return None
    quick = _heuristic_cubic(pts, p0, p3, span)
    if passes <= 1 and quick is not None and quick["dev"] <= 1.15:
        return quick
    if len(pts) > 28:
        pick = np.unique(np.linspace(0, len(pts) - 1, 24).astype(np.int32))
        work = pts[pick]
    else:
        pick = None
        work = pts
    t0, tb = _end_tangents(work)
    ts = chord_params(work)
    h1 = p0 + t0 * (span / 3.0)
    h2 = p3 + tb * (span / 3.0)
    err = None
    for _ in range(max(1, passes)):
        alpha, beta = _solve_handles(work, ts, p0, p3, t0, tb)
        alpha = float(np.clip(alpha, 0.0, span * 1.45))
        beta = float(np.clip(beta, 0.0, span * 1.45))
        h1 = p0 + t0 * np.float32(alpha)
        h2 = p3 + tb * np.float32(beta)
        ts = _reparam(work, p0, h1, h2, p3, ts, steps=1)
        err = np.linalg.norm(eval_cubic(ts, p0, h1, h2, p3) - work, axis=1)
        if float(err.max()) <= 0.45:
            break
    errors = np.zeros(len(pts), np.float32)
    if pick is None:
        errors[:] = err
    else:
        errors[pick] = err
    return {"h1": h1.astype(np.float32), "h2": h2.astype(np.float32), "dev": float(err.max()), "errors": errors}


def _heuristic_cubic(pts, p0, p3, span):
    count = len(pts)
    k = min(6, count - 1)
    t0 = unit(pts[k] - p0)
    tb = unit(pts[count - 1 - k] - p3)
    if float(np.linalg.norm(t0)) < 0.5 or float(np.linalg.norm(tb)) < 0.5:
        return None
    h1 = p0 + t0 * np.float32(span / 3.0)
    h2 = p3 + tb * np.float32(span / 3.0)
    pick = np.linspace(0, count - 1, 8).astype(np.int32) if count > 8 else np.arange(count)
    samples = eval_cubic(np.linspace(0.0, 1.0, len(pick), dtype=np.float32), p0, h1, h2, p3)
    err = np.linalg.norm(samples - pts[pick], axis=1)
    return {"h1": h1, "h2": h2, "dev": float(err.max()), "errors": None}


def _end_tangents(pts: np.ndarray):
    count = len(pts)
    k = int(np.clip(round(max(4, count * 0.18)), 4, 12))
    k = min(k, count)
    chord = unit(pts[-1] - pts[0])
    t0 = unit(pts[min(k, count - 1)] - pts[0])
    tb = unit(pts[max(0, count - 1 - k)] - pts[-1])
    if dot(t0, chord) < 0.0:
        t0 = chord
    if dot(tb, -chord) < 0.0:
        tb = -chord
    if abs(dot(t0, chord)) < 0.2:
        t0 = chord
    if abs(dot(tb, -chord)) < 0.2:
        tb = -chord
    return t0.astype(np.float32), tb.astype(np.float32)


def _solve_handles(pts, ts, p0, p3, t0, tb):
    u = 1.0 - ts
    u2 = 3.0 * (u * u) * ts
    u3 = 3.0 * u * (ts * ts)
    u0 = (u * u) * (1.0 + 2.0 * ts)
    u1 = (ts * ts) * (3.0 - 2.0 * ts)
    rhs = pts - u0[:, None] * p0 - u1[:, None] * p3
    c0 = np.empty(len(pts) * 2, np.float64)
    c1 = np.empty(len(pts) * 2, np.float64)
    target = np.empty(len(pts) * 2, np.float64)
    c0[0::2] = u2 * float(t0[0])
    c1[0::2] = u3 * float(tb[0])
    c0[1::2] = u2 * float(t0[1])
    c1[1::2] = u3 * float(tb[1])
    target[0::2] = rhs[:, 0]
    target[1::2] = rhs[:, 1]
    a00 = float(c0 @ c0)
    a01 = float(c0 @ c1)
    a11 = float(c1 @ c1)
    b0 = float(c0 @ target)
    b1 = float(c1 @ target)
    det = a00 * a11 - a01 * a01
    if abs(det) < 1e-8:
        return 0.0, 0.0
    return (b0 * a11 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det


def _reparam(pts, p0, h1, h2, p3, ts: np.ndarray, steps: int = 3) -> np.ndarray:
    current = np.asarray(ts, dtype=np.float32).copy()
    for _ in range(steps):
        point, d1, d2 = eval_cubic_derivs(current, p0, h1, h2, p3)
        diff = point - pts
        num = np.sum(diff * d1, axis=1)
        den = np.sum(d1 * d1, axis=1) + np.sum(diff * d2, axis=1)
        den = np.where(np.abs(den) < 1e-6, np.float32(1e-6), den)
        current = np.clip(current - num / den, 0.0, 1.0).astype(np.float32)
    return current
