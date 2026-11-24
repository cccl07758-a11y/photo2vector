"""跨交点长笔画、断口焊接、高光双脊塌缩与主次分级。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from vectorize.geom import (
    angle_between,
    arc_length,
    cross2,
    dot,
    point_segment_distance,
    principal_axis,
    resample_polyline,
    sample_nearest,
    sample_bilinear,
    sample_cubic_polyline,
    turn_through,
    unit,
)
from vectorize.skeleton import Edge, SkeletonGraph, _orient_from

_PRIMARY = 0.50
_COS36 = float(np.cos(np.deg2rad(36.0)))


@dataclass
class Stroke:
    pts: np.ndarray
    radii: np.ndarray
    cusp: np.ndarray
    closed: bool = False
    bridges: list = field(default_factory=list)
    score: float = 0.0
    tier: str = "secondary"


def trace_strokes(graph: SkeletonGraph, max_turn_deg: float) -> list[Stroke]:
    pairs = _pair_junctions(graph, max_turn_deg)
    consumed: set[int] = set()
    strokes: list[Stroke] = []

    def walk(edge_id: int, start_node: int) -> Stroke | None:
        chunks_p: list[np.ndarray] = []
        chunks_r: list[np.ndarray] = []
        chunks_c: list[np.ndarray] = []
        first_edge = edge_id
        first_node = start_node
        closed = False
        if start_node in graph.junctions:
            _push_point(chunks_p, chunks_r, chunks_c, graph.junctions[start_node].xy, graph.junctions[start_node].radius, start_node in graph.cusp_nodes)
        guard = 0
        while guard < 8000:
            guard += 1
            if edge_id in consumed or edge_id not in graph.edges:
                break
            edge = graph.edges[edge_id]
            consumed.add(edge_id)
            if edge.u == edge.v:
                seq = edge.pts
                rad = edge.radii
                if edge.u in graph.junctions and len(seq):
                    center = graph.junctions[edge.u].xy
                    seq = np.vstack((center, seq, center)).astype(np.float32)
                    mid_r = np.float32(graph.junctions[edge.u].radius)
                    rad = np.concatenate(([mid_r], rad, [mid_r])).astype(np.float32)
                chunks_p.append(np.asarray(seq, np.float32))
                chunks_r.append(np.asarray(rad, np.float32))
                flag = np.zeros(len(seq), dtype=bool)
                chunks_c.append(flag)
                closed = True
                break
            if start_node == edge.u:
                seq, rad, nxt = edge.pts, edge.radii, edge.v
            elif start_node == edge.v:
                seq, rad, nxt = edge.pts[::-1], edge.radii[::-1], edge.u
            else:
                break
            if chunks_p:
                _bridge_junction(chunks_p, chunks_r, chunks_c, start_node, graph, seq[0])
            chunks_p.append(np.ascontiguousarray(seq, dtype=np.float32))
            chunks_r.append(np.ascontiguousarray(rad, dtype=np.float32))
            chunks_c.append(np.zeros(len(seq), dtype=bool))
            partner = pairs.get((edge_id, nxt))
            if partner is None or partner in consumed or partner not in graph.edges:
                if nxt in graph.junctions:
                    _push_point(
                        chunks_p,
                        chunks_r,
                        chunks_c,
                        graph.junctions[nxt].xy,
                        graph.junctions[nxt].radius,
                        nxt in graph.cusp_nodes,
                    )
                if partner == first_edge and nxt == first_node:
                    closed = True
                break
            edge_id = int(partner)
            start_node = nxt
        if not chunks_p:
            return None
        pts = np.vstack(chunks_p).astype(np.float32)
        radii = np.concatenate(chunks_r).astype(np.float32)
        cusp = np.concatenate(chunks_c).astype(bool)
        if closed and len(pts) >= 4 and float(np.hypot(*(pts[0] - pts[-1]))) < 1.6:
            pts = pts[:-1]
            radii = radii[:-1]
            cusp = cusp[:-1]
        return Stroke(pts, radii, cusp, closed=closed)

    for node in list(graph.adj):
        if graph.degree(node) > 1:
            continue
        for edge_id in list(graph.adj.get(node, ())):
            if edge_id not in consumed:
                stroke = walk(edge_id, node)
                if stroke is not None and len(stroke.pts) >= 2:
                    strokes.append(stroke)
    for edge_id, edge in list(graph.edges.items()):
        if edge_id in consumed:
            continue
        start = edge.u
        if (edge_id, edge.u) in pairs and (edge_id, edge.v) not in pairs:
            start = edge.v
        elif (edge_id, edge.v) in pairs and (edge_id, edge.u) not in pairs:
            start = edge.u
        stroke = walk(edge_id, start)
        if stroke is not None and len(stroke.pts) >= 2:
            strokes.append(stroke)
    return strokes


def weld_gaps(strokes: list[Stroke], oklab: np.ndarray, gap_max: float) -> tuple[list[Stroke], int]:
    if gap_max < 2.0 or len(strokes) < 1:
        return strokes, 0
    ends = _open_ends(strokes, min_len=10.0)
    matches = _match_ends(ends, strokes, oklab, gap_max)
    merged, used = _merge_matches(strokes, matches)
    leftovers = [strokes[index] for index in range(len(strokes)) if index not in used]
    merged.extend(leftovers)
    extra = _snap_tees(merged, gap_max)
    return merged, len(matches) + extra


def collapse_highlights(strokes: list[Stroke], oklab: np.ndarray, enabled: bool) -> tuple[list[Stroke], int]:
    if not enabled or len(strokes) < 2:
        return strokes, 0
    feats = [_feat(stroke) for stroke in strokes]
    drop: set[int] = set()
    collapsed = 0
    cell = 28.0
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, feat in enumerate(feats):
        if feat is None:
            continue
        x0, y0, x1, y1 = feat["bbox"]
        for gy in range(int(y0 // cell), int(y1 // cell) + 1):
            for gx in range(int(x0 // cell), int(x1 // cell) + 1):
                buckets.setdefault((gx, gy), []).append(index)
    seen: set[tuple[int, int]] = set()
    for group in buckets.values():
        for ia in range(len(group)):
            i = group[ia]
            if i in drop or feats[i] is None:
                continue
            for ib in range(ia + 1, len(group)):
                j = group[ib]
                if j <= i or j in drop or feats[j] is None:
                    continue
                key = (i, j)
                if key in seen:
                    continue
                seen.add(key)
                if not _parallel_pair(feats[i], feats[j]):
                    continue
                if not _bright_core(feats[i], feats[j], strokes[i], strokes[j], oklab):
                    continue
                weaker = j if _strength(strokes[i], feats[i]) >= _strength(strokes[j], feats[j]) else i
                drop.add(weaker)
                collapsed += 1
                if weaker == i:
                    break
    kept = [stroke for index, stroke in enumerate(strokes) if index not in drop]
    return kept, collapsed


def score_strokes(strokes: list[Stroke], oklab: np.ndarray) -> list[Stroke]:
    for stroke in strokes:
        _assign_score(stroke, oklab)
    return strokes


def filter_detail(strokes: list[Stroke], detail_keep: float, show_secondary: bool) -> list[Stroke]:
    floor = (1.0 - float(np.clip(detail_keep, 0.0, 1.0))) * _PRIMARY
    kept = []
    for stroke in strokes:
        if stroke.tier != "primary" and stroke.score < floor:
            continue
        if stroke.tier != "primary" and not show_secondary:
            continue
        kept.append(stroke)
    return kept


def _pair_junctions(graph: SkeletonGraph, max_turn: float) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = {}
    for node in graph.junctions:
        incident = []
        for edge_id in graph.adj.get(node, ()):
            edge = graph.edges[edge_id]
            if edge.u == edge.v:
                continue
            incident.append((edge_id, _inward_tangent(graph, edge, node)))
        if len(incident) < 3:
            continue
        candidates = []
        for i in range(len(incident)):
            for k in range(i + 1, len(incident)):
                angle = turn_through(incident[i][1], incident[k][1])
                if angle < max_turn:
                    candidates.append((angle, i, k))
        candidates.sort()
        used: set[int] = set()
        for _angle, i, k in candidates:
            if i in used or k in used:
                continue
            used.add(i)
            used.add(k)
            left = incident[i][0]
            right = incident[k][0]
            pairs[(left, node)] = right
            pairs[(right, node)] = left
    return pairs


def _inward_tangent(graph: SkeletonGraph, edge: Edge, node: int) -> np.ndarray:
    junction = graph.junctions[node]
    pts, _rad = _orient_from(edge, node)
    if len(pts) == 0:
        return np.array([1.0, 0.0], np.float32)
    dist = np.linalg.norm(pts - junction.xy, axis=1)
    outside = np.flatnonzero(dist > 1.5 * max(junction.radius, 0.6))
    if len(outside) >= 2:
        sample = pts[outside[:12]]
    else:
        sample = pts[: min(8, len(pts))]
    if len(sample) < 2:
        return unit(junction.xy - pts[0])
    return unit(sample[0] - sample[-1])


def _push_point(chunks_p, chunks_r, chunks_c, xy, radius, cusp: bool) -> None:
    chunks_p.append(np.asarray(xy, np.float32).reshape(1, 2))
    chunks_r.append(np.array([radius], np.float32))
    chunks_c.append(np.array([cusp], dtype=bool))


def _bridge_junction(chunks_p, chunks_r, chunks_c, node: int, graph: SkeletonGraph, nxt_xy: np.ndarray) -> None:
    if node not in graph.junctions:
        return
    junction = graph.junctions[node]
    prev = chunks_p[-1][-1]
    if float(np.hypot(*(prev - junction.xy))) > 0.75:
        _push_point(chunks_p, chunks_r, chunks_c, junction.xy, junction.radius, node in graph.cusp_nodes)
    elif node in graph.cusp_nodes:
        chunks_c[-1][-1] = True


def _open_ends(strokes: list[Stroke], min_len: float) -> list[dict]:
    ends = []
    for index, stroke in enumerate(strokes):
        if stroke.closed or len(stroke.pts) < 2:
            continue
        length = arc_length(stroke.pts)
        if length < min_len:
            continue
        radius = float(np.median(stroke.radii))
        ends.append(
            {
                "stroke": index,
                "end": 0,
                "p": stroke.pts[0].copy(),
                "t": _outward(stroke.pts, True),
                "r": radius,
                "len": length,
            }
        )
        ends.append(
            {
                "stroke": index,
                "end": 1,
                "p": stroke.pts[-1].copy(),
                "t": _outward(stroke.pts, False),
                "r": radius,
                "len": length,
            }
        )
    return ends


def _outward(pts: np.ndarray, at_start: bool) -> np.ndarray:
    if at_start:
        window = pts[: min(8, len(pts))]
        hint = pts[0] - pts[min(6, len(pts) - 1)]
    else:
        window = pts[-min(8, len(pts)) :]
        hint = pts[-1] - pts[max(0, len(pts) - 7)]
    if len(window) < 2 or float(np.linalg.norm(hint)) < 1e-6:
        return unit(hint)
    return unit(window[0] - window[-1] if at_start else window[-1] - window[0])


def _match_ends(ends: list[dict], strokes: list[Stroke], oklab: np.ndarray, gap_max: float) -> list[dict]:
    count = len(ends)
    if count < 2:
        return []
    candidates: list[tuple[int, int, float]] = []
    cell = max(gap_max, 4.0)
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, item in enumerate(ends):
        key = (int(item["p"][0] // cell), int(item["p"][1] // cell))
        buckets.setdefault(key, []).append(index)
    seen: set[tuple[int, int]] = set()
    for index, item in enumerate(ends):
        cx = int(item["p"][0] // cell)
        cy = int(item["p"][1] // cell)
        for oy in (cy - 1, cy, cy + 1):
            for ox in (cx - 1, cx, cx + 1):
                for other in buckets.get((ox, oy), ()):
                    if other <= index or (index, other) in seen:
                        continue
                    seen.add((index, other))
                    cost = _pair_cost(ends[index], ends[other], strokes, oklab, gap_max)
                    if cost is None:
                        continue
                    candidates.append((index, other, cost))
    if not candidates:
        return []
    parent = np.arange(count, dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for i, j, _cost in candidates:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups: dict[int, list[int]] = {}
    involved = {node for pair in candidates for node in pair[:2]}
    for node in involved:
        groups.setdefault(find(node), []).append(node)
    chosen: list[tuple[int, int]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        local = {node: pos for pos, node in enumerate(group)}
        matrix = np.full((len(group), len(group)), 1e5, dtype=np.float64)
        allowed: set[tuple[int, int]] = set()
        for i, j, cost in candidates:
            if i not in local or j not in local:
                continue
            matrix[local[i], local[j]] = cost
            matrix[local[j], local[i]] = cost
            allowed.add((i, j))
            allowed.add((j, i))
        rows, cols = linear_sum_assignment(matrix)
        for row, col in zip(rows, cols):
            if row >= col or matrix[row, col] >= 1e4:
                continue
            left = group[int(row)]
            right = group[int(col)]
            if (left, right) in allowed or (right, left) in allowed:
                chosen.append((left, right))
    matches = []
    for left, right in chosen:
        record = _bridge_record(ends[left], ends[right])
        matches.append(record)
    return matches


def _pair_cost(a: dict, b: dict, strokes: list[Stroke], oklab: np.ndarray, gap_max: float) -> float | None:
    if a["stroke"] == b["stroke"] and a["end"] == b["end"]:
        return None
    delta = b["p"] - a["p"]
    dist = float(np.hypot(delta[0], delta[1]))
    if dist < 2.0 or dist > gap_max:
        return None
    direction = delta / dist
    if not _axial(delta, a["t"]) or not _axial(-delta, b["t"]):
        return None
    if dot(a["t"], direction) <= _COS36 or dot(b["t"], -direction) <= _COS36:
        return None
    radius_gap = abs(a["r"] - b["r"]) / max(a["r"], b["r"], 0.6)
    if radius_gap >= 0.60:
        return None
    if a["stroke"] == b["stroke"]:
        stroke = strokes[a["stroke"]]
        if a["len"] < max(40.0, dist * 3.0):
            return None
        mid = stroke.pts[len(stroke.pts) // 2]
        if float(point_segment_distance(mid.reshape(1, 2), a["p"], b["p"])[0]) > 0.55 * dist:
            return None
    penalty = _barrier(oklab, a["p"], b["p"])
    if penalty is None:
        return None
    average = max(0.6, 0.5 * (a["r"] + b["r"]))
    cost = dist / average + 4.0 * ((1.0 - dot(a["t"], direction)) + (1.0 + dot(b["t"], direction)))
    cost += penalty
    if cost > 28.0:
        return None
    return float(cost)


def _axial(delta, tangent) -> bool:
    parallel = dot(delta, tangent)
    lateral = abs(cross2(delta, tangent))
    return parallel > 0.0 and parallel > 1.75 * lateral


def _barrier(oklab: np.ndarray, start: np.ndarray, end: np.ndarray) -> float | None:
    samples = np.stack([start * (1.0 - t) + end * t for t in (0.0, 0.5, 1.0)], axis=0).astype(np.float32)
    colors = sample_bilinear(oklab, samples)
    de_left = float(np.linalg.norm(colors[1] - colors[0]))
    de_right = float(np.linalg.norm(colors[1] - colors[2]))
    de_ends = float(np.linalg.norm(colors[0] - colors[2]))
    if de_left > 0.12 and de_right > 0.12 and de_ends < 0.08:
        return None
    if max(de_left, de_right) > 0.18:
        return 4.0
    if max(de_left, de_right) > 0.14:
        return 2.0
    return 0.0


def _bridge_record(a: dict, b: dict) -> dict:
    delta = b["p"] - a["p"]
    dist = max(float(np.hypot(delta[0], delta[1])), 1e-3)
    h1 = a["p"] + a["t"] * (dist / 3.0)
    h2 = b["p"] + b["t"] * (dist / 3.0)
    samples = sample_cubic_polyline(a["p"], h1, h2, b["p"], 1.15)[1:-1]
    if len(samples) == 0:
        samples = np.vstack((a["p"], b["p"])).astype(np.float32)
    radii = np.linspace(a["r"], b["r"], len(samples), dtype=np.float32)
    return {
        "i": a["stroke"],
        "ei": a["end"],
        "j": b["stroke"],
        "ej": b["end"],
        "samples": samples.astype(np.float32),
        "radii": radii,
        "record": {
            "p0": [float(a["p"][0]), float(a["p"][1])],
            "h1": [float(h1[0]), float(h1[1])],
            "h2": [float(h2[0]), float(h2[1])],
            "p1": [float(b["p"][0]), float(b["p"][1])],
            "kind": "gap",
        },
    }


def _merge_matches(strokes: list[Stroke], matches: list[dict]) -> tuple[list[Stroke], set[int]]:
    link: dict[tuple[int, int], dict] = {}
    for match in matches:
        link[(match["i"], match["ei"])] = {**match, "forward": True}
        link[(match["j"], match["ej"])] = {
            **match,
            "i": match["j"],
            "ei": match["ej"],
            "j": match["i"],
            "ej": match["ei"],
            "forward": False,
        }
    used: set[int] = set()
    output: list[Stroke] = []
    for start in range(len(strokes)):
        if start in used or start not in {item["i"] for item in matches}:
            continue
        if (start, 0) not in link:
            flip = False
        elif (start, 1) not in link:
            flip = True
        else:
            flip = False
        sequence = [(start, flip)]
        bridges = []
        used.add(start)
        tail_sid, tail_flip = start, flip
        closed = False
        guard = 0
        while guard < len(strokes) + 2:
            guard += 1
            tail_end = 0 if tail_flip else 1
            match = link.get((tail_sid, tail_end))
            if match is None:
                break
            other = int(match["j"])
            other_flip = int(match["ej"]) == 1
            if other == start:
                bridges.append(match)
                closed = True
                break
            if other in used:
                break
            bridges.append(match)
            used.add(other)
            sequence.append((other, other_flip))
            tail_sid, tail_flip = other, other_flip
        output.append(_assemble(strokes, sequence, bridges, closed))
    return output, used


def _assemble(strokes: list[Stroke], sequence: list[tuple[int, bool]], bridges: list[dict], closed: bool) -> Stroke:
    parts_p = []
    parts_r = []
    parts_c = []
    records = []
    for index, (sid, flip) in enumerate(sequence):
        if index > 0:
            _append_bridge(parts_p, parts_r, parts_c, records, bridges[index - 1])
        stroke = strokes[sid]
        pts = stroke.pts[::-1] if flip else stroke.pts
        rad = stroke.radii[::-1] if flip else stroke.radii
        cusp = stroke.cusp[::-1] if flip else stroke.cusp
        parts_p.append(np.asarray(pts, np.float32))
        parts_r.append(np.asarray(rad, np.float32))
        parts_c.append(np.asarray(cusp, bool))
        for record in stroke.bridges:
            records.append(record)
    if closed and bridges:
        _append_bridge(parts_p, parts_r, parts_c, records, bridges[-1])
        parts_p.append(parts_p[0][:1])
        parts_r.append(parts_r[0][:1])
        parts_c.append(np.zeros(1, dtype=bool))
    return Stroke(
        np.vstack(parts_p).astype(np.float32),
        np.concatenate(parts_r).astype(np.float32),
        np.concatenate(parts_c).astype(bool),
        closed=closed,
        bridges=records,
    )


def _append_bridge(parts_p, parts_r, parts_c, records, match: dict) -> None:
    samples = match["samples"]
    radii = match["radii"]
    if not match.get("forward", True):
        samples = samples[::-1]
        radii = radii[::-1]
    if len(samples):
        parts_p.append(np.asarray(samples, np.float32))
        parts_r.append(np.asarray(radii, np.float32))
        parts_c.append(np.zeros(len(samples), dtype=bool))
    record = dict(match["record"])
    if not match.get("forward", True):
        record = {
            "p0": match["record"]["p1"],
            "h1": match["record"]["h2"],
            "h2": match["record"]["h1"],
            "p1": match["record"]["p0"],
            "kind": match["record"].get("kind", "gap"),
        }
    records.append(record)


def _snap_tees(strokes: list[Stroke], gap_max: float) -> int:
    reach = 0.55 * gap_max
    if reach < 2.0:
        return 0
    probes = []
    for stroke in strokes:
        length = arc_length(stroke.pts) if len(stroke.pts) >= 2 else 0.0
        if len(stroke.pts) < 2 or length < 10.0:
            probes.append(None)
            continue
        samples = stroke.pts[::3]
        if len(samples) == 0 or float(np.hypot(*(samples[-1] - stroke.pts[-1]))) > 0.1:
            samples = np.vstack((samples, stroke.pts[-1])).astype(np.float32)
        probes.append({"pts": samples, "len": length})
    cell = max(reach, 4.0)
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index, probe in enumerate(probes):
        if probe is None:
            continue
        pts = probe["pts"]
        for k in range(len(pts)):
            key = (int(pts[k, 0] // cell), int(pts[k, 1] // cell))
            buckets.setdefault(key, []).append((index, k))
    count = 0
    for index, stroke in enumerate(strokes):
        if stroke.closed or probes[index] is None:
            continue
        for at_start in (True, False):
            origin = stroke.pts[0] if at_start else stroke.pts[-1]
            tangent = _outward(stroke.pts, at_start)
            hit = _first_hit(probes, buckets, cell, index, origin, tangent, reach, float(np.median(stroke.radii)))
            if hit is None:
                continue
            target, point, dist = hit
            _extend_to(stroke, at_start, origin, tangent, point, dist)
            _insert_on(strokes[target], point)
            count += 1
    return count


def _first_hit(probes, buckets, cell, owner, origin, tangent, reach, radius):
    best = None
    best_dist = reach
    lateral_max = max(1.7, 0.5 * radius)
    steps = max(1, int(reach / cell) + 1)
    seen: set[tuple[int, int]] = set()
    for step in range(steps + 1):
        pos = origin + tangent * (reach * step / steps)
        cx = int(pos[0] // cell)
        cy = int(pos[1] // cell)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                for index, choice in buckets.get((cx + ox, cy + oy), ()):
                    if index == owner or (index, choice) in seen:
                        continue
                    seen.add((index, choice))
                    point = probes[index]["pts"][choice]
                    rel = point - origin
                    ahead = float(rel[0] * tangent[0] + rel[1] * tangent[1])
                    if ahead < 2.0 or ahead > reach or ahead >= best_dist:
                        continue
                    lateral = abs(float(rel[0] * tangent[1] - rel[1] * tangent[0]))
                    if lateral > lateral_max:
                        continue
                    pts = probes[index]["pts"]
                    prev_i = max(0, choice - 1)
                    next_i = min(len(pts) - 1, choice + 1)
                    seg = pts[next_i] - pts[prev_i]
                    acute = angle_between(tangent, seg)
                    acute = min(acute, 180.0 - acute)
                    if acute <= 35.0:
                        continue
                    best_dist = ahead
                    best = (index, point.copy(), ahead)
    return best


def _extend_to(stroke: Stroke, at_start: bool, origin, tangent, point, dist: float) -> None:
    h1 = origin + tangent * (dist / 3.0)
    h2 = point - tangent * (dist / 3.0)
    samples = sample_cubic_polyline(origin, h1, h2, point, 1.15)
    radius = float(np.median(stroke.radii))
    extra_p = samples[1:-1]
    record = {
        "p0": [float(origin[0]), float(origin[1])],
        "h1": [float(h1[0]), float(h1[1])],
        "h2": [float(h2[0]), float(h2[1])],
        "p1": [float(point[0]), float(point[1])],
        "kind": "t",
    }
    if len(extra_p) == 0:
        extra_p = point.reshape(1, 2)
    else:
        extra_p = np.vstack((extra_p, point.reshape(1, 2)))
    extra_r = np.full(len(extra_p), radius, np.float32)
    extra_c = np.zeros(len(extra_p), dtype=bool)
    if at_start:
        stroke.pts = np.vstack((extra_p[::-1], stroke.pts)).astype(np.float32)
        stroke.radii = np.concatenate((extra_r[::-1], stroke.radii))
        stroke.cusp = np.concatenate((extra_c[::-1], stroke.cusp))
    else:
        stroke.pts = np.vstack((stroke.pts, extra_p)).astype(np.float32)
        stroke.radii = np.concatenate((stroke.radii, extra_r))
        stroke.cusp = np.concatenate((stroke.cusp, extra_c))
    stroke.bridges.append(record)


def _insert_on(stroke: Stroke, point: np.ndarray) -> None:
    pts = stroke.pts
    if len(pts) < 2:
        return
    nearest = float(np.min(np.hypot(pts[:, 0] - point[0], pts[:, 1] - point[1])))
    if nearest < 0.75:
        return
    ab = pts[1:] - pts[:-1]
    ap = point - pts[:-1]
    denom = np.maximum(np.sum(ab * ab, axis=1), 1e-8)
    t = np.clip(np.sum(ap * ab, axis=1) / denom, 0.0, 1.0)
    proj = pts[:-1] + t[:, None] * ab
    dist = np.linalg.norm(proj - point, axis=1)
    k = int(np.argmin(dist))
    if float(dist[k]) > 1.4:
        return
    snapped = proj[k].astype(np.float32)
    radius = np.float32((1.0 - t[k]) * stroke.radii[k] + t[k] * stroke.radii[k + 1])
    stroke.pts = np.insert(pts, k + 1, snapped, axis=0)
    stroke.radii = np.insert(stroke.radii, k + 1, radius)
    stroke.cusp = np.insert(stroke.cusp, k + 1, False)


def _feat(stroke: Stroke):
    length = arc_length(stroke.pts) if len(stroke.pts) >= 2 else 0.0
    if len(stroke.pts) < 4 or length < 12.0:
        return None
    step_index = max(1, len(stroke.pts) // 18)
    pts = stroke.pts[::step_index]
    rad = stroke.radii[::step_index]
    step = max(length / max(len(pts) - 1, 1), 1.0)
    if len(pts) < 4:
        return None
    tang = np.gradient(pts, axis=0)
    tang = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    return {
        "pts": pts,
        "rad": rad,
        "tang": tang.astype(np.float32),
        "bbox": (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        ),
        "length": length,
        "step": step,
    }


def _parallel_pair(fa, fb) -> bool:
    if fa is None or fb is None:
        return False
    if fa["bbox"][2] < fb["bbox"][0] - 8 or fb["bbox"][2] < fa["bbox"][0] - 8:
        return False
    if fa["bbox"][3] < fb["bbox"][1] - 8 or fb["bbox"][3] < fa["bbox"][1] - 8:
        return False
    dist = np.hypot(fa["pts"][:, None, 0] - fb["pts"][None, :, 0], fa["pts"][:, None, 1] - fb["pts"][None, :, 1])
    nearest = np.argmin(dist, axis=1)
    rows = np.arange(len(fa["pts"]))
    dists = dist[rows, nearest]
    close = dists <= 7.5
    if int(close.sum()) < 3:
        return False
    overlap = float(close.mean())
    shorter = min(fa["length"], fb["length"])
    overlap_ratio = float(close.sum() * fa["step"] / max(shorter, 1.0))
    if max(overlap, overlap_ratio) <= 0.55:
        return False
    median = float(np.median(dists[close]))
    if median < 1.5 or median > 7.5 or float(np.std(dists[close])) > 2.8:
        return False
    paired = fb["tang"][nearest]
    cos = np.clip(np.abs(np.sum(fa["tang"] * paired, axis=1)), 0.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return float(np.median(ang[close])) < 14.0


def _bright_core(fa, fb, stroke_a: Stroke, stroke_b: Stroke, oklab: np.ndarray) -> bool:
    count = min(16, len(fa["pts"]))
    pick = np.linspace(0, len(fa["pts"]) - 1, count).astype(np.int32)
    pa = fa["pts"][pick]
    dist = np.hypot(pa[:, None, 0] - fb["pts"][None, :, 0], pa[:, None, 1] - fb["pts"][None, :, 1])
    nearest = np.argmin(dist, axis=1)
    rows = np.arange(len(pa))
    gaps = dist[rows, nearest]
    keep = (gaps >= 1.5) & (gaps <= 7.5)
    if int(keep.sum()) < 3:
        return False
    pa = pa[keep]
    pb = fb["pts"][nearest[keep]]
    ra = fa["rad"][pick][keep]
    rb = fb["rad"][nearest[keep]]
    sep = pa - pb
    norm = np.maximum(np.hypot(sep[:, 0], sep[:, 1]), 1e-3)[:, None]
    sep = sep / norm
    mids = (pa + pb) * 0.5
    outs = np.vstack((pa + sep * (1.5 * ra + 2.6)[:, None], pb - sep * (1.5 * rb + 2.6)[:, None]))
    mid_l = sample_nearest(oklab, mids)[..., 0]
    out_l = sample_nearest(oklab, outs)[..., 0]
    return float(np.median(mid_l)) > float(np.median(out_l)) + 0.055


def _strength(stroke: Stroke, feat) -> float:
    return float(feat["length"]) * (0.45 + float(np.median(stroke.radii)))


def _assign_score(stroke: Stroke, oklab: np.ndarray) -> None:
    length = arc_length(stroke.pts)
    if len(stroke.pts) < 2 or length < 1.0:
        stroke.score = 0.0
        stroke.tier = "secondary"
        return
    step_index = max(1, len(stroke.pts) // 8)
    samples = stroke.pts[::step_index]
    rads = stroke.radii[::step_index]
    if len(samples) < 2:
        stroke.score = 0.15
        stroke.tier = "secondary"
        return
    tang = np.gradient(samples, axis=0)
    tang = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    normal = np.stack((-tang[:, 1], tang[:, 0]), axis=1)
    offset = (1.5 * rads + 3.0).astype(np.float32)
    left = sample_nearest(oklab, samples + normal * offset[:, None])
    right = sample_nearest(oklab, samples - normal * offset[:, None])
    delta = float(np.median(np.linalg.norm(left - right, axis=1)))
    span = float(np.hypot(np.ptp(samples[:, 0]), np.ptp(samples[:, 1])))
    score = (
        0.50 * float(np.clip(delta / 0.16, 0.0, 1.0))
        + 0.30 * float(np.clip(length / 90.0, 0.0, 1.0))
        + 0.20 * float(np.clip(span / 70.0, 0.0, 1.0))
    )
    stroke.score = float(np.clip(score, 0.0, 1.0))
    stroke.tier = "primary" if stroke.score >= _PRIMARY else "secondary"
