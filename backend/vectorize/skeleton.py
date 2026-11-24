"""中轴、超级交点簇，以及毛刺剪枝。像素只走一遍链式追踪。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from vectorize.geom import arc_length, neighbor_table, turn_through, unit


@dataclass
class Junction:
    id: int
    xy: np.ndarray
    radius: float


@dataclass
class Edge:
    id: int
    u: int
    v: int
    pts: np.ndarray
    radii: np.ndarray


@dataclass
class SkeletonGraph:
    edges: dict[int, Edge] = field(default_factory=dict)
    adj: dict[int, list[int]] = field(default_factory=dict)
    junctions: dict[int, Junction] = field(default_factory=dict)
    cusp_nodes: set[int] = field(default_factory=set)
    deg: dict[int, int] = field(default_factory=dict)
    _next_edge: int = 1
    _next_endpoint: int = 1_000_000

    def degree(self, node: int) -> int:
        return int(self.deg.get(node, 0))

    def add_edge(self, u: int, v: int, pts: np.ndarray, radii: np.ndarray) -> Edge:
        edge = Edge(self._next_edge, int(u), int(v), pts, radii)
        self._next_edge += 1
        self.edges[edge.id] = edge
        self.adj.setdefault(edge.u, []).append(edge.id)
        if edge.v != edge.u:
            self.adj.setdefault(edge.v, []).append(edge.id)
        self._shift_degree(edge, 1)
        return edge

    def remove_edge(self, edge_id: int) -> None:
        edge = self.edges.pop(edge_id, None)
        if edge is None:
            return
        self._shift_degree(edge, -1)
        for node in {edge.u, edge.v}:
            bucket = self.adj.get(node)
            if not bucket:
                continue
            self.adj[node] = [item for item in bucket if item != edge_id]
            if not self.adj[node]:
                del self.adj[node]

    def _shift_degree(self, edge: Edge, sign: int) -> None:
        if edge.u == edge.v:
            self.deg[edge.u] = self.deg.get(edge.u, 0) + 2 * sign
            return
        self.deg[edge.u] = self.deg.get(edge.u, 0) + sign
        self.deg[edge.v] = self.deg.get(edge.v, 0) + sign


def extract_skeleton_graph(ink: np.ndarray, threshold: float = 0.32) -> SkeletonGraph:
    mask = np.nan_to_num(ink, nan=0.0) >= float(threshold)
    graph = SkeletonGraph()
    if not np.any(mask):
        return graph
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    skel = cv2.ximgproc.thinning(mask_u8, cv2.ximgproc.THINNING_ZHANGSUEN) > 0
    if not np.any(skel):
        return graph
    height, width = skel.shape
    skel_u8 = skel.astype(np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, cv2.CV_16S, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
    degree = neighbor_count.astype(np.int16) - 1
    degree[~skel] = 0
    radii = np.maximum(dist, 0.6).astype(np.float32)
    junction_mask = skel & (degree >= 3)
    labels = _cluster_junctions(junction_mask, radii)
    graph.junctions = _junction_table(labels, junction_mask, radii)
    _trace_chains(graph, skel & ~junction_mask, labels, radii, height, width)
    _dissolve_degree_two(graph)
    return graph


def prune_barbs(graph: SkeletonGraph, params: dict) -> SkeletonGraph:
    ratio = float(params["prune_ratio"])
    max_len = float(params["max_barb_len"])
    victims: list[int] = []
    for edge in list(graph.edges.values()):
        if edge.u == edge.v:
            continue
        junction_id, endpoint = _barb_ends(graph, edge)
        if junction_id is None or endpoint is None:
            continue
        length = arc_length(edge.pts)
        junction = graph.junctions[junction_id]
        if length >= ratio * max(junction.radius, 0.6) or length >= max_len:
            continue
        if _keep_spike(graph, edge, junction_id, endpoint):
            graph.cusp_nodes.add(junction_id)
            continue
        victims.append(edge.id)
    for edge_id in victims:
        graph.remove_edge(edge_id)
    _dissolve_degree_two(graph)
    _drop_isolated_short(graph, float(params["min_stroke_len"]))
    return graph


def _cluster_junctions(junction_mask: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """按局部半径膨胀后做连通域，把碎开的双 Y 收成超级交点。"""
    labels = np.zeros(junction_mask.shape, np.int32)
    if not np.any(junction_mask):
        return labels
    thin = junction_mask & (radii < 1.8)
    thick = junction_mask & ~thin
    offset = 0
    for mask, radius in ((thin, 2), (thick, 3)):
        if not np.any(mask):
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel)
        count, cc = cv2.connectedComponents(dilated, connectivity=8)
        active = mask & (cc > 0)
        if not np.any(active):
            continue
        labels[active] = cc[active] + offset
        offset += int(count)
    return labels


def _junction_table(labels: np.ndarray, junction_mask: np.ndarray, radii: np.ndarray) -> dict[int, Junction]:
    ys, xs = np.nonzero(junction_mask)
    table: dict[int, Junction] = {}
    if len(ys) == 0:
        return table
    labs = labels[ys, xs]
    order = np.argsort(labs, kind="mergesort")
    ys = ys[order]
    xs = xs[order]
    labs = labs[order]
    rads = radii[ys, xs]
    breaks = np.flatnonzero(np.diff(labs)) + 1
    starts = np.concatenate((np.array([0], np.int32), breaks))
    ends = np.concatenate((breaks, np.array([len(labs)], np.int32)))
    next_id = int(labs.max()) + 1 if len(labs) else 1
    for start, end in zip(starts, ends):
        label = int(labs[start])
        if label <= 0:
            continue
        block_x = xs[start:end].astype(np.float32)
        block_y = ys[start:end].astype(np.float32)
        block_r = rads[start:end]
        cx = float(block_x.mean())
        cy = float(block_y.mean())
        reach = np.hypot(block_x - cx, block_y - cy)
        if float(reach.max()) <= 7.0:
            table[label] = Junction(
                label,
                np.array([cx, cy], np.float32),
                max(float(np.median(block_r)), 0.6),
            )
            continue
        keys = (block_x.astype(np.int32) // 6) * 100000 + (block_y.astype(np.int32) // 6)
        for key in np.unique(keys):
            chosen = keys == key
            table[next_id] = Junction(
                next_id,
                np.array([float(block_x[chosen].mean()), float(block_y[chosen].mean())], np.float32),
                max(float(np.median(block_r[chosen])), 0.6),
            )
            labels[block_y[chosen].astype(np.int32), block_x[chosen].astype(np.int32)] = next_id
            next_id += 1
    return table


def _trace_chains(
    graph: SkeletonGraph,
    chain_mask: np.ndarray,
    labels: np.ndarray,
    radii: np.ndarray,
    height: int,
    width: int,
) -> None:
    ys, xs = np.nonzero(chain_mask)
    count = int(len(ys))
    if count == 0:
        return
    table = neighbor_table(ys.astype(np.int32), xs.astype(np.int32), height, width)
    chain_degree = np.count_nonzero(table >= 0, axis=1).astype(np.int8)
    visited = np.zeros(count, np.bool_)
    neigh: list[list[int]] = [[] for _ in range(count)]
    for column in range(8):
        ids = table[:, column]
        hit = np.flatnonzero(ids >= 0)
        if len(hit) == 0:
            continue
        for index, other in zip(hit.tolist(), ids[hit].tolist()):
            neigh[index].append(other)

    def labels_near(index: int) -> list[int]:
        y = int(ys[index])
        x = int(xs[index])
        found: list[int] = []
        for dy in (-1, 0, 1):
            yy = y + dy
            if yy < 0 or yy >= height:
                continue
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                xx = x + dx
                if xx < 0 or xx >= width:
                    continue
                label = int(labels[yy, xx])
                if label > 0 and label not in found:
                    found.append(label)
        return found

    def emit(path: list[int], closed: bool) -> None:
        if len(path) == 0:
            return
        idx = np.asarray(path, dtype=np.int32)
        pts = np.empty((len(idx), 2), np.float32)
        pts[:, 0] = xs[idx]
        pts[:, 1] = ys[idx]
        rad = radii[ys[idx], xs[idx]].astype(np.float32)
        if closed:
            node = graph._next_endpoint
            graph._next_endpoint += 1
            if arc_length(pts) < 4.0:
                return
            graph.add_edge(node, node, pts, rad)
            return
        left = labels_near(path[0])
        right = labels_near(path[-1])
        u = left[0] if left else 0
        v = right[0] if right else 0
        if u and v and u == v:
            for label in right + left:
                if label != u:
                    v = label
                    break
        if not u:
            u = _new_endpoint(graph)
        if not v:
            v = _new_endpoint(graph)
        if u == v and u in graph.junctions and arc_length(pts) < max(6.0, 3.2 * graph.junctions[u].radius):
            return
        graph.add_edge(u, v, pts, rad)

    for start in np.flatnonzero(chain_degree != 2):
        start = int(start)
        if visited[start]:
            continue
        if chain_degree[start] == 0:
            visited[start] = True
            emit([start], False)
            continue
        path = [start]
        visited[start] = True
        prev = -1
        cur = start
        guard = 0
        while guard < count + 2:
            guard += 1
            nxt = _next_unvisited(neigh[cur], prev, visited)
            if nxt < 0:
                break
            path.append(nxt)
            visited[nxt] = True
            prev, cur = cur, nxt
        emit(path, False)

    for start in range(count):
        if visited[start]:
            continue
        path = [start]
        visited[start] = True
        prev = -1
        cur = start
        closed = False
        guard = 0
        while guard < count + 2:
            guard += 1
            if len(path) >= 3 and start in neigh[cur]:
                closed = True
                break
            nxt = _next_unvisited(neigh[cur], prev, visited)
            if nxt < 0:
                break
            path.append(nxt)
            visited[nxt] = True
            prev, cur = cur, nxt
        emit(path, closed or (len(path) >= 4 and start in neigh[path[-1]]))


def _next_unvisited(neighbors: list[int], prev: int, visited: np.ndarray) -> int:
    for item in neighbors:
        if item != prev and not visited[item]:
            return int(item)
    return -1


def _new_endpoint(graph: SkeletonGraph) -> int:
    node = graph._next_endpoint
    graph._next_endpoint += 1
    return node


def _barb_ends(graph: SkeletonGraph, edge: Edge) -> tuple[int | None, int | None]:
    u_j = edge.u in graph.junctions
    v_j = edge.v in graph.junctions
    if u_j and not v_j and graph.degree(edge.v) <= 1:
        return edge.u, edge.v
    if v_j and not u_j and graph.degree(edge.u) <= 1:
        return edge.v, edge.u
    return None, None


def _keep_spike(graph: SkeletonGraph, edge: Edge, junction_id: int, endpoint: int) -> bool:
    others = []
    for edge_id in graph.adj.get(junction_id, ()):
        if edge_id == edge.id:
            continue
        others.append(_leaving_tangent(graph, graph.edges[edge_id], junction_id))
    if len(others) != 2:
        return False
    if turn_through(others[0], others[1]) <= 145.0:
        return False
    pts, rad = _orient_from(edge, junction_id)
    if len(rad) < 2:
        return False
    tip = float(rad[-1])
    root = float(np.median(rad[: max(1, len(rad) // 3)]))
    return tip + 0.12 < root


def _leaving_tangent(graph: SkeletonGraph, edge: Edge, node: int) -> np.ndarray:
    junction = graph.junctions.get(node)
    pts, _rad = _orient_from(edge, node)
    if junction is None:
        if len(pts) >= 2:
            return unit(pts[min(4, len(pts) - 1)] - pts[0])
        return np.array([1.0, 0.0], np.float32)
    if len(pts) == 1:
        return unit(pts[0] - junction.xy)
    step = min(len(pts) - 1, max(3, int(round(junction.radius + 2))))
    return unit(pts[step] - junction.xy)


def _orient_from(edge: Edge, node: int) -> tuple[np.ndarray, np.ndarray]:
    if edge.u == node:
        return edge.pts, edge.radii
    if edge.v == node:
        return edge.pts[::-1], edge.radii[::-1]
    return edge.pts, edge.radii


def _orient_toward(edge: Edge, node: int) -> tuple[np.ndarray, np.ndarray, int]:
    if edge.v == node and edge.u != node:
        return edge.pts, edge.radii, edge.u
    if edge.u == node and edge.v != node:
        return edge.pts[::-1], edge.radii[::-1], edge.v
    return edge.pts, edge.radii, edge.u


def _dissolve_degree_two(graph: SkeletonGraph) -> None:
    skipped: set[int] = set()
    guard = 0
    while guard < 4000:
        guard += 1
        target = None
        for node in list(graph.adj.keys()):
            if node in skipped or graph.degree(node) != 2:
                continue
            edge_ids = list(graph.adj.get(node, ()))
            if len(edge_ids) == 1 and graph.edges[edge_ids[0]].u == graph.edges[edge_ids[0]].v:
                continue
            target = node
            break
        if target is None:
            return
        if not _dissolve_node(graph, target):
            skipped.add(target)


def _dissolve_node(graph: SkeletonGraph, node: int) -> bool:
    incidences: list[int] = []
    for edge_id in list(graph.adj.get(node, ())):
        edge = graph.edges.get(edge_id)
        if edge is None or edge.u == edge.v:
            return False
        if edge.u == node:
            incidences.append(edge_id)
        if edge.v == node:
            incidences.append(edge_id)
    if len(incidences) != 2 or incidences[0] == incidences[1]:
        return False
    first = graph.edges[incidences[0]]
    second = graph.edges[incidences[1]]
    to_pts, to_rad, outer_a = _orient_toward(first, node)
    from_pts, from_rad = _orient_from(second, node)
    outer_b = _other(second, node)
    if len(to_pts) and len(from_pts) and float(np.hypot(*(to_pts[-1] - from_pts[0]))) < 1.25:
        pts = np.vstack((to_pts, from_pts[1:])) if len(from_pts) > 1 else to_pts.copy()
        rad = np.concatenate((to_rad, from_rad[1:])) if len(from_rad) > 1 else to_rad.copy()
    else:
        middle = graph.junctions[node].xy if node in graph.junctions else (to_pts[-1] + from_pts[0]) * 0.5
        mid_r = np.float32((float(to_rad[-1]) + float(from_rad[0])) * 0.5)
        pts = np.vstack((to_pts, middle.reshape(1, 2), from_pts))
        rad = np.concatenate((to_rad, np.array([mid_r], np.float32), from_rad))
    was_cusp = node in graph.cusp_nodes
    graph.remove_edge(first.id)
    graph.remove_edge(second.id)
    graph.junctions.pop(node, None)
    graph.cusp_nodes.discard(node)
    graph.add_edge(outer_a, outer_b, pts.astype(np.float32), rad.astype(np.float32))
    if was_cusp:
        # 尖角交点被溶解后，几何尖角留给拟合阶段的转角检测。
        pass
    return True


def _other(edge: Edge, node: int) -> int:
    return edge.v if edge.u == node else edge.u


def _drop_isolated_short(graph: SkeletonGraph, min_len: float) -> None:
    seen: set[int] = set()
    for seed in list(graph.adj.keys()):
        if seed in seen:
            continue
        stack = [seed]
        nodes: list[int] = []
        edges: set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            nodes.append(node)
            for edge_id in graph.adj.get(node, ()):
                edges.add(edge_id)
                edge = graph.edges[edge_id]
                other = edge.v if edge.u == node else edge.u
                if other not in seen:
                    stack.append(other)
        if any(node in graph.junctions and graph.degree(node) >= 3 for node in nodes):
            continue
        total = 0.0
        for edge_id in edges:
            total += arc_length(graph.edges[edge_id].pts)
        if total < min_len:
            for edge_id in list(edges):
                graph.remove_edge(edge_id)
