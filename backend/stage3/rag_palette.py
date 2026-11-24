"""以线筑坝的超像素层次合并，以及主体加权 OKLab 调色盘。"""

from __future__ import annotations

import heapq
import math

import cv2
import numpy as np

from color_space import oklab_to_srgb
from stage3.edge_routing import lab_to_hex, raster_dams
from stage3.saliency import _compact, _segment


def build(lab: np.ndarray, saliency: np.ndarray, dams: list, params: dict) -> dict:
    lab = np.ascontiguousarray(lab, dtype=np.float32)
    saliency = np.clip(np.asarray(saliency, dtype=np.float32), 0.0, 1.0)
    height, width = lab.shape[:2]
    target = int(np.clip(round(height * width / 780.0), 600, 850))
    labels = _segment(lab, target, 600, 850)
    barrier = raster_dams((height, width), dams)
    labels = _split_barriers(labels, barrier)
    parent = _agglomerate(labels, lab, saliency, barrier, float(params["bg_simplify"]), float(params["fg_detail"]))
    labels = _compact(parent[labels])
    labels = _absorb_small(labels, lab, saliency, barrier, _min_area(height, width))
    labels, palette = _snap_palette(labels, lab, saliency, barrier, int(params["palette_k"]))
    return _pack(labels, lab, saliency, palette)


def _min_area(height: int, width: int) -> int:
    return max(80, int(0.00028 * height * width))


def _split_barriers(labels: np.ndarray, barrier: np.ndarray) -> np.ndarray:
    if not np.any(barrier):
        return _compact(labels)
    free = np.where(barrier, np.uint8(0), np.uint8(1))
    parts, cc = cv2.connectedComponents(free, connectivity=4)
    key = labels.astype(np.int64) * np.int64(max(parts, 1)) + cc.astype(np.int64)
    key[barrier] = -1
    out = np.full(labels.shape, -1, np.int32)
    valid = key >= 0
    _uniq, inverse = np.unique(key[valid], return_inverse=True)
    out[valid] = inverse.astype(np.int32)
    return _compact(_fill_unknown(out))


def _fill_unknown(labels: np.ndarray) -> np.ndarray:
    filled = labels.copy()
    for _ in range(6):
        unknown = filled < 0
        if not np.any(unknown):
            break
        acc = np.full(filled.shape, -1, np.int32)
        for shifted in (_roll(filled, 1, 0), _roll(filled, -1, 0), _roll(filled, 0, 1), _roll(filled, 0, -1)):
            take = unknown & (acc < 0) & (shifted >= 0)
            acc[take] = shifted[take]
        if not np.any(acc >= 0):
            break
        hit = acc >= 0
        filled[hit] = acc[hit]
    filled[filled < 0] = 0
    return filled


def _roll(labels: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.full(labels.shape, -1, np.int32)
    if dy == 1:
        out[1:, :] = labels[:-1, :]
    elif dy == -1:
        out[:-1, :] = labels[1:, :]
    elif dx == 1:
        out[:, 1:] = labels[:, :-1]
    elif dx == -1:
        out[:, :-1] = labels[:, 1:]
    return out


def _features(labels: np.ndarray, lab: np.ndarray, saliency: np.ndarray):
    count = int(labels.max()) + 1
    flat = labels.ravel()
    area = np.bincount(flat, minlength=count).astype(np.float64)
    sums = np.stack(
        [np.bincount(flat, weights=lab[..., channel].ravel().astype(np.float64), minlength=count) for channel in range(3)],
        axis=1,
    )
    ssum = np.bincount(flat, weights=saliency.ravel().astype(np.float64), minlength=count)
    color = sums / np.maximum(area[:, None], 1.0)
    smean = ssum / np.maximum(area, 1.0)
    chroma = np.hypot(color[:, 1], color[:, 2])
    return color, area, smean, chroma


def _edges(labels: np.ndarray, barrier: np.ndarray, color: np.ndarray):
    count = int(labels.max()) + 1
    shared: dict[int, float] = {}
    dams: dict[int, float] = {}

    def accumulate(left, right, gate):
        mask = left != right
        if not np.any(mask):
            return
        aa = left[mask].astype(np.int64)
        bb = right[mask].astype(np.int64)
        lo = np.minimum(aa, bb)
        hi = np.maximum(aa, bb)
        key = lo * count + hi
        gate_v = gate[mask].astype(np.float64)
        uniq, inverse = np.unique(key, return_inverse=True)
        share = np.bincount(inverse).astype(np.float64)
        covered = np.bincount(inverse, weights=gate_v)
        for item, share_i, dam_i in zip(uniq, share, covered):
            slot = int(item)
            shared[slot] = shared.get(slot, 0.0) + float(share_i)
            dams[slot] = dams.get(slot, 0.0) + float(dam_i)

    accumulate(labels[:, :-1], labels[:, 1:], barrier[:, :-1] | barrier[:, 1:])
    accumulate(labels[:-1, :], labels[1:, :], barrier[:-1, :] | barrier[1:, :])
    edges = []
    for key, share in shared.items():
        left = int(key // count)
        right = int(key % count)
        if left >= len(color) or right >= len(color):
            continue
        delta = float(np.linalg.norm(color[left] - color[right]))
        edges.append((left, right, dams[key] / max(share, 1.0), delta))
    return edges


def _agglomerate(labels, lab, saliency, barrier, tau_bg: float, tau_fg: float) -> np.ndarray:
    color, area, smean, chroma = _features(labels, lab, saliency)
    raw = _edges(labels, barrier, color)
    count = len(area)
    parent = np.arange(count, dtype=np.int32)
    color = color.copy()
    area = area.copy()
    smean = smean.copy()
    chroma = chroma.copy()
    edge: dict[tuple[int, int], tuple[float, float]] = {}
    nbr: list[set[int]] = [set() for _ in range(count)]
    for left, right, dam, delta in raw:
        key = (left, right) if left < right else (right, left)
        edge[key] = (delta, float(dam))
        nbr[left].add(right)
        nbr[right].add(left)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def cost(left: int, right: int, delta: float, dam: float) -> float:
        sbar = float(np.clip(0.5 * (smean[left] + smean[right]), 0.0, 1.0))
        tau = tau_bg * ((1.0 - sbar) ** 0.85) + tau_fg * (sbar ** 0.85)
        # 主体内部没有水坝的低饱和渐变放宽到能并成固有色，点睛色和明暗跳变仍分开。
        if dam < 0.10 and chroma[left] < 0.09 and chroma[right] < 0.09:
            tau = max(tau, 0.105)
        jump = 1.0 if abs(float(smean[left] - smean[right])) > 0.32 else 0.0
        return delta / max(tau, 1e-4) + 7.5 * dam + 1.8 * jump

    def blocked(left: int, right: int, delta: float) -> bool:
        if delta <= 0.028:
            return False
        if _protected(smean[left], chroma[left]) or _protected(smean[right], chroma[right]):
            return True
        vivid_l = chroma[left] > 0.12 and smean[left] > 0.28
        vivid_r = chroma[right] > 0.12 and smean[right] > 0.28
        return bool((vivid_l or vivid_r) and delta > 0.04)

    heap: list[tuple[float, int, int]] = []
    for (left, right), (delta, dam) in edge.items():
        heapq.heappush(heap, (cost(left, right, delta, dam), left, right))
    pops = 0
    while heap and pops < 250000:
        pops += 1
        popped, i0, j0 = heapq.heappop(heap)
        left, right = find(i0), find(j0)
        if left == right:
            continue
        key = (left, right) if left < right else (right, left)
        info = edge.get(key)
        if info is None:
            continue
        delta, dam = info
        now = cost(left, right, delta, dam)
        if now > popped + 1e-4:
            heapq.heappush(heap, (now, left, right))
            continue
        if now >= 1.0:
            if now <= popped + 1e-3:
                break
            heapq.heappush(heap, (now, left, right))
            continue
        if blocked(left, right, delta):
            continue
        if area[right] > area[left]:
            left, right = right, left
        ai = float(area[left])
        aj = float(area[right])
        area[left] = ai + aj
        color[left] = (color[left] * ai + color[right] * aj) / area[left]
        smean[left] = (smean[left] * ai + smean[right] * aj) / area[left]
        chroma[left] = float(math.hypot(color[left, 1], color[left, 2]))
        parent[right] = left
        for neighbor in list(nbr[right]):
            root = find(neighbor)
            if root == left:
                continue
            old = (min(right, root), max(right, root))
            carried = edge.get(old)
            if carried is None:
                continue
            fresh = (min(left, root), max(left, root))
            if fresh in edge:
                d0, dam0 = edge[fresh]
                d1, dam1 = carried
                edge[fresh] = carried if cost(left, root, d1, dam1) < cost(left, root, d0, dam0) else (d0, dam0)
            else:
                edge[fresh] = carried
            nbr[root].discard(right)
            nbr[root].add(left)
            nbr[left].add(root)
            keep_d, keep_dam = edge[fresh]
            heapq.heappush(heap, (cost(left, root, keep_d, keep_dam), left, root))
        nbr[right].clear()
    return _resolve(parent)


def _protected(saliency: float, chroma: float) -> bool:
    return bool((saliency > 0.45 and chroma > 0.075) or (chroma > 0.11 and saliency > 0.18))


def _resolve(parent: np.ndarray) -> np.ndarray:
    pointer = parent.copy()
    while True:
        nxt = pointer[pointer]
        if np.array_equal(nxt, pointer):
            return pointer
        pointer = nxt


def _absorb_small(labels, lab, saliency, barrier, min_area: int) -> np.ndarray:
    for _ in range(2):
        color, area, smean, chroma = _features(labels, lab, saliency)
        edges = _edges(labels, barrier, color)
        nbr: dict[int, list[tuple[int, float, float]]] = {}
        for left, right, dam, delta in edges:
            nbr.setdefault(left, []).append((right, dam, delta))
            nbr.setdefault(right, []).append((left, dam, delta))
        mapping = np.arange(len(area), dtype=np.int32)
        changed = False
        for index in np.argsort(area):
            if area[index] <= 0 or area[index] >= min_area:
                continue
            if _protected(float(smean[index]), float(chroma[index])) and area[index] >= 40:
                continue
            best = None
            for other, dam, delta in nbr.get(int(index), []):
                if area[other] < area[index] or (area[other] == area[index] and other > index):
                    continue
                if dam > 0.55 and area[index] >= 40:
                    continue
                if best is None or delta < best[0]:
                    best = (delta, int(other), dam)
            if best is None:
                continue
            delta, other, dam = best
            if delta > 0.055 and dam > 0.20 and area[index] >= 50:
                continue
            mapping[index] = other
            changed = True
        if not changed:
            break
        for _hop in range(10):
            mapping = mapping[mapping]
        labels = _compact(mapping[labels])
    return labels


def _snap_palette(labels, lab, saliency, barrier, palette_k: int):
    color, area, smean, chroma = _features(labels, lab, saliency)
    alive = area > 0
    if not np.any(alive):
        return labels, []
    weights = (np.maximum(area, 1.0) ** 0.6) * (0.12 + np.clip(smean, 0.0, 1.0) ** 1.6) * (
        1.0 + 3.0 * np.clip(chroma, 0.0, None) ** 0.75
    )
    protect = (smean > 0.45) & (chroma > 0.075) & alive
    centers = _cluster(color[alive], weights[alive], palette_k, color[protect] if np.any(protect) else None)
    dist = _distance(color, centers)
    assign = np.argmin(dist, axis=1).astype(np.int32)
    snapped = centers[assign]
    # 同色心且没有水坝的相邻区域合成一块，避免调色盘吸附后留下无意义的碎缝。
    edges = _edges(labels, barrier, color)
    parent = np.arange(len(area), dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    for left, right, dam, _delta in sorted(edges, key=lambda item: item[3]):
        if dam > 0.18 or assign[left] != assign[right]:
            continue
        if abs(float(smean[left] - smean[right])) > 0.45:
            continue
        a, b = find(left), find(right)
        if a == b:
            continue
        if area[b] > area[a]:
            a, b = b, a
        parent[b] = a
    resolved = _resolve(parent)
    pixel = resolved[labels]
    uniq, inverse = np.unique(pixel.ravel(), return_inverse=True)
    new_labels = inverse.reshape(labels.shape).astype(np.int32)
    fill = np.zeros((len(uniq), 3), np.float32)
    for new_id, old_id in enumerate(uniq):
        fill[new_id] = snapped[int(old_id)]
    return new_labels, fill


def _cluster(colors: np.ndarray, weights: np.ndarray, k: int, accents: np.ndarray | None) -> np.ndarray:
    count = len(colors)
    if count == 0:
        return np.zeros((1, 3), np.float32)
    k = int(np.clip(k, 1, count))
    chosen = [int(np.argmax(weights))]
    nearest = np.full(count, np.inf, np.float64)
    for _ in range(1, k):
        dist = np.linalg.norm(colors - colors[chosen[-1]], axis=1)
        nearest = np.minimum(nearest, dist)
        score = nearest * np.sqrt(np.maximum(weights, 1e-6))
        score[chosen] = -1.0
        nxt = int(np.argmax(score))
        if score[nxt] <= 1e-8:
            break
        chosen.append(nxt)
    centers = colors[np.array(chosen, dtype=np.int32)].astype(np.float64).copy()
    assign = np.zeros(count, np.int32)
    for _ in range(8):
        dist = _distance(colors, centers)
        assign = np.argmin(dist, axis=1)
        for index in range(len(centers)):
            sel = assign == index
            if not np.any(sel):
                continue
            centers[index] = np.average(colors[sel], axis=0, weights=np.maximum(weights[sel], 1e-6))
    if accents is not None and len(accents):
        for accent in accents:
            dist = np.linalg.norm(centers - accent, axis=1)
            if float(dist.min()) > 0.055 and len(centers) < k + 8:
                centers = np.vstack((centers, np.asarray(accent, np.float64)))
    # 过近的色心收成一张，避免脸颊和袍子被拆成一串相近米色。
    if len(centers) > 1:
        dist = _distance(centers, centers)
        order = list(range(len(centers)))
        drop = set()
        for left in order:
            if left in drop:
                continue
            for right in order:
                if right <= left or right in drop:
                    continue
                if float(dist[left, right]) < 0.046:
                    drop.add(right)
        if drop:
            centers = centers[[index for index in order if index not in drop]]
    return centers.astype(np.float32)


def _distance(colors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    delta = colors[:, None, :].astype(np.float32) - centers[None, :, :].astype(np.float32)
    return np.sqrt(np.maximum(np.sum(delta * delta, axis=2), 0.0))


def _pack(labels: np.ndarray, lab: np.ndarray, saliency: np.ndarray, fill: np.ndarray) -> dict:
    if isinstance(fill, list) or fill is None or len(np.asarray(fill)) == 0:
        color, area, smean, chroma = _features(labels, lab, saliency)
        fill = color.astype(np.float32)
    else:
        color, area, smean, chroma = _features(labels, lab, saliency)
        fill = np.asarray(fill, dtype=np.float32)
        if len(fill) < len(color):
            grown = color.astype(np.float32).copy()
            grown[: len(fill)] = fill
            fill = grown
    total = float(area.sum()) + 1e-6
    buckets: dict[str, float] = {}
    hexes = []
    for index, pixel_area in enumerate(area):
        if pixel_area <= 0:
            hexes.append("#000000")
            continue
        hexes.append(lab_to_hex(fill[index]))
        buckets[hexes[-1]] = buckets.get(hexes[-1], 0.0) + float(pixel_area)
    palette = [
        {"hex": key, "ratio": round(value / total, 4)}
        for key, value in sorted(buckets.items(), key=lambda item: -item[1])
    ]
    return {
        "labels": labels.astype(np.int32),
        "fill": fill.astype(np.float32),
        "mean": color.astype(np.float32),
        "area": area.astype(np.float64),
        "saliency": smean.astype(np.float32),
        "chroma": chroma.astype(np.float32),
        "hex": hexes,
        "palette": palette,
    }


def flat_rgb(partition: dict) -> np.ndarray:
    labels = partition["labels"]
    rgb = np.clip(np.round(oklab_to_srgb(partition["fill"]) * 255.0), 0, 255).astype(np.uint8)
    if len(rgb) < int(labels.max()) + 1:
        rgb = np.vstack((rgb, np.zeros((int(labels.max()) + 1 - len(rgb), 3), np.uint8)))
    return rgb[labels]
