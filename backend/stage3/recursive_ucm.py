"""P1/P2 有向分水岭树，以及 P3 自顶向下的 L0→L1→L2 色块。"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import h_minima
from skimage.segmentation import watershed

from color_space import oklab_to_srgb
from stage3.edge_routing import lab_to_hex, raster_dams
from stage3.leveling import flatten

_HEAVY: OrderedDict[tuple, dict] = OrderedDict()
_HEAVY_LIMIT = 3


def build(
    lab: np.ndarray,
    saliency: np.ndarray,
    dams: list,
    params: dict,
    energy_norm: np.ndarray | None = None,
    cache_token: tuple | None = None,
) -> dict:
    lab = np.ascontiguousarray(lab, dtype=np.float32)
    saliency = np.clip(np.asarray(saliency, dtype=np.float32), 0.0, 1.0)
    if saliency.shape[:2] != lab.shape[:2]:
        saliency = cv2.resize(saliency, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)
    heavy = _heavy(lab, saliency, dams, params, energy_norm, cache_token)
    carved = _carve(heavy, lab, params, boost=0)
    if carved["count"] < 20:
        carved = _carve(heavy, lab, params, boost=1)
    if carved["count"] < 20:
        carved = _carve(heavy, lab, params, boost=2)
    if carved["count"] > 44:
        carved = _cull(carved, 38)
    colors = _harmonize(
        carved["colors"],
        carved["area"],
        carved["saliency"],
        carved["chroma"],
        int(params["palette_k"]),
    )
    carved["colors"] = colors
    return _pack(carved, heavy)


def _heavy(lab, saliency, dams, params, energy_norm, cache_token):
    if cache_token is not None:
        found = _HEAVY.get(cache_token)
        if found is not None:
            _HEAVY.move_to_end(cache_token)
            return found
    dust_ratio = 0.0009 * float(params["dust_scale"])
    lab_flat, l_star, p0_rgb = flatten(lab, saliency, dust_ratio)
    barrier = raster_dams(lab.shape[:2], dams)
    field = _watershed(lab_flat, barrier, energy_norm)
    color, area, smean = _stats(field["labels"], lab, saliency)
    field.update(
        {
            "lab_flat": lab_flat,
            "l_star": l_star,
            "p0_rgb": p0_rgb,
            "barrier": barrier,
            "color": color,
            "area": area,
            "smean": smean,
            "saliency": saliency,
        }
    )
    if cache_token is not None:
        _HEAVY[cache_token] = field
        _HEAVY.move_to_end(cache_token)
        while len(_HEAVY) > _HEAVY_LIMIT:
            _HEAVY.popitem(last=False)
    return field


def _watershed(lab_flat: np.ndarray, barrier: np.ndarray, energy_norm: np.ndarray | None) -> dict:
    luma = lab_flat[..., 0]
    chroma_a = lab_flat[..., 1]
    chroma_b = lab_flat[..., 2]
    gx_l, gy_l, mag_l = _slope(luma)
    gx_a, gy_a, mag_a = _slope(chroma_a)
    gx_b, gy_b, mag_b = _slope(chroma_b)
    grad_ab = np.hypot(mag_a, mag_b)
    energy = (np.float32(1.65) * grad_ab + np.float32(0.42) * mag_l).astype(np.float32)
    if energy_norm is not None:
        extra = _resize_channel(energy_norm, energy.shape)
        peak = float(np.percentile(extra, 99)) + 1e-6
        energy = energy + np.float32(0.22) * (extra / peak) * (float(np.percentile(energy, 95)) + 1e-6)
    if np.any(barrier):
        ridge = cv2.dilate(barrier.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        energy = energy + ridge * np.float32(float(np.percentile(energy, 98)) + 1e-6) * np.float32(1.35)
    energy = cv2.GaussianBlur(energy, (0, 0), 0.65)
    gx = np.float32(0.4) * gx_l + gx_a + gx_b
    gy = np.float32(0.4) * gy_l + gy_a + gy_b
    mag = np.hypot(gx, gy) + np.float32(1e-6)
    tangent_x = (-gy / mag).astype(np.float32)
    tangent_y = (gx / mag).astype(np.float32)
    markers = _markers(energy, barrier)
    basins = watershed(energy.astype(np.float64), markers, compactness=0.0005)
    basins = _fill_zeros(basins.astype(np.int32))
    basins = _compact(basins)
    links, dam_ratio = _oriented_links(basins, energy, barrier, tangent_x, tangent_y)
    return {"labels": basins, "links": links, "dam_ratio": dam_ratio, "energy": energy}


def _slope(channel: np.ndarray):
    gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, np.hypot(gx, gy).astype(np.float32)


def _markers(energy: np.ndarray, barrier: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(energy, 99)) + 1e-6
    normal = np.clip(energy / scale, 0.0, 1.5).astype(np.float64)
    blur = cv2.GaussianBlur(normal, (0, 0), 1.05)
    blocked = barrier.astype(bool) if barrier is not None else None
    best_markers = None
    best_gap = 10**9
    for depth in (0.035, 0.06, 0.09, 0.13, 0.18):
        minima = np.asarray(h_minima(blur, depth), dtype=bool)
        if blocked is not None:
            minima &= ~blocked
        count, markers = cv2.connectedComponents(minima.astype(np.uint8), connectivity=8)
        regions = count - 1
        gap = abs(regions - 260)
        if gap < best_gap and regions >= 2:
            best_gap = gap
            best_markers = markers
        if 180 <= regions <= 400:
            return markers.astype(np.int32)
    if best_markers is not None and best_gap <= 140:
        return best_markers.astype(np.int32)
    return _local_minima(blur, blocked).astype(np.int32)


def _local_minima(blur: np.ndarray, blocked: np.ndarray | None) -> np.ndarray:
    best = None
    best_gap = 10**9
    for window, pct in ((5, 58), (7, 46), (9, 36), (11, 28), (15, 20)):
        kernel = np.ones((window, window), np.uint8)
        floor = cv2.erode(blur.astype(np.float32), kernel)
        seeds = blur <= floor + 1e-6
        seeds &= blur <= np.percentile(blur, pct)
        if blocked is not None:
            seeds &= ~blocked
        count, markers = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)
        regions = count - 1
        gap = abs(regions - 260)
        if regions >= 2 and gap < best_gap:
            best_gap = gap
            best = markers
        if 180 <= regions <= 400:
            return markers
    if best is not None:
        return best
    grid = np.zeros(blur.shape, np.int32)
    step = 28
    seed = 1
    for y in range(step // 2, blur.shape[0], step):
        for x in range(step // 2, blur.shape[1], step):
            grid[y, x] = seed
            seed += 1
    return grid


def _fill_zeros(labels: np.ndarray) -> np.ndarray:
    filled = labels.copy()
    for _ in range(8):
        unknown = filled <= 0
        if not np.any(unknown):
            break
        donor = np.zeros(filled.shape, np.int32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.zeros_like(filled)
            if dy == 1:
                shifted[1:, :] = filled[:-1, :]
            elif dy == -1:
                shifted[:-1, :] = filled[1:, :]
            elif dx == 1:
                shifted[:, 1:] = filled[:, :-1]
            else:
                shifted[:, :-1] = filled[:, 1:]
            take = unknown & (donor <= 0) & (shifted > 0)
            donor[take] = shifted[take]
        if not np.any(donor > 0):
            break
        hit = donor > 0
        filled[hit] = donor[hit]
        unknown = filled <= 0
    filled[filled <= 0] = 1
    return filled


def _compact(labels: np.ndarray) -> np.ndarray:
    _uniq, inverse = np.unique(labels.ravel(), return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def _oriented_links(labels, energy, barrier, tangent_x, tangent_y):
    count = int(labels.max()) + 1
    store: dict[int, list[float]] = {}
    _pair(labels[:, :-1], labels[:, 1:], np.maximum(energy[:, :-1], energy[:, 1:]), np.maximum(barrier[:, :-1], barrier[:, 1:]), np.maximum(np.abs(tangent_y[:, :-1]), np.abs(tangent_y[:, 1:])), count, store)
    _pair(labels[:-1, :], labels[1:, :], np.maximum(energy[:-1, :], energy[1:, :]), np.maximum(barrier[:-1, :], barrier[1:, :]), np.maximum(np.abs(tangent_x[:-1, :]), np.abs(tangent_x[1:, :])), count, store)
    raws = []
    packed = []
    perim = np.zeros(count, np.float64)
    dam_acc = np.zeros(count, np.float64)
    for key, slot in store.items():
        length, energy_sum, dam_sum, align_sum, weight_sum = slot
        align = align_sum / max(weight_sum, 1e-6)
        raw = (energy_sum / max(length, 1.0)) * (0.08 + 0.92 * align)
        dam = dam_sum / max(length, 1.0)
        left = int(key // count)
        right = int(key % count)
        packed.append((left, right, raw, dam, length))
        raws.append(raw)
        perim[left] += length
        perim[right] += length
        dam_acc[left] += dam * length
        dam_acc[right] += dam * length
    scale = float(np.percentile(raws, 90)) if raws else 1.0
    scale = max(scale, 1e-6)
    links = {}
    for left, right, raw, dam, length in packed:
        strength = float(np.clip(raw / scale, 0.0, 2.5))
        links[(left, right) if left < right else (right, left)] = (strength, float(np.clip(dam, 0.0, 1.0)), float(length))
    ratio = dam_acc / np.maximum(perim, 1.0)
    return links, ratio.astype(np.float32)


def _pair(left, right, energy, dam, align, count, store):
    mask = left != right
    if not np.any(mask):
        return
    aa = left[mask].astype(np.int64)
    bb = right[mask].astype(np.int64)
    lo = np.minimum(aa, bb)
    hi = np.maximum(aa, bb)
    key = lo * np.int64(count) + hi
    weight = np.maximum(energy[mask].astype(np.float64), 1e-4)
    uniq, inverse = np.unique(key, return_inverse=True)
    lengths = np.bincount(inverse).astype(np.float64)
    energies = np.bincount(inverse, weights=energy[mask].astype(np.float64))
    dams = np.bincount(inverse, weights=dam[mask].astype(np.float64))
    aligns = np.bincount(inverse, weights=align[mask].astype(np.float64) * weight)
    weights = np.bincount(inverse, weights=weight)
    for item, length, energy_sum, dam_sum, align_sum, weight_sum in zip(uniq.tolist(), lengths, energies, dams, aligns, weights):
        slot = store.get(int(item))
        if slot is None:
            store[int(item)] = [float(length), float(energy_sum), float(dam_sum), float(align_sum), float(weight_sum)]
        else:
            slot[0] += float(length)
            slot[1] += float(energy_sum)
            slot[2] += float(dam_sum)
            slot[3] += float(align_sum)
            slot[4] += float(weight_sum)


def _stats(labels: np.ndarray, lab: np.ndarray, saliency: np.ndarray):
    count = int(labels.max()) + 1
    flat = labels.ravel()
    area = np.bincount(flat, minlength=count).astype(np.float64)
    sums = np.stack(
        [np.bincount(flat, weights=lab[..., channel].ravel().astype(np.float64), minlength=count) for channel in range(3)],
        axis=1,
    )
    color = (sums / np.maximum(area[:, None], 1.0)).astype(np.float64)
    smean = np.bincount(flat, weights=saliency.ravel().astype(np.float64), minlength=count) / np.maximum(area, 1.0)
    return color, area, smean.astype(np.float64)


def _resize_channel(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape[:2] == shape:
        return arr
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _carve(heavy: dict, lab: np.ndarray, params: dict, boost: int) -> dict:
    labels = heavy["labels"]
    height, width = labels.shape
    count = int(labels.max()) + 1
    zone, subject_mask = _zones(heavy, labels)
    albedo = float(params["albedo_group"])
    w_l = float(np.clip(0.60 - albedo, 0.10, 0.50))
    bg_target = int(round(np.interp(float(params["bg_simplify"]), [0.03, 0.22], [7, 3]))) + boost * 2
    fg_target = int(round(np.interp(float(params["fg_detail"]), [0.008, 0.09], [14, 6]))) + boost * 5
    bg_target = int(np.clip(bg_target, 3, 8))
    fg_target = int(np.clip(fg_target, 6, 22))
    merger = _Merger(heavy["color"], heavy["area"], heavy["links"])
    min_area = max(48, int(0.00009 * height * width))
    subjects = sorted({int(item) for item in zone if item > 0})
    merger.absorb(zone, 0, min_area, 0.45)
    merger.merge_to(zone, 0, bg_target, 0.45, 0.09, 0.072)
    fg_areas = []
    for subject in subjects:
        member = np.where(zone == subject)[0]
        fg_areas.append(float(heavy["area"][member].sum()) if len(member) else 0.0)
    shares = _share(fg_areas, fg_target)
    for subject, share in zip(subjects, shares):
        merger.absorb(zone, subject, min_area, w_l)
        merger.merge_to(zone, subject, share, w_l, 0.058, None)
    return _layers(heavy, lab, merger, zone, subject_mask, subjects, params, w_l)


class _Merger:
    def __init__(self, color, area, links):
        self.n = len(area)
        self.parent = np.arange(self.n, dtype=np.int32)
        self.color = np.asarray(color, dtype=np.float64).copy()
        self.area = np.asarray(area, dtype=np.float64).copy()
        self.edge = {key: (float(val[0]), float(val[1])) for key, val in links.items()}
        self.nbr: list[set[int]] = [set() for _ in range(self.n)]
        for left, right in self.edge:
            self.nbr[left].add(right)
            self.nbr[right].add(left)

    def find(self, index: int) -> int:
        parent = self.parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def cost(self, left: int, right: int, w_l: float) -> float | None:
        measured = self.metrics(left, right, w_l)
        if measured is None:
            return None
        return measured[0]

    def metrics(self, left: int, right: int, w_l: float):
        left, right = self.find(left), self.find(right)
        if left == right:
            return None
        info = self.edge.get((left, right) if left < right else (right, left))
        if info is None:
            return None
        strength, dam = info
        delta_l = abs(float(self.color[left, 0] - self.color[right, 0]))
        delta_ab = math.hypot(float(self.color[left, 1] - self.color[right, 1]), float(self.color[left, 2] - self.color[right, 2]))
        distance = math.hypot(w_l * delta_l, delta_ab)
        price = distance * (1.0 + 1.35 * strength) + 0.58 * dam
        return price, delta_ab, dam, distance

    def unite(self, left: int, right: int) -> int:
        left, right = self.find(left), self.find(right)
        if left == right:
            return left
        if self.area[right] > self.area[left]:
            left, right = right, left
        self.parent[right] = left
        total = float(self.area[left] + self.area[right])
        self.color[left] = (self.color[left] * self.area[left] + self.color[right] * self.area[right]) / max(total, 1e-6)
        self.area[left] = total
        for neighbor in list(self.nbr[right]):
            root = self.find(neighbor)
            if root == left:
                continue
            carried = self._pop_edge(right, neighbor)
            if carried is None:
                continue
            key = (left, root) if left < root else (root, left)
            prev = self.edge.get(key)
            if prev is None:
                self.edge[key] = carried
            else:
                self.edge[key] = (min(prev[0], carried[0]), max(prev[1], carried[1]))
            self.nbr[root].discard(right)
            self.nbr[root].discard(neighbor)
            self.nbr[root].add(left)
            self.nbr[left].add(root)
        self.nbr[right].clear()
        return left

    def _pop_edge(self, left: int, right: int):
        key = (left, right) if left < right else (right, left)
        return self.edge.pop(key, None)

    def roots(self, zone: np.ndarray, zone_id: int) -> list[int]:
        found = []
        seen = set()
        for index, item in enumerate(zone):
            if int(item) != zone_id or self.area[index] <= 0:
                continue
            root = self.find(index)
            if root not in seen and int(zone[root]) == zone_id:
                seen.add(root)
                found.append(root)
        return found

    def absorb(self, zone: np.ndarray, zone_id: int, min_area: float, w_l: float) -> None:
        for _ in range(6):
            roots = sorted(self.roots(zone, zone_id), key=lambda item: self.area[item])
            moved = False
            for root in roots:
                if self.area[root] >= min_area or self.area[root] <= 0:
                    continue
                if _vivid(self.color[root]) and self.area[root] >= 60:
                    continue
                best = None
                for neighbor in list(self.nbr[root]):
                    other = self.find(neighbor)
                    if other == root or int(zone[other]) != zone_id:
                        continue
                    price = self.cost(root, other, w_l)
                    if price is None:
                        continue
                    if best is None or price < best[0]:
                        best = (price, other)
                if best is None or best[0] > 0.28:
                    continue
                self.unite(root, best[1])
                moved = True
                break
            if not moved:
                break

    def merge_to(self, zone: np.ndarray, zone_id: int, target: int, w_l: float, limit_dab: float, limit_dist: float | None) -> None:
        roots = self.roots(zone, zone_id)
        alive = len(roots)
        target = max(1, min(int(target), max(alive, 1)))
        if alive <= target:
            return
        heap: list[tuple[float, int, int]] = []
        seen = set()
        for root in roots:
            for neighbor in self.nbr[root]:
                other = self.find(neighbor)
                if other == root or int(zone[other]) != zone_id:
                    continue
                key = (root, other) if root < other else (other, root)
                if key in seen:
                    continue
                seen.add(key)
                price = self.cost(root, other, w_l)
                if price is not None:
                    heapq.heappush(heap, (price, key[0], key[1]))
        while heap and alive > target:
            price, left, right = heapq.heappop(heap)
            a, b = self.find(left), self.find(right)
            if a == b or int(zone[a]) != zone_id or int(zone[b]) != zone_id:
                continue
            measured = self.metrics(a, b, w_l)
            if measured is None:
                continue
            now, dab, _dam, distance = measured
            if now > price + 1e-4:
                heapq.heappush(heap, (now, a, b))
                continue
            # 固有色差太大就留下这条边：脸和衣服不能为了凑块数焊在一起。
            if dab > limit_dab or (limit_dist is not None and distance > limit_dist):
                continue
            self.unite(a, b)
            alive -= 1
            survivor = self.find(a)
            for neighbor in list(self.nbr[survivor]):
                other = self.find(neighbor)
                if other == survivor or int(zone[other]) != zone_id:
                    continue
                fresh = self.cost(survivor, other, w_l)
                if fresh is not None:
                    heapq.heappush(heap, (fresh, survivor, other))


def _vivid(color: np.ndarray) -> bool:
    return bool(math.hypot(float(color[1]), float(color[2])) > 0.11)


def _zones(heavy: dict, labels: np.ndarray):
    """从高显著流域向外长，只穿过同色或被主体包住的邻居，得到 1~3 块剪影。"""
    color = heavy["color"]
    area = heavy["area"]
    smean = heavy["smean"]
    count = len(area)
    total = float(max(area.sum(), 1.0))
    touch = np.zeros(count, np.bool_)
    touch[labels[0, :]] = True
    touch[labels[-1, :]] = True
    touch[labels[:, 0]] = True
    touch[labels[:, -1]] = True
    nbr: list[list[tuple[int, float, float]]] = [[] for _ in range(count)]
    perim = np.zeros(count, np.float64)
    for (left, right), (strength, _dam, length) in heavy["links"].items():
        nbr[left].append((right, float(strength), float(length)))
        nbr[right].append((left, float(strength), float(length)))
        perim[left] += float(length)
        perim[right] += float(length)
    perim += np.where(touch, 24.0, 0.0)
    fg = np.zeros(count, np.bool_)
    for tau in (0.56, 0.48, 0.40, 0.32):
        core = (smean >= tau) & (area > 40) & ~(touch & (smean < tau + 0.1))
        if float(area[core].sum()) >= 0.05 * total:
            fg = core
            break
    if not np.any(fg):
        fg = (smean >= float(np.percentile(smean, 70))) & (area > 40)

    def dab(left: int, right: int) -> float:
        return float(np.hypot(color[left, 1] - color[right, 1], color[left, 2] - color[right, 2]))

    for _ in range(10):
        added = []
        for index in range(count):
            if fg[index] or area[index] <= 0:
                continue
            contact = 0.0
            nearest = 9.0
            blocked = 0.0
            for other, strength, length in nbr[index]:
                if not fg[other]:
                    continue
                contact += length
                nearest = min(nearest, dab(index, other))
                if strength > 1.2:
                    blocked += length
            if contact <= 0:
                continue
            frac = contact / max(perim[index], 1.0)
            # 大块低显著区域是墙、天、海，禁止顺着相近颜色连锁吞进去。
            if area[index] > 0.03 * total and smean[index] < 0.55:
                continue
            surrounded = frac >= 0.72 and not touch[index] and area[index] < 0.012 * total
            similar = nearest < 0.055 and smean[index] >= 0.24 and not (touch[index] and smean[index] < 0.55)
            soft = nearest < 0.09 and smean[index] >= 0.30 and not touch[index] and frac >= 0.18 and blocked < 0.55 * contact
            if surrounded or similar or soft:
                added.append(index)
        if not added:
            break
        for index in added:
            fg[index] = True
    mask = fg[labels]
    mask = _fill_holes(mask)
    mask = _drop_specks(mask, max(180, int(0.0005 * mask.size)))
    parts, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    hist = np.zeros((count, max(parts, 1)), np.int32)
    np.add.at(hist, (labels.ravel(), comp.ravel()), 1)
    owner = np.argmax(hist, axis=1).astype(np.int32)
    areas = np.bincount(comp.ravel(), minlength=max(parts, 1))
    ranked = sorted(((int(areas[item]), item) for item in range(1, parts)), reverse=True)
    floor = max(240, int(0.0014 * labels.size))
    kept = [item for area_i, item in ranked if area_i >= floor][:3]
    if not kept and ranked:
        kept.append(ranked[0][1])
    keep = set(kept)
    zone = np.zeros(count, np.int32)
    for index in range(count):
        if area[index] <= 0:
            zone[index] = -1
            continue
        choice = int(owner[index])
        zone[index] = choice if choice in keep else 0
    covered = np.isin(comp, list(keep)) if keep else np.zeros(mask.shape, np.bool_)
    return zone, covered


def _drop_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    areas = np.bincount(labels.ravel(), minlength=count)
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    flood = np.where(mask, np.uint8(0), np.uint8(255))
    cv2.floodFill(flood, np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8), (0, 0), 0)
    holes = flood == 255
    return mask | holes


def _share(areas: list[float], total: int) -> list[int]:
    if not areas:
        return []
    weights = np.maximum(np.asarray(areas, dtype=np.float64), 1.0)
    total = max(int(total), len(areas))
    raw = weights / weights.sum() * total
    base = np.maximum(np.floor(raw).astype(np.int32), 1)
    while int(base.sum()) > total and np.any(base > 1):
        base[int(np.argmax(base))] -= 1
    while int(base.sum()) < total:
        base[int(np.argmax(raw - base))] += 1
    return [int(item) for item in base]


def _layers(heavy, lab, merger, zone, subject_mask, subjects, params, w_l) -> dict:
    labels = heavy["labels"]
    root = np.array([merger.find(index) for index in range(merger.n)], dtype=np.int32)
    l0 = _paint_l0(labels, root, zone, subjects)
    l1 = _paint_ids(labels, root)
    l0_color = _medians(lab, l0["labels"], l0["count"])
    l1_color = _part_colors(lab, labels, root, l1["labels"], heavy["color"])
    final, colors, kind, parent, score = _details(
        labels,
        l1["labels"],
        l1_color,
        heavy,
        lab,
        params,
        root,
        zone,
    )
    _ = (w_l, subject_mask)
    return _describe(final, colors, kind, parent, score, l0, l0_color, l1, l1_color, heavy, lab)


def _paint_l0(labels, root, zone, subjects):
    basin_id = np.full(len(root), -1, np.int32)
    bg_roots = []
    seen = set()
    for index, item in enumerate(zone):
        if item != 0:
            continue
        survivor = int(root[index])
        if survivor in seen:
            continue
        seen.add(survivor)
        bg_roots.append(survivor)
    lookup = {item: pos for pos, item in enumerate(bg_roots)}
    for index, item in enumerate(zone):
        if item == 0:
            basin_id[index] = lookup.get(int(root[index]), 0)
        elif item > 0:
            basin_id[index] = len(bg_roots) + subjects.index(int(item))
    painted = basin_id[labels]
    painted[basin_id[labels] < 0] = 0
    return {"labels": painted.astype(np.int32), "count": int(painted.max()) + 1 if painted.size else 1}


def _paint_ids(labels, root):
    uniq, inverse = np.unique(root, return_inverse=True)
    painted = inverse[labels].astype(np.int32)
    return {"labels": painted, "count": int(len(uniq)), "inverse": inverse, "roots": uniq}


def _part_colors(lab, basins, root, l1_labels, means):
    _ = (basins, root, means)
    count = int(l1_labels.max()) + 1
    colors = np.zeros((count, 3), np.float32)
    for index in range(count):
        colors[index] = _median_mask(lab, l1_labels == index)
    return colors


def _details(basins, l1, l1_color, heavy, lab, params, root, zone):
    shade_threshold = float(params["shade_threshold"])
    final = l1.copy()
    next_id = int(l1.max()) + 1
    count_room = next_id + 48
    colors = np.zeros((count_room, 3), np.float32)
    colors[: len(l1_color)] = l1_color
    kind = np.array(["part"] * count_room, dtype=object)
    parent = np.full(count_room, -1, np.int32)
    score = np.zeros(count_room, np.float64)
    bg_roots = {int(root[index]) for index, item in enumerate(zone) if item == 0}
    compact_root = _paint_ids(basins, root)
    root_to_l1 = {int(item): pos for pos, item in enumerate(compact_root["roots"])}
    for index, item in enumerate(zone):
        if item == 0 and int(root[index]) in root_to_l1:
            kind[root_to_l1[int(root[index])]] = "bg"
    proposals = []
    for part in range(len(l1_color)):
        if kind[part] == "bg":
            continue
        mask = l1 == part
        area = int(mask.sum())
        if area < 120:
            continue
        member = [index for index in range(len(root)) if root_to_l1.get(int(root[index])) == part and zone[index] > 0]
        proposals.extend(_pattern_proposals(part, member, mask, heavy, basins, area))
    proposals.sort(key=lambda item: -item["score"])
    kept_per: dict[int, int] = {}
    accepted = []
    for item in proposals:
        if kept_per.get(item["part"], 0) >= 3 or len(accepted) >= 8:
            continue
        if item["score"] < 0.42:
            continue
        kept_per[item["part"]] = kept_per.get(item["part"], 0) + 1
        accepted.append(item)
    pattern_mask = np.zeros(l1.shape, np.bool_)
    for item in accepted:
        if next_id >= len(colors):
            break
        blob = item["mask"] & ~pattern_mask
        if int(blob.sum()) < 60:
            continue
        final[blob] = next_id
        colors[next_id] = _median_mask(lab, blob)
        kind[next_id] = "pattern"
        parent[next_id] = item["part"]
        score[next_id] = item["score"]
        pattern_mask |= blob
        next_id += 1
    for part in range(len(l1_color)):
        if kind[part] != "part":
            continue
        remain = (l1 == part) & ~pattern_mask
        shades = _shade_masks(remain, heavy["l_star"], shade_threshold)
        base_pixels = remain.copy()
        for blob in shades:
            base_pixels &= ~blob
        if int(base_pixels.sum()) >= 30:
            colors[part] = _median_mask(lab, base_pixels)
        for blob in shades:
            if next_id >= len(colors) or int(blob.sum()) < 50:
                continue
            final[blob] = next_id
            colors[next_id] = _shade_color(lab, blob, colors[part])
            kind[next_id] = "shadow"
            parent[next_id] = part
            score[next_id] = float(blob.sum())
            next_id += 1
    return final, colors[:next_id], kind[:next_id], parent[:next_id], score[:next_id]


def _pattern_proposals(part, member, part_mask, heavy, basins, part_area):
    if not member:
        return []
    means = heavy["color"]
    host = np.median(means[np.array(member, dtype=np.int32)], axis=0)
    picked = []
    for index in member:
        area = float(heavy["area"][index])
        if area < 70 or area > 0.46 * part_area:
            continue
        delta = math.hypot(float(means[index, 1] - host[1]), float(means[index, 2] - host[2]))
        enclosed = float(heavy["dam_ratio"][index]) >= 0.34 and delta >= 0.03
        if delta < 0.048 and not enclosed:
            continue
        picked.append(index)
    if not picked:
        return []
    clusters = _color_groups(picked, means, heavy["links"])
    found = []
    for group in clusters:
        mask = np.isin(basins, np.array(group, dtype=np.int32))
        closed = _bridge(mask, part_mask, part_area)
        count, comp = cv2.connectedComponents(closed.astype(np.uint8), connectivity=8)
        best_area = 0
        best = None
        for item in range(1, count):
            blob = comp == item
            area = int(blob.sum())
            if area > best_area:
                best_area = area
                best = blob
        if best is None or best_area < 70 or best_area > 0.5 * part_area:
            continue
        med = _median_mask(heavy["lab_flat"], best)
        delta = math.hypot(float(med[1] - host[1]), float(med[2] - host[2]))
        if delta < 0.04:
            continue
        found.append({"part": part, "mask": best, "score": delta * math.sqrt(best_area)})
    found.sort(key=lambda item: -item["score"])
    return found[:3]


def _color_groups(members, means, links):
    parent = {item: item for item in members}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    member_set = set(members)
    for left, right in links:
        if left not in member_set or right not in member_set:
            continue
        delta = math.hypot(float(means[left, 1] - means[right, 1]), float(means[left, 2] - means[right, 2]))
        if delta > 0.045:
            continue
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    groups: dict[int, list[int]] = {}
    for item in members:
        groups.setdefault(find(item), []).append(item)
    return list(groups.values())


def _bridge(mask: np.ndarray, part_mask: np.ndarray, part_area: int) -> np.ndarray:
    radius = int(np.clip(round(math.sqrt(max(part_area, 1)) * 0.045), 3, 11))
    if radius % 2 == 0:
        radius += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    return closed & part_mask


def _shade_masks(remain: np.ndarray, l_star: np.ndarray, shade_threshold: float) -> list[np.ndarray]:
    if int(remain.sum()) < 220:
        return []
    values = l_star[remain]
    if float(values.max() - values.min()) < shade_threshold:
        return []
    try:
        level = float(threshold_otsu(values))
    except ValueError:
        return []
    dark = values <= level
    frac = float(dark.mean())
    if not np.any(dark) or not np.any(~dark):
        return []
    delta = float(values[~dark].mean() - values[dark].mean())
    if delta < shade_threshold or not (0.12 <= frac <= 0.62):
        return []
    mask = np.zeros(remain.shape, np.uint8)
    mask[remain] = dark.astype(np.uint8)
    radius = int(np.clip(round(math.sqrt(int(remain.sum())) * 0.04), 3, 17))
    if radius % 2 == 0:
        radius += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = ((mask > 0) & remain).astype(np.uint8)
    count, comp = cv2.connectedComponents(mask, connectivity=8)
    ranked = sorted(((int((comp == item).sum()), item) for item in range(1, count)), reverse=True)
    kept = []
    part_area = max(int(remain.sum()), 1)
    for area, item in ranked:
        if area < int(0.08 * part_area):
            continue
        if len(kept) == 1 and area < int(0.45 * kept[0][0]):
            break
        if len(kept) >= 2:
            break
        kept.append((area, item))
    return [comp == item for _area, item in kept]


def _shade_color(lab, mask, parent_color) -> np.ndarray:
    med = _median_mask(lab, mask)
    out = parent_color.astype(np.float32).copy()
    out[0] = np.float32(min(float(med[0]), float(parent_color[0]) - 0.045))
    out[1] = np.float32(0.84 * float(parent_color[1]) + 0.16 * float(med[1]))
    out[2] = np.float32(0.84 * float(parent_color[2]) + 0.16 * float(med[2]))
    out[0] = np.clip(out[0], 0.0, 1.0)
    return out


def _median_mask(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(3, np.float32)
    pixels = lab[mask]
    if len(pixels) > 8000:
        step = int(math.ceil(len(pixels) / 8000))
        pixels = pixels[::step]
    return np.median(pixels, axis=0).astype(np.float32)


def _medians(lab, labels, count):
    colors = np.zeros((count, 3), np.float32)
    for index in range(count):
        colors[index] = _median_mask(lab, labels == index)
    return colors


def _describe(final, colors, kind, parent, score, l0, l0_color, l1, l1_color, heavy, lab) -> dict:
    count = len(colors)
    area = np.bincount(final.ravel(), minlength=count).astype(np.float64)
    smean = _mean_channel(heavy["saliency"], final, count)
    for index, name in enumerate(kind):
        if area[index] <= 0:
            continue
        if name == "bg":
            smean[index] = min(float(smean[index]), 0.16)
        else:
            smean[index] = max(float(smean[index]), 0.82)
    chroma = np.hypot(colors[:, 1], colors[:, 2]).astype(np.float32)
    alive = int(np.count_nonzero(area > 0))
    return {
        "labels": final.astype(np.int32),
        "colors": colors.astype(np.float32),
        "kind": kind,
        "parent": parent,
        "score": score,
        "area": area,
        "saliency": smean.astype(np.float32),
        "chroma": chroma,
        "count": alive,
        "l0": l0["labels"],
        "l0_color": l0_color,
        "l1": l1["labels"],
        "l1_color": l1_color,
        "basins": int(heavy["labels"].max()) + 1,
        "lab": lab,
    }


def _mean_channel(image, labels, count):
    flat = labels.ravel()
    acc = np.bincount(flat, weights=image.ravel().astype(np.float64), minlength=count)
    area = np.bincount(flat, minlength=count).astype(np.float64)
    return acc / np.maximum(area, 1.0)


def _cull(carved: dict, limit: int) -> dict:
    labels = carved["labels"].copy()
    kind = list(carved["kind"])
    parent = carved["parent"].copy()
    score = carved["score"].copy()
    area = np.bincount(labels.ravel(), minlength=len(kind)).astype(np.float64)

    def alive():
        return int(np.count_nonzero(np.bincount(labels.ravel(), minlength=len(kind)) > 0))

    details = [index for index, name in enumerate(kind) if name in {"pattern", "shadow"} and area[index] > 0]
    details.sort(key=lambda index: (0 if kind[index] == "pattern" else 1, float(score[index]), float(area[index])))
    for index in details:
        if alive() <= limit:
            break
        host = int(parent[index])
        if host < 0 or host >= len(kind):
            continue
        labels[labels == index] = host
    if alive() > limit:
        _merge_smallest(labels, carved["colors"], limit)
    carved = dict(carved)
    carved["labels"] = labels
    carved["area"] = np.bincount(labels.ravel(), minlength=len(kind)).astype(np.float64)
    carved["count"] = alive()
    return carved


def _merge_smallest(labels: np.ndarray, colors: np.ndarray, limit: int) -> None:
    stuck: set[int] = set()
    for _ in range(240):
        area = np.bincount(labels.ravel(), minlength=len(colors))
        alive = int(np.count_nonzero(area > 0))
        if alive <= limit:
            return
        order = [index for index in np.argsort(area) if area[index] > 0 and int(index) not in stuck]
        if not order:
            return
        victim = int(order[0])
        mask = labels == victim
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ring = dilated & ~mask
        if not np.any(ring):
            stuck.add(victim)
            continue
        neighbors = np.unique(labels[ring])
        best = None
        for other in neighbors:
            other = int(other)
            if other == victim or area[other] <= 0:
                continue
            delta = float(np.linalg.norm(colors[victim] - colors[other]))
            if area[victim] >= 360 and delta > 0.07:
                continue
            if best is None or delta < best[0]:
                best = (delta, other)
        if best is None:
            stuck.add(victim)
            continue
        labels[mask] = best[1]


def _harmonize(colors, area, saliency, chroma, palette_k: int) -> np.ndarray:
    alive = area > 0
    if int(alive.sum()) <= 1:
        return colors.astype(np.float32)
    weights = (np.maximum(area, 1.0) ** 0.55) * (0.15 + np.clip(saliency, 0.0, 1.0) ** 1.4) * (
        1.0 + 2.4 * np.clip(chroma, 0.0, None) ** 0.8
    )
    centers = _cluster(colors[alive], weights[alive], palette_k)
    out = colors.astype(np.float32).copy()
    for index in np.where(alive)[0]:
        delta = centers - colors[index]
        dist = np.linalg.norm(delta, axis=1)
        choice = int(np.argmin(dist))
        gap = float(dist[choice])
        if chroma[index] > 0.10 and gap > 0.055:
            continue
        if gap > 0.07:
            continue
        out[index] = centers[choice]
    return out


def _cluster(colors: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    count = len(colors)
    k = int(np.clip(k, 1, max(count, 1)))
    if count == 0:
        return np.zeros((1, 3), np.float32)
    chosen = [int(np.argmax(weights))]
    nearest = np.full(count, np.inf)
    for _ in range(1, k):
        dist = np.linalg.norm(colors - colors[chosen[-1]], axis=1)
        nearest = np.minimum(nearest, dist)
        rank = nearest * np.sqrt(np.maximum(weights, 1e-6))
        rank[chosen] = -1.0
        nxt = int(np.argmax(rank))
        if rank[nxt] <= 1e-8:
            break
        chosen.append(nxt)
    centers = colors[np.array(chosen, dtype=np.int32)].astype(np.float64).copy()
    for _ in range(6):
        dist = np.linalg.norm(colors[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(dist, axis=1)
        for index in range(len(centers)):
            sel = assign == index
            if not np.any(sel):
                continue
            centers[index] = np.average(colors[sel], axis=0, weights=np.maximum(weights[sel], 1e-6))
    return centers.astype(np.float32)


def _pack(carved: dict, heavy: dict) -> dict:
    labels = carved["labels"]
    colors = carved["colors"]
    area = carved["area"]
    total = float(area.sum()) + 1e-6
    buckets: dict[str, float] = {}
    hexes = []
    for index, pixels in enumerate(area):
        if index >= len(colors):
            hexes.append("#000000")
            continue
        hexes.append(lab_to_hex(colors[index]) if pixels > 0 else "#000000")
        if pixels > 0:
            buckets[hexes[-1]] = buckets.get(hexes[-1], 0.0) + float(pixels)
    palette = [{"hex": key, "ratio": round(value / total, 4)} for key, value in sorted(buckets.items(), key=lambda item: -item[1])]
    return {
        "labels": labels.astype(np.int32),
        "fill": colors.astype(np.float32),
        "mean": colors.astype(np.float32),
        "area": area.astype(np.float64),
        "saliency": carved["saliency"].astype(np.float32),
        "chroma": carved["chroma"].astype(np.float32),
        "hex": hexes,
        "palette": palette,
        "previews": {
            "p0": heavy["p0_rgb"],
            "l0": _paint(carved["l0"], carved["l0_color"]),
            "l1": _paint(carved["l1"], carved["l1_color"]),
        },
        "basins": int(carved.get("basins", 0)),
        "region_count": int(carved["count"]),
    }


def _paint(labels: np.ndarray, colors: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.round(oklab_to_srgb(colors.astype(np.float32)) * 255.0), 0, 255).astype(np.uint8)
    if len(rgb) <= int(labels.max()):
        rgb = np.vstack((rgb, np.zeros((int(labels.max()) + 1 - len(rgb), 3), np.uint8)))
    return rgb[labels]
