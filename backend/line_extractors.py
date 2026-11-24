"""Stage 1C：ETF 切向平滑与四种提线模式。全部是 remap / 高斯 / NumPy 向量运算。"""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np

from tensor_norm import TensorFields

_GRIDS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_GRID_LOCK = threading.Lock()
# OpenCV 的 setNumThreads 是进程全局的。多层并行提线时串行化 remap，避免互相改线程数写出 NaN。
_EXTRACT_LOCK = threading.Lock()


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    key = (h, w)
    with _GRID_LOCK:
        cached = _GRIDS.get(key)
        if cached is None:
            yy, xx = np.indices((h, w), dtype=np.float32)
            cached = (xx, yy)
            if len(_GRIDS) > 8:
                _GRIDS.clear()
            _GRIDS[key] = cached
        return cached


def _remap(src: np.ndarray, mx: np.ndarray, my: np.ndarray, dst: np.ndarray | None = None) -> np.ndarray:
    """复用 dst 时，返回值只在下一次写入同一块缓冲前有效。"""
    if dst is None:
        dst = np.empty(src.shape, np.float32)
    cv2.remap(
        src,
        np.ascontiguousarray(mx),
        np.ascontiguousarray(my),
        interpolation=cv2.INTER_LINEAR,
        dst=dst,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return dst


def _pct(arr: np.ndarray, q: float) -> float:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    step = max(1, flat.size // 250_000)
    return float(np.percentile(flat[::step], q))


def extract_soft(
    mode: str,
    fields: TensorFields,
    params: dict,
    smoothed_rgb: np.ndarray,
    *,
    parallel_safe: bool = False,
) -> np.ndarray:
    """提线。OpenCV 5.1 的多线程 remap 会偶发写出 NaN，所以默认把线程数钉在 1。

    `parallel_safe=True` 时不再改全局线程数，调用方必须已经设成 1，才能多路并行。
    """
    if parallel_safe:
        return _extract_soft_impl(mode, fields, params, smoothed_rgb)
    with _EXTRACT_LOCK:
        threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            return _extract_soft_impl(mode, fields, params, smoothed_rgb)
        finally:
            cv2.setNumThreads(threads)


def _extract_soft_impl(mode: str, fields: TensorFields, params: dict, smoothed_rgb: np.ndarray) -> np.ndarray:
    mode = mode or "hybrid"
    if mode == "ps_dodge":
        ink = ps_dodge(smoothed_rgb, params)
    else:
        tx, ty = smooth_tangent(
            fields.tx,
            fields.ty,
            fields.energy_norm,
            int(params["etf_iters"]),
            int(params["etf_radius"]),
            float(params["eta"]),
        )
        # 平滑后的切向重新正交出法向，法向采样跟着流线走。
        nx = ty
        ny = np.negative(tx)
        paths = trace_flow(tx, ty, _flow_steps(params))
        if mode == "ridge":
            ink = ridge_response(fields, params, nx, ny, paths)
        elif mode == "fdog":
            ink = fdog_response(fields, params, nx, ny, paths)
        elif mode == "hybrid":
            ink = hybrid_response(fields, params, nx, ny, paths)
        else:
            raise ValueError(f"未知 extract_mode: {mode}")
    ink = np.nan_to_num(ink, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(ink, 0.0, 1.0).astype(np.float32)


def ps_dodge(smoothed_rgb: np.ndarray, params: dict) -> np.ndarray:
    """PS 减淡极值基线：去色 → 反相最小值（即局部最大）→ G/(Gmax) → 反转成墨线。

    contrast_norm=0 时保持原始比值，亮部线条被大分母压没、暗部噪声被小分母放大。
    contrast_norm>0 时叠上局部 min-max 对比，用来对照「归一化之后会好在哪里」。
    """
    rgb = smoothed_rgb.astype(np.float32)
    if float(rgb.max()) > 1.5:
        rgb = rgb / 255.0
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    radius = max(1, int(round(float(params["line_width"]))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    g_max = cv2.dilate(gray, kernel)
    ink = np.clip(1.0 - gray / (g_max + 1e-5), 0.0, 1.0).astype(np.float32)
    contrast = float(params.get("contrast_norm", 0.0))
    if contrast > 1e-4:
        g_min = cv2.erode(gray, kernel)
        local = np.clip((g_max - gray) / (g_max - g_min + 1e-4), 0.0, 1.0)
        ink = np.clip((1.0 - contrast) * ink + contrast * np.maximum(ink, local), 0.0, 1.0)
    return ink.astype(np.float32)


def smooth_tangent(
    tx: np.ndarray,
    ty: np.ndarray,
    energy: np.ndarray,
    iters: int,
    radius: int,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """ETF：先做符号无关的双角高斯投票，再沿切向 remap 做精确幅值加权。

    弱像素的方向被邻域强边缘接管。幅值权重就是
    0.5 * (1 + tanh(η (E_neighbor - E_center)))。
    """
    tx = np.ascontiguousarray(tx, dtype=np.float32)
    ty = np.ascontiguousarray(ty, dtype=np.float32)
    energy = np.ascontiguousarray(energy, dtype=np.float32)
    iters = int(np.clip(iters, 1, 4))
    radius = int(np.clip(radius, 1, 5))
    eta = float(eta)
    xx, yy = _grid(tx.shape[0], tx.shape[1])
    sigma = max(1.15, radius * 1.25)
    for _ in range(iters):
        tx, ty = _double_angle_vote(tx, ty, energy, sigma, eta)
    # 精确的切向幅值权重只在方向场稳定后再沿流线收一次。大图放到半分辨率做。
    if min(tx.shape) >= 480:
        tx, ty = _half_tangent_pass(tx, ty, energy, radius, eta)
    else:
        tx, ty = _tangent_remap_pass(tx, ty, energy, xx, yy, radius, eta)
    return tx, ty


def _half_tangent_pass(tx, ty, energy, radius, eta):
    height, width = tx.shape
    half_size = (width // 2, height // 2)
    tx_s = cv2.resize(tx, half_size, interpolation=cv2.INTER_AREA)
    ty_s = cv2.resize(ty, half_size, interpolation=cv2.INTER_AREA)
    en_s = cv2.resize(energy, half_size, interpolation=cv2.INTER_AREA)
    mag = np.maximum(np.sqrt(tx_s * tx_s + ty_s * ty_s), np.float32(1e-6))
    tx_s = (tx_s / mag).astype(np.float32)
    ty_s = (ty_s / mag).astype(np.float32)
    xx, yy = _grid(tx_s.shape[0], tx_s.shape[1])
    tx_s, ty_s = _tangent_remap_pass(tx_s, ty_s, en_s, xx, yy, radius, eta)
    tx_u = cv2.resize(tx_s, (width, height), interpolation=cv2.INTER_LINEAR)
    ty_u = cv2.resize(ty_s, (width, height), interpolation=cv2.INTER_LINEAR)
    dot = tx * tx_u + ty * ty_u
    sgn = np.copysign(np.float32(1.0), dot)
    tx_o = tx + tx_u * sgn
    ty_o = ty + ty_u * sgn
    mag = np.maximum(np.sqrt(tx_o * tx_o + ty_o * ty_o), np.float32(1e-6))
    return (tx_o / mag).astype(np.float32), (ty_o / mag).astype(np.float32)


def _double_angle_vote(tx, ty, energy, sigma, eta):
    # 切向没有正反：模糊 (cos2θ, sin2θ)，强边缘用 tanh(ηE) 加大投票权。
    vote = (0.5 * (1.0 + np.tanh(eta * energy))).astype(np.float32)
    weight = vote * (np.float32(0.05) + energy)
    vx = weight * (tx * tx - ty * ty)
    vy = weight * (np.float32(2.0) * tx * ty)
    sw = cv2.GaussianBlur(weight, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    sx = cv2.GaussianBlur(vx, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    sy = cv2.GaussianBlur(vy, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    ang = (0.5 * np.arctan2(sy, sx)).astype(np.float32)
    ntx = np.cos(ang).astype(np.float32)
    nty = np.sin(ang).astype(np.float32)
    weak = sw < 1e-5
    if np.any(weak):
        ntx = np.where(weak, tx, ntx)
        nty = np.where(weak, ty, nty)
    return ntx, nty


def _tangent_remap_pass(tx, ty, energy, xx, yy, radius, eta):
    accx = tx.copy()
    accy = ty.copy()
    accw = np.ones(tx.shape, np.float32)
    sigma = max(radius * 0.65, 0.8)
    buf_x = np.empty_like(tx)
    buf_y = np.empty_like(ty)
    buf_e = np.empty_like(energy)
    mx = np.empty_like(tx)
    my = np.empty_like(ty)
    dot = np.empty_like(tx)
    sgn = np.empty_like(tx)
    weighted = np.empty_like(tx)
    eta32 = np.float32(eta)
    for k in range(-radius, radius + 1):
        if k == 0:
            continue
        np.multiply(tx, np.float32(k), out=mx)
        np.add(xx, mx, out=mx)
        np.multiply(ty, np.float32(k), out=my)
        np.add(yy, my, out=my)
        _remap(tx, mx, my, buf_x)
        _remap(ty, mx, my, buf_y)
        _remap(energy, mx, my, buf_e)
        spatial = np.float32(math.exp(-0.5 * (k / sigma) ** 2))
        np.subtract(buf_e, energy, out=dot)
        np.multiply(dot, eta32, out=dot)
        np.tanh(dot, out=dot)
        # wm = spatial * 0.5 * (1 + tanh(eta*(En-Ec)))
        np.add(dot, np.float32(1.0), out=dot)
        np.multiply(dot, spatial * np.float32(0.5), out=dot)
        np.multiply(buf_x, tx, out=sgn)
        np.multiply(buf_y, ty, out=weighted)
        np.add(sgn, weighted, out=sgn)
        np.copysign(np.float32(1.0), sgn, out=sgn)
        np.multiply(buf_x, sgn, out=buf_x)
        np.multiply(buf_y, sgn, out=buf_y)
        np.multiply(buf_x, dot, out=weighted)
        accx += weighted
        np.multiply(buf_y, dot, out=weighted)
        accy += weighted
        accw += dot
    vx = accx / accw
    vy = accy / accw
    mag = np.sqrt(vx * vx + vy * vy)
    good = mag >= 1e-4
    tx_o = np.where(good, vx / np.maximum(mag, 1e-8), tx).astype(np.float32)
    ty_o = np.where(good, vy / np.maximum(mag, 1e-8), ty).astype(np.float32)
    return tx_o, ty_o


def trace_flow(tx: np.ndarray, ty: np.ndarray, steps: int, dt: float = 1.0):
    """RK2 中点法沿切向走出采样坐标。路径只跟切向场有关，几种墨线共用。

    长边超过 480 时在半分辨率上积分，再把坐标乘回全图。弯曲方向不变，采样量大约降到 1/4。
    """
    if steps <= 0:
        return []
    height, width = tx.shape
    if min(height, width) >= 480:
        half_size = (width // 2, height // 2)
        tx_s = cv2.resize(tx, half_size, interpolation=cv2.INTER_AREA)
        ty_s = cv2.resize(ty, half_size, interpolation=cv2.INTER_AREA)
        mag = np.sqrt(tx_s * tx_s + ty_s * ty_s)
        mag = np.maximum(mag, np.float32(1e-6))
        tx_s = (tx_s / mag).astype(np.float32)
        ty_s = (ty_s / mag).astype(np.float32)
        # 坐标留在半分辨率像素里，积分时把标量场缩小到同一网格，避免 remap 尺寸不一致。
        return _trace_flow_full(tx_s, ty_s, steps, dt)
    return _trace_flow_full(tx, ty, steps, dt)


def _trace_flow_full(tx: np.ndarray, ty: np.ndarray, steps: int, dt: float):
    xx, yy = _grid(tx.shape[0], tx.shape[1])
    sigma = max(steps * dt * 0.55, 0.8)
    half = np.float32(0.5 * dt)
    step = np.float32(dt)
    buf_x = np.empty_like(tx)
    buf_y = np.empty_like(ty)
    buf_mx = np.empty_like(tx)
    buf_my = np.empty_like(ty)
    dot = np.empty_like(tx)
    sgn = np.empty_like(tx)
    cx = np.empty_like(tx)
    cy = np.empty_like(ty)
    paths = []
    for direction in (1.0, -1.0):
        px = xx.copy()
        py = yy.copy()
        if direction > 0:
            np.copyto(cx, tx)
            np.copyto(cy, ty)
        else:
            np.negative(tx, out=cx)
            np.negative(ty, out=cy)
        for s in range(1, steps + 1):
            _remap(tx, px, py, buf_x)
            _remap(ty, px, py, buf_y)
            np.multiply(buf_x, cx, out=dot)
            np.multiply(buf_y, cy, out=sgn)
            np.add(dot, sgn, out=dot)
            np.copysign(np.float32(1.0), dot, out=sgn)
            np.multiply(buf_x, sgn, out=buf_x)
            np.multiply(buf_y, sgn, out=buf_y)
            # 中点 = p + 0.5 dt t(p)，再取 t(mid)。缓冲复用在乘完之后。
            np.multiply(buf_x, half, out=buf_mx)
            np.add(px, buf_mx, out=buf_mx)
            np.multiply(buf_y, half, out=buf_my)
            np.add(py, buf_my, out=buf_my)
            # 把对齐后的 t(p) 留在 cx/cy，供中点切向对齐，remap 会覆盖 buf。
            np.copyto(cx, buf_x)
            np.copyto(cy, buf_y)
            _remap(tx, buf_mx, buf_my, buf_x)
            _remap(ty, buf_mx, buf_my, buf_y)
            np.multiply(buf_x, cx, out=dot)
            np.multiply(buf_y, cy, out=sgn)
            np.add(dot, sgn, out=dot)
            np.copysign(np.float32(1.0), dot, out=sgn)
            np.multiply(buf_x, sgn, out=buf_x)
            np.multiply(buf_y, sgn, out=buf_y)
            np.multiply(buf_x, step, out=dot)
            np.add(px, dot, out=px)
            np.multiply(buf_y, step, out=dot)
            np.add(py, dot, out=py)
            np.copyto(cx, buf_x)
            np.copyto(cy, buf_y)
            weight = math.exp(-0.5 * ((s * dt) / sigma) ** 2)
            paths.append((px.copy(), py.copy(), weight))
    return paths


def integrate_paths(field: np.ndarray, paths) -> np.ndarray:
    """一维高斯加权：把破碎弱线沿流线积成连续墨，反向噪声互相抵消。"""
    if not paths:
        return field
    target = field
    restore = None
    map_h, map_w = paths[0][0].shape
    if (map_h, map_w) != field.shape:
        target = cv2.resize(field, (map_w, map_h), interpolation=cv2.INTER_AREA)
        restore = (field.shape[1], field.shape[0])
    acc = np.array(target, dtype=np.float32, copy=True)
    buf = np.zeros_like(acc)
    extra = np.zeros_like(acc)
    wsum = 1.0
    for mx, my, weight in paths:
        _remap(target, mx, my, buf)
        cv2.scaleAdd(buf, float(weight), acc, dst=extra)
        np.copyto(acc, extra)
        wsum += weight
    if wsum != 1.0:
        acc *= np.float32(1.0 / wsum)
    if restore is not None:
        acc = cv2.resize(acc, restore, interpolation=cv2.INTER_LINEAR)
    return acc.astype(np.float32, copy=False)


def _flow_steps(params: dict) -> int:
    return int(np.clip(round(float(params["flow_coherence"]) * 6.0), 0, 8))


def _sample_offset(field, xx, yy, nx, ny, dist, dst=None):
    return _remap(field, xx + np.float32(dist) * nx, yy + np.float32(dist) * ny, dst)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    span = np.float32(edge1 - edge0)
    t = np.clip((x - np.float32(edge0)) / span, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def ridge_response(fields, params, nx, ny, paths) -> np.ndarray:
    """法向二阶脊度：宽缓 AO 上远处能量仍平坦，自动归零；阶跃中心留下单根脊线。

    T 型交点和帽檐尖角的相干度天然只有 0.2~0.3。高能量处豁免这次惩罚，
    碎噪点（低能量且低相干）仍被压死。软影参考点取样远于 NMS，避免 2~3px
    的真实轮廓把自己减掉一半。
    """
    power = float(params["coh_power"])
    coherence = np.clip(fields.coherence, 0.0, 1.0)
    if power <= 1e-4:
        coh_w = np.ones_like(coherence, dtype=np.float32)
    else:
        coh_protected = np.maximum(coherence, _smoothstep(0.32, 0.72, fields.energy_norm))
        coh_w = np.power(np.clip(coh_protected, 0.0, 1.0), np.float32(power)).astype(np.float32)
    measure = (fields.energy_norm * coh_w).astype(np.float32)
    dist = max(float(params["line_width"]), 0.85)
    ao_dist = max(dist * 1.65, 2.4)
    xx, yy = _grid(measure.shape[0], measure.shape[1])
    plus = _sample_offset(measure, xx, yy, nx, ny, dist, np.empty_like(measure))
    minus = _sample_offset(measure, xx, yy, nx, ny, -dist, np.empty_like(measure))
    plus_far = _sample_offset(measure, xx, yy, nx, ny, ao_dist, np.empty_like(measure))
    minus_far = _sample_offset(measure, xx, yy, nx, ny, -ao_dist, np.empty_like(measure))
    ao = np.float32(params["ao_suppression"])
    ridge = measure - ao * np.float32(0.44) * (plus_far + minus_far)
    ridge = np.maximum(ridge, 0.0)
    side = np.maximum(plus, minus)
    prom = measure - side
    scale = _pct(np.abs(prom), 96.0) + 1e-6
    nms = 1.0 / (1.0 + np.exp(np.float32(-4.5) * prom / np.float32(scale)))
    ridge = (ridge * nms).astype(np.float32)
    return _calibrate(integrate_paths(ridge, paths))


def fdog_response(fields, params, nx, ny, paths) -> np.ndarray:
    """Kang 2007：法向 1D DoG + 流线积分 + XDoG 软阈值。"""
    field = _contrast_field(fields, params)
    sigma_c = float(params["fdog_sigma"]) * max(float(params["line_width"]), 0.6) / 2.0
    sigma_c = float(np.clip(sigma_c, 0.65, 3.0))
    sigma_s = 1.6 * sigma_c
    rho = float(params["fdog_rho"])
    radius = int(np.clip(math.ceil(sigma_s * 1.8), 2, 4))
    taps = np.arange(-radius, radius + 1, dtype=np.float32)
    g1 = np.exp(-0.5 * (taps / np.float32(sigma_c)) ** 2)
    g2 = np.exp(-0.5 * (taps / np.float32(sigma_s)) ** 2)
    g1 /= np.sum(g1)
    g2 /= np.sum(g2)
    kernel = (g1 - np.float32(rho) * g2).astype(np.float32)

    xx, yy = _grid(field.shape[0], field.shape[1])
    response = np.zeros(field.shape, np.float32)
    buf = np.empty_like(field)
    for offset, weight in zip(taps, kernel):
        if abs(float(weight)) < 2e-3:
            continue
        sample = _sample_offset(field, xx, yy, nx, ny, float(offset), buf)
        response += np.float32(weight) * sample
    response = integrate_paths(response, paths)
    return _xdog(response, float(params["fdog_eps"]), float(params["fdog_phi"]))


def hybrid_response(fields, params, nx, ny, paths) -> np.ndarray:
    """脊线负责彩色/亮暗阶跃的单线，FDoG 负责窄暗缝（眼缝/勾线）的实心墨。

    关键门控：暗凹谷（valley）必须同时具备高相干度与非零归一化结构能量，
    避免水泥麻点、碎石暗坑绕过除法归一化的安静度门控爆出胡椒噪点。
    """
    ridge = ridge_response(fields, params, nx, ny, paths)
    fdog = fdog_response(fields, params, nx, ny, paths)
    dist = max(float(params["line_width"]), 0.85)
    xx, yy = _grid(fields.L.shape[0], fields.L.shape[1])
    plus = _sample_offset(fields.L, xx, yy, nx, ny, dist, np.empty_like(fields.L))
    minus = _sample_offset(fields.L, xx, yy, nx, ny, -dist, np.empty_like(fields.L))
    valley = np.maximum(0.5 * (plus + minus) - fields.L, 0.0).astype(np.float32)
    # 乘上相干度与能量门控：无方向暗麻点 coh 低、energy_norm 低，禁止激活 fdog
    e_gate = np.clip(fields.energy_norm * np.float32(3.0), 0.0, 1.0)
    valley = valley * fields.coherence * e_gate
    hi = _pct(valley, 99.2) + 1e-6
    weight = np.clip((valley / np.float32(hi) - np.float32(0.32)) / np.float32(0.68), 0.0, 1.0)
    weight = cv2.GaussianBlur(weight, (0, 0), 0.7, borderType=cv2.BORDER_REFLECT)
    # fdog 分支同样受归一化能量轻门控保护，防止平坦噪点漏入
    fdog_clean = fdog * np.clip(fields.energy_norm * np.float32(4.0) + np.float32(0.15) * fields.coherence, 0.0, 1.0)
    return (ridge * (1.0 - weight) + fdog_clean * weight).astype(np.float32)


def _contrast_field(fields: TensorFields, params: dict) -> np.ndarray:
    """局部对比度均衡后的明度，再减去色度跳变，避免等明度撞色被 DoG 漏掉。"""
    luma = fields.L
    sigma = max(6.0, float(params["sigma_norm"]) * 0.55)
    radius = int(round(sigma))
    kernel = 2 * radius + 1
    mean = cv2.blur(luma, (kernel, kernel), borderType=cv2.BORDER_REFLECT)
    var = cv2.blur((luma - mean) ** 2, (kernel, kernel), borderType=cv2.BORDER_REFLECT)
    equalized = (luma - mean) / (np.sqrt(var + 1e-4) + np.float32(0.10))
    equalized = np.clip(equalized, -2.5, 2.5).astype(np.float32)
    ax = cv2.Scharr(fields.a, cv2.CV_32F, 1, 0) * np.float32(1.0 / 32.0)
    ay = cv2.Scharr(fields.a, cv2.CV_32F, 0, 1) * np.float32(1.0 / 32.0)
    bx = cv2.Scharr(fields.b, cv2.CV_32F, 1, 0) * np.float32(1.0 / 32.0)
    by = cv2.Scharr(fields.b, cv2.CV_32F, 0, 1) * np.float32(1.0 / 32.0)
    chroma = np.sqrt(ax * ax + ay * ay + bx * bx + by * by).astype(np.float32)
    hi = _pct(chroma, 99.0) + 1e-6
    chroma_n = np.clip(chroma / np.float32(hi), 0.0, 1.0)
    boost = float(params["chroma_weight"])
    return (equalized - np.float32(boost * 1.4) * chroma_n).astype(np.float32)


def _xdog(response: np.ndarray, eps: float, phi: float) -> np.ndarray:
    # 暗线在归一化后落在负半轴。T 在阈值之上为白，之下用 tanh 软落到墨色。
    neg = np.maximum(-response, 0.0)
    scale = _pct(neg, 95.0) + 1e-6
    normed = response / np.float32(scale)
    soft = 1.0 + np.tanh(np.float32(phi) * (normed - np.float32(eps)))
    thresholded = np.where(normed >= np.float32(eps), np.float32(1.0), soft)
    return np.clip(1.0 - thresholded, 0.0, 1.0).astype(np.float32)


def _calibrate(response: np.ndarray) -> np.ndarray:
    """把稀疏正响应拉到稳定的 [0, 1]，弱线略抬、近零噪声留在阈值之下。"""
    raw = np.maximum(response, 0.0).astype(np.float32)
    hi = _pct(raw, 99.2)
    if hi < 1e-4:
        return np.zeros_like(raw)
    x = np.clip(raw / np.float32(hi), 0.0, 1.0)
    # 底部留给纸白。分母过小时如果把噪声一起拉起来，滞后阈值会粘成整片黑。
    x = np.clip((x - np.float32(0.06)) / np.float32(0.94), 0.0, 1.0)
    k = np.float32(0.5)
    return np.clip(x * (1.0 + k) / (x + k), 0.0, 1.0).astype(np.float32)
