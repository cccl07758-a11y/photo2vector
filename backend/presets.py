"""阶段一预设与参数校准。数值在四张基准样张上自校准后可直接使用。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 通用均衡是所有预设的底稿。场景预设只覆盖需要拧动的旋钮。
UNIVERSAL: dict = {
    "texture_method": "rgf",
    "texture_scale": 4.8,
    "chroma_weight": 1.0,
    "sigma_g": 1.35,
    "sigma_norm": 16.0,
    "quietness": 0.76,
    "contrast_norm": 0.68,
    "extract_mode": "hybrid",
    "line_width": 1.85,
    "coh_power": 1.30,
    "ao_suppression": 0.88,
    "flow_coherence": 0.56,
    "fdog_sigma": 1.15,
    "fdog_rho": 0.97,
    "fdog_eps": -0.32,
    "fdog_phi": 8.0,
    "eta": 2.4,
    "etf_iters": 2,
    "etf_radius": 2,
    "threshold_mode": "hysteresis",
    "thresh_low": 0.18,
    "thresh_high": 0.38,
    "min_component_size": 56,
    "min_span": 18.0,
    "thin_1px": False,
    "max_side": 1024,
    # 分区去纹理：纹理区走重度滤波，高相干主体保留轻度基底。
    "spatial_adaptive": True,
    "spatial_strength": 0.75,
    # 暗部 / 中调 / 高光三层虚拟曝光。single 退回单层管线做对照。
    "band_blend_mode": "weighted_max",
    "shadow_weight": 1.0,
    "mid_weight": 1.0,
    "highlight_weight": 1.0,
    "shadow_push": 0.65,
    "highlight_pull": 0.70,
    "shadow_hi": 0.42,
    "highlight_lo": 0.58,
    "band_overlap": 0.14,
    "mid_contrast": 1.15,
}

_OVERRIDES: dict[str, dict] = {
    "ps_baseline": {
        "texture_method": "bilateral",
        "texture_scale": 3.8,
        "chroma_weight": 0.0,
        "contrast_norm": 0.0,
        "quietness": 0.0,
        "extract_mode": "ps_dodge",
        "line_width": 2.0,
        "ao_suppression": 0.0,
        "flow_coherence": 0.0,
        "coh_power": 0.0,
        "threshold_mode": "hard",
        "thresh_low": 0.045,
        "thresh_high": 0.055,
        "min_component_size": 0,
        "min_span": 0.0,
        "thin_1px": False,
        "spatial_adaptive": False,
        "spatial_strength": 0.0,
        "band_blend_mode": "single",
    },
    "knight_l0": {
        "texture_method": "l0",
        "texture_scale": 4.2,
        "contrast_norm": 0.76,
        "quietness": 0.68,
        "line_width": 1.75,
        "ao_suppression": 0.92,
        "coh_power": 1.50,
        "flow_coherence": 0.65,
        "chroma_weight": 1.15,
        "thresh_low": 0.16,
        "thresh_high": 0.34,
        "min_component_size": 52,
        "min_span": 18.0,
    },
    "grid_rtv": {
        "texture_method": "rtv",
        "texture_scale": 9.2,
        "contrast_norm": 0.34,
        "quietness": 0.80,
        "chroma_weight": 0.90,
        "coh_power": 1.35,
        "ao_suppression": 0.85,
        "flow_coherence": 0.58,
        "thresh_low": 0.22,
        "thresh_high": 0.44,
        "min_component_size": 68,
        "min_span": 22.0,
    },
    "beach_hdr": {
        "texture_method": "rgf",
        "texture_scale": 5.8,
        "contrast_norm": 0.72,
        "quietness": 0.82,
        "chroma_weight": 1.25,
        "line_width": 1.80,
        "ao_suppression": 0.65,
        "coh_power": 1.40,
        "flow_coherence": 0.58,
        "sigma_norm": 18.0,
        "thresh_low": 0.19,
        "thresh_high": 0.38,
        "min_component_size": 56,
        "min_span": 18.0,
    },
    "toy_3d": {
        "texture_method": "rtv",
        "texture_scale": 7.6,
        "contrast_norm": 0.50,
        "quietness": 0.52,
        "ao_suppression": 1.0,
        "coh_power": 1.55,
        "extract_mode": "hybrid",
        "flow_coherence": 0.48,
        "thresh_low": 0.18,
        "thresh_high": 0.36,
        "min_component_size": 40,
        "min_span": 16.0,
    },
}

PRESET_META: list[dict] = [
    {
        "id": "auto",
        "label": "✦ 智能自适应",
        "badge": "Auto",
        "description": "12ms 纯数学探针自动诊断高低频纹理、光比与软阴影，自动选策略并开启空间分区保护。",
    },
    {
        "id": "universal",
        "label": "通用均衡",
        "badge": "默认",
        "description": "滚动引导去纹理 + 混合提线。四张样张共用的均衡起点。",
    },
    {
        "id": "ps_baseline",
        "label": "PS 减淡基线",
        "badge": "对照",
        "description": "双边模糊模拟特殊模糊，再做反相颜色减淡与单阈值。亮部丢线、暗部爆噪会被原样保留。",
    },
    {
        "id": "knight_l0",
        "label": "厚涂去笔触",
        "badge": "骑士",
        "description": "OKLab 上的 L0 梯度最小化，烫平笔触与锁子甲，保住甲片、剑和剪影。",
    },
    {
        "id": "grid_rtv",
        "label": "压瓷砖网格",
        "badge": "人像",
        "description": "相对总变分压低周期性格线，阈值只放行更强的主体轮廓。",
    },
    {
        "id": "beach_hdr",
        "label": "海滩强光均衡",
        "badge": "海滩",
        "description": "高局部对比度归一化拉起白墨镜，安静度压住纯蓝天空与水泥麻点。",
    },
    {
        "id": "toy_3d",
        "label": "3D 抑软影去格纹",
        "badge": "潮玩",
        "description": "RTV 去掉格纹和半调网点，脊线项抑制宽缓 AO 软阴影。",
    },
]

SAMPLES: list[dict] = [
    {
        "id": "1webp.webp",
        "filename": "1webp.webp",
        "short": "骑士",
        "title": "厚涂骑士",
        "scene": "厚涂骑士概念画（大明暗跨度 + 笔触/锁子甲 + 金属高光）",
        "scene_preset": "knight_l0",
    },
    {
        "id": "d8f5338163ae2544623ba480c39f80b1.jpg",
        "filename": "d8f5338163ae2544623ba480c39f80b1.jpg",
        "short": "蓝砖",
        "title": "广角蓝砖人像",
        "scene": "广角蓝砖人像（透视瓷砖网格 + 金色卷发 + 局部过曝）",
        "scene_preset": "grid_rtv",
    },
    {
        "id": "3ea9607179248d8eab1ef166f047c49b.jpg",
        "filename": "3ea9607179248d8eab1ef166f047c49b.jpg",
        "short": "海滩",
        "title": "正午海滩女孩",
        "scene": "正午海滩女孩（硬投影 + 水泥噪点 + 白墨镜贴浅肤）",
        "scene_preset": "beach_hdr",
    },
    {
        "id": "397974f785f4416252dfa74b507c6a4f.jpg",
        "filename": "397974f785f4416252dfa74b507c6a4f.jpg",
        "short": "潮玩",
        "title": "3D 街头潮玩",
        "scene": "3D 街头潮玩（衣服格纹/半调网点 + 柔和 AO 阴影）",
        "scene_preset": "toy_3d",
    },
]

_FLOATS = {
    "texture_scale": (0.0, 10.0),
    "chroma_weight": (0.0, 2.5),
    "sigma_g": (0.4, 6.0),
    "sigma_norm": (4.0, 40.0),
    "quietness": (0.0, 1.0),
    "contrast_norm": (0.0, 1.0),
    "line_width": (0.5, 8.0),
    "coh_power": (0.0, 4.0),
    "ao_suppression": (0.0, 1.25),
    "flow_coherence": (0.0, 1.0),
    "fdog_sigma": (0.6, 3.5),
    "fdog_rho": (0.85, 0.999),
    "fdog_eps": (-0.5, 0.2),
    "fdog_phi": (0.5, 40.0),
    "eta": (0.0, 8.0),
    "min_span": (0.0, 80.0),
    "spatial_strength": (0.0, 1.0),
    "thresh_low": (0.0, 1.0),
    "thresh_high": (0.0, 1.0),
    "shadow_weight": (0.0, 1.5),
    "mid_weight": (0.0, 1.5),
    "highlight_weight": (0.0, 1.5),
    "shadow_push": (0.0, 1.0),
    "highlight_pull": (0.0, 1.0),
    "shadow_hi": (0.15, 0.70),
    "highlight_lo": (0.30, 0.90),
    "band_overlap": (0.02, 0.35),
    "mid_contrast": (0.80, 1.80),
}
_BOOLS = {"thin_1px", "spatial_adaptive"}
_INTS = {
    "etf_iters": (1, 4),
    "etf_radius": (1, 5),
    "min_component_size": (0, 400),
    "max_side": (512, 1280),
}
_ENUMS = {
    "texture_method": ("none", "bilateral", "rgf", "l0", "rtv"),
    "extract_mode": ("ps_dodge", "ridge", "fdog", "hybrid"),
    "threshold_mode": ("hysteresis", "hard", "soft"),
    "band_blend_mode": ("weighted_max", "soft_sum", "single"),
}


def get_preset(name: str) -> dict:
    if name not in {m["id"] for m in PRESET_META}:
        raise KeyError(name)
    params = dict(UNIVERSAL)
    params.update(_OVERRIDES.get(name, {}))
    return params


def preset_payloads() -> list[dict]:
    out = []
    for meta in PRESET_META:
        item = dict(meta)
        item["params"] = get_preset(meta["id"])
        out.append(item)
    return out


def merge_params(overrides: dict | None) -> dict:
    """以通用预设为底，允许先套用 preset 再被显式字段覆盖。"""
    params = dict(UNIVERSAL)
    raw = dict(overrides or {})
    preset_name = raw.pop("preset", None)
    if preset_name:
        params = get_preset(str(preset_name))
    for key, value in raw.items():
        if key not in UNIVERSAL:
            continue
        params[key] = _coerce(key, value, params[key])
    if params["thresh_low"] > params["thresh_high"]:
        params["thresh_low"], params["thresh_high"] = params["thresh_high"], params["thresh_low"]
    return params


def _coerce(key: str, value, fallback):
    try:
        if key in _ENUMS:
            text = str(value)
            return text if text in _ENUMS[key] else fallback
        if key in _BOOLS:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if key in _INTS:
            lo, hi = _INTS[key]
            return int(np_clip(round(float(value)), lo, hi))
        if key in _FLOATS:
            lo, hi = _FLOATS[key]
            return float(min(hi, max(lo, float(value))))
    except (TypeError, ValueError):
        return fallback
    return fallback


def np_clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v
