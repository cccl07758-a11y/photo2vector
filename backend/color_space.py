"""精确 OKLab 感知色彩空间（Björn Ottosson 标准系数，float32 向量化）。"""

from __future__ import annotations

import numpy as np

# Linear sRGB -> LMS
M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)

# LMS' -> OKLab
M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)

M1_INV = np.linalg.inv(M1).astype(np.float32)
M2_INV = np.linalg.inv(M2).astype(np.float32)
M1 = M1.astype(np.float32)
M2 = M2.astype(np.float32)

_C_LINEAR = np.float32(0.04045)
_C_SRGB = np.float32(0.0031308)
_A = np.float32(0.055)
_GAMMA = np.float32(2.4)
_INV_GAMMA = np.float32(1.0 / 2.4)
_SLOPE = np.float32(12.92)
_SCALE = np.float32(1.055)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.float32, copy=False)
    return np.where(c <= _C_LINEAR, c / _SLOPE, ((c + _A) / _SCALE) ** _GAMMA).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.maximum(c, 0).astype(np.float32, copy=False)
    return np.where(c <= _C_SRGB, c * _SLOPE, _SCALE * np.power(c, _INV_GAMMA) - _A).astype(np.float32)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb float32 HxWx3，范围 [0, 1]（或 uint8，会自动归一化）。"""
    arr = rgb.astype(np.float32, copy=False)
    if arr.dtype != np.float32 or float(np.max(arr)) > 1.5:
        arr = arr.astype(np.float32)
        if float(np.nanmax(arr)) > 1.5:
            arr = arr / np.float32(255.0)
    lin = srgb_to_linear(np.clip(arr, 0.0, 1.0))
    lms = lin @ M1.T
    lms_c = np.cbrt(np.maximum(lms, 0.0)).astype(np.float32)
    return (lms_c @ M2.T).astype(np.float32)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """OKLab -> sRGB float32 [0, 1]。用于去纹理后的预览图。"""
    lab = lab.astype(np.float32, copy=False)
    lms_c = lab @ M2_INV.T
    lms = lms_c * lms_c * lms_c
    lin = lms @ M1_INV.T
    return np.clip(linear_to_srgb(lin), 0.0, 1.0).astype(np.float32)
