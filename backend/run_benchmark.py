"""四张基准样张：PS 减淡基线 vs 通用固定参数 vs 零干预 Auto，并写出横向对比图。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import get_diagnosis, load_rgb, run_phase1  # noqa: E402
from presets import ROOT, SAMPLES, get_preset  # noqa: E402

# 基准期望只存在于测试脚本里。判别器本身看不到文件名。
_EXPECT = {
    "1webp.webp": {"method": "l0", "contrast_norm": (0.70, 0.80), "chroma_weight": (1.08, 1.25)},
    "d8f5338163ae2544623ba480c39f80b1.jpg": {
        "method": "rtv",
        "texture_scale": (8.5, 9.5),
        "contrast_norm": (0.32, 0.42),
    },
    "3ea9607179248d8eab1ef166f047c49b.jpg": {
        "method": "rgf",
        "contrast_norm": (0.70, 0.78),
        "quietness": (0.78, 0.84),
        "chroma_weight": (1.12, 1.28),
    },
    "397974f785f4416252dfa74b507c6a4f.jpg": {
        "method": "rtv",
        "texture_scale": (6.8, 8.2),
        "ao_suppression": (0.95, 1.01),
        "quietness": (0.50, 0.60),
    },
}

OUT_DIR = ROOT / "outputs" / "phase1_benchmark"


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _panel(rgb: np.ndarray, title: str, font: ImageFont.ImageFont) -> Image.Image:
    image = Image.fromarray(rgb)
    bar = 36
    canvas = Image.new("RGB", (image.width, image.height + bar), (22, 24, 28))
    canvas.paste(image, (0, bar))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(236, 232, 223), font=font)
    return canvas


def _save_line(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path, format="PNG", optimize=True)


def _check(label: str, stats: dict, preset: str) -> None:
    ratio = stats["line_ratio"]
    count = stats["components"]
    if preset == "ps_baseline":
        if ratio < 0.005:
            raise SystemExit(f"{label} PS 基线几乎没有线条 ratio={ratio:.3f}")
        return
    if ratio < 0.012 or ratio > 0.34 or count < 8:
        raise SystemExit(f"{label} 线稿失配 ratio={ratio:.3f} cc={count}")


def _expect(filename: str, diag: dict, failures: list[str]) -> None:
    spec = _EXPECT[filename]
    params = diag["recommended_params"]
    scores = diag["scores"]
    if scores["diagnose_ms"] >= 20:
        failures.append(f"{filename} diagnose_ms={scores['diagnose_ms']} 超过 20ms")
    if params["texture_method"] != spec["method"]:
        failures.append(f"{filename} method={params['texture_method']} 期望 {spec['method']}")
    for key, bounds in spec.items():
        if key == "method":
            continue
        value = float(params[key])
        lo, hi = bounds
        if value < lo or value > hi:
            failures.append(f"{filename} {key}={value:.3f} 不在 [{lo}, {hi}]")


def _save_bands(stem: str, result, font: ImageFont.ImageFont) -> None:
    bands = result.band_images or {}
    single = result.single_line if result.single_line is not None else result.final
    cells = (
        (result.original, "原图"),
        (single, "单层未分带对照"),
        (bands.get("shadow_line"), "暗部层线稿"),
        (bands.get("mid_line"), "中调层线稿"),
        (bands.get("highlight_line"), "高光层线稿"),
        (bands.get("color_bands"), "三色分带叠看"),
        (result.final, "多层合成线稿"),
    )
    missing = [title for image, title in cells if image is None]
    if missing:
        raise SystemExit(f"{stem} 缺少分带图：{', '.join(missing)}")
    panels = [_panel(image, title, font) for image, title in cells]
    gap = 8
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (width, height), (12, 13, 16))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    path = OUT_DIR / f"{stem}_bands.png"
    sheet.save(path, format="PNG", optimize=True)
    timing = result.timings_ms
    ratio = result.stats.get("band_ratio", {})
    print(
        f"  bands pack {timing.get('bands', 0):.0f} ms  compose {timing.get('compose', 0):.0f} ms"
        f"  shadow {ratio.get('shadow', 0):.3f}  mid {ratio.get('midtone', 0):.3f}"
        f"  highlight {ratio.get('highlight', 0):.3f}",
        flush=True,
    )
    print(f"  bands {path}", flush=True)


def main() -> None:
    from discriminator import diagnose_image

    diagnose_image(np.zeros((96, 128, 3), dtype=np.uint8))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = _font(18)
    failures = []
    for sample in SAMPLES:
        path = ROOT / sample["filename"]
        rgb = load_rgb(path)
        token = "bench:" + sample["id"]
        print(f"\n[{sample['filename']}] {sample['scene']}", flush=True)
        diag = get_diagnosis(rgb, token)
        scores = diag["scores"]
        params = diag["recommended_params"]
        metrics = diag["metrics"]
        print(
            f"  diagnose {scores['diagnose_ms']:5.1f} ms  {diag['texture_method']:4}  {diag['strategy_label']}",
            flush=True,
        )
        print(
            f"    周期 {scores['periodicity']:.2f}  肌理 {scores['roughness']:.2f}"
            f"  光比 {scores['hdr_unevenness']:.2f}  软影 {scores['soft_ao']:.2f}"
            f"  撞色 {scores['chroma_edge']:.2f}  σ {scores['noise_sigma']:.2e}",
            flush=True,
        )
        print(
            f"    scale {params['texture_scale']:.2f}  contrast {params['contrast_norm']:.2f}"
            f"  quiet {params['quietness']:.2f}  ao {params['ao_suppression']:.2f}"
            f"  chroma {params['chroma_weight']:.2f}  spatial {params['spatial_strength']:.2f}"
            f"  period {metrics['dominant_period_px']:.1f}px",
            flush=True,
        )
        for reason in diag["reasons"]:
            print(f"    · {reason}", flush=True)
        _expect(sample["filename"], diag, failures)
        results = {}
        jobs = (
            ("ps_baseline", get_preset("ps_baseline")),
            ("universal", get_preset("universal")),
            ("auto", params),
        )
        for preset_name, preset_params in jobs:
            result = run_phase1(rgb, preset_params, token=token + ":" + preset_name, diagnosis=diag)
            results[preset_name] = result
            timing = result.timings_ms
            stats = result.stats
            print(
                f"  {preset_name:12} total {timing['total']:7.1f} ms"
                f"  texture {timing['texture']:6.1f}  tensor {timing['tensor']:6.1f}"
                f"  extract {timing['extract']:6.1f}  post {timing['post']:5.1f}"
                f"  | {stats['width']}x{stats['height']}"
                f"  lines {stats['line_ratio'] * 100:5.2f}%  cc {stats['components']}",
                flush=True,
            )
            try:
                _check(sample["short"] + "/" + preset_name, stats, preset_name)
            except SystemExit as exc:
                failures.append(str(exc))
        stem = Path(sample["filename"]).stem
        _save_line(OUT_DIR / f"{stem}_ps_baseline.png", results["ps_baseline"].final)
        _save_line(OUT_DIR / f"{stem}_universal.png", results["universal"].final)
        _save_line(OUT_DIR / f"{stem}_auto.png", results["auto"].final)
        auto = results["auto"]
        labels = (
            (auto.original, "原图"),
            (results["ps_baseline"].final, "PS 减淡基线"),
            (auto.w_tex, "纹理蒙版 W_tex"),
            (auto.smoothed, "分区去纹理图"),
            (auto.final, f"全自动线稿 Auto · {auto.timings_ms['total']:.0f} ms"),
        )
        panels = [_panel(image, title, font) for image, title in labels]
        gap = 8
        width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
        height = max(panel.height for panel in panels)
        sheet = Image.new("RGB", (width, height), (12, 13, 16))
        x = 0
        for panel in panels:
            sheet.paste(panel, (x, 0))
            x += panel.width + gap
        compare_path = OUT_DIR / f"{stem}_compare.png"
        sheet.save(compare_path, format="PNG", optimize=True)
        print(f"  compare {compare_path}", flush=True)
        _save_bands(stem, auto, font)
    if failures:
        print("\n未通过：", flush=True)
        for item in failures:
            print(" ", item, flush=True)
        raise SystemExit(1)
    print(f"\n全部通过。对比图目录：{OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
