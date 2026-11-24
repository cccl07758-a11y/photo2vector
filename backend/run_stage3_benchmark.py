"""四张基准样张：阶段 1.5 → 阶段二 → 阶段三/四叠纸插画，并写出五栏对比图。"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import get_diagnosis, load_rgb, run_phase1  # noqa: E402
from presets import ROOT, SAMPLES  # noqa: E402
from stage2 import vectorize  # noqa: E402
from stage3 import illustrate  # noqa: E402

OUT_DIR = ROOT / "outputs" / "stage3_benchmark"
SVG_NS = "http://www.w3.org/2000/svg"
LAYERS = (
    "bg-shapes",
    "fg-base-silhouettes",
    "fg-detail-shapes",
    "strokes-secondary",
    "strokes-primary",
)


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
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    bar = 36
    canvas = Image.new("RGB", (image.width, image.height + bar), (22, 24, 28))
    canvas.paste(image, (0, bar))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(236, 232, 223), font=font)
    return canvas


def _check_svg(text: str, filename: str, failures: list[str]) -> None:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        failures.append(f"{filename} SVG 无法解析: {exc}")
        return
    if root.tag != f"{{{SVG_NS}}}svg":
        failures.append(f"{filename} 根节点不是 svg")
    ids = {node.attrib.get("id") for node in root.iter(f"{{{SVG_NS}}}g")}
    for required in LAYERS:
        if required not in ids:
            failures.append(f"{filename} 缺少图层 {required}")
    rects = [node.attrib.get("id") for node in root.iter(f"{{{SVG_NS}}}rect")]
    if "canvas-ground" not in rects:
        failures.append(f"{filename} 缺少 canvas-ground")


def _count_color(rgb: np.ndarray, predicate) -> int:
    return int(np.count_nonzero(predicate(rgb)))


def _subject_checks(short: str, rgb: np.ndarray, failures: list[str]) -> None:
    red = _count_color(rgb, lambda image: (image[:, :, 0] > 150) & (image[:, :, 1] < 110) & (image[:, :, 2] < 110))
    yellow = _count_color(
        rgb,
        lambda image: (image[:, :, 0] > 165)
        & (image[:, :, 1] > 125)
        & (image[:, :, 2] > 60)
        & (image[:, :, 2] < 185)
        & (image[:, :, 0] > image[:, :, 1])
        & (image[:, :, 1] + 15 > image[:, :, 2]),
    )
    magenta = _count_color(rgb, lambda image: (image[:, :, 0] > 150) & (image[:, :, 2] > 90) & (image[:, :, 1] < 130))
    if short == "骑士" and red < 40:
        failures.append(f"骑士红十字区域过少 ({red}px)")
    if short == "蓝砖" and yellow < 20:
        failures.append(f"蓝砖黄字区域过少 ({yellow}px)")
    if short == "海滩" and magenta < 20:
        failures.append(f"海滩洋红帽区域过少 ({magenta}px)")
    if short == "潮玩" and red + magenta < 30:
        failures.append(f"潮玩高饱和主体过少 ({red + magenta}px)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = _font(18)
    failures: list[str] = []
    for sample in SAMPLES:
        path = ROOT / sample["filename"]
        rgb = load_rgb(path)
        token = "stage3:" + sample["id"]
        diag = get_diagnosis(rgb, token)
        print(f"\n[{sample['short']}] {sample['scene']}", flush=True)
        result = run_phase1(rgb, diag["recommended_params"], token=token, diagnosis=diag)
        if result.ink is None or result.oklab is None:
            failures.append(f"{sample['short']} 阶段一没有墨线或 OKLab")
            continue
        vector = vectorize(
            result.ink,
            result.oklab,
            source_key="bench:" + sample["id"],
            display_size=(result.stats["width"], result.stats["height"]),
        )
        out = illustrate(
            result.oklab,
            vector["vector_data"]["paths"],
            display_size=(result.stats["width"], result.stats["height"]),
            source_key="bench:" + sample["id"],
            paths_key="default",
            w_tex=diag.get("w_tex_256"),
            energy_norm=result.energy_norm,
        )
        stats = out["illustration_stats"]
        total_shapes = stats["shape_count"]["bg"] + stats["shape_count"]["fg"]
        print(
            f"  stage3 {stats['stage3_ms']:7.1f} ms  shapes {stats['shape_count']['bg']}+{stats['shape_count']['fg']}={total_shapes}"
            f"  basins {stats.get('basins', 0)}  regions {stats.get('regions', 0)}"
            f"  anchors {stats['shape_anchors']}  strokes {stats['stroke_paths']}"
            f"  retained {stats['retained_stroke_ratio'] * 100:5.1f}%"
            f"  coverage {stats['coverage_ratio'] * 100:6.2f}%  shape {stats.get('shape_coverage', 0) * 100:6.2f}%",
            flush=True,
        )
        print("  palette " + " ".join(item["hex"] for item in out["palette"][:8]), flush=True)
        if stats["stage3_ms"] > 15000:
            failures.append(f"{sample['short']} stage3_ms={stats['stage3_ms']} 超过 15000ms")
        if stats["coverage_ratio"] < 0.999:
            failures.append(f"{sample['short']} 画布覆盖率 {stats['coverage_ratio']}")
        if stats.get("shape_coverage", 0) < 0.97:
            failures.append(f"{sample['short']} 色块并集覆盖率 {stats.get('shape_coverage')}")
        if not 20 <= total_shapes <= 60:
            failures.append(f"{sample['short']} 色块 {total_shapes} 不在 20~60")
        if len(out["palette"]) < 6:
            failures.append(f"{sample['short']} 调色盘不足 6 色")
        svg_name = f"{sample['short']}_illustration.svg"
        svg_path = OUT_DIR / svg_name
        svg_path.write_text(out["svg_text"], encoding="utf-8")
        _check_svg(out["svg_text"], svg_name, failures)
        _subject_checks(sample["short"], out["previews"]["shapes"], failures)
        panels = [
            _panel(result.original, "原图", font),
            _panel(out["previews"]["p0"], "P0 形态学平涂底稿", font),
            _panel(out["previews"]["l1"], "L1 固有色完整大部件 (无碎光影)", font),
            _panel(out["previews"]["shapes"], "L2 纯叠纸递归色块 (含单层暗面)", font),
            _panel(out["previews"]["final"], "最终线面矢量插画 SVG", font),
        ]
        gap = 8
        sheet_w = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
        sheet_h = max(panel.height for panel in panels)
        sheet = Image.new("RGB", (sheet_w, sheet_h), (12, 13, 16))
        x = 0
        for panel in panels:
            sheet.paste(panel, (x, 0))
            x += panel.width + gap
        compare = OUT_DIR / f"{sample['short']}_stage3_compare.png"
        sheet.save(compare, format="PNG", optimize=True)
        print(f"  svg {svg_path}", flush=True)
        print(f"  compare {compare}", flush=True)
    if failures:
        print("\n未通过：", flush=True)
        for item in failures:
            print(f"  - {item}", flush=True)
        raise SystemExit(1)
    print("\n阶段三基准通过", flush=True)


if __name__ == "__main__":
    main()
