"""四张基准样张：阶段 1.5 Auto 提线之后做阶段二极简 SVG，并写出四栏对比图。"""

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

OUT_DIR = ROOT / "outputs" / "stage2_benchmark"
SVG_NS = "http://www.w3.org/2000/svg"


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


def _sample_cubic(p0, h1, h2, p1, steps: int = 12):
    points = []
    for index in range(steps + 1):
        t = index / steps
        u = 1.0 - t
        x = u**3 * p0[0] + 3 * u**2 * t * h1[0] + 3 * u * t**2 * h2[0] + t**3 * p1[0]
        y = u**3 * p0[1] + 3 * u**2 * t * h1[1] + 3 * u * t**2 * h2[1] + t**3 * p1[1]
        points.append((x, y))
    return points


def _draw_vector(paths: list[dict], width: int, height: int, microscope: bool) -> np.ndarray:
    image = Image.new("RGB", (width, height), (246, 243, 236))
    draw = ImageDraw.Draw(image)
    cursor = None
    start = None
    for path in paths:
        cursor = None
        start = None
        for cmd in path["commands"]:
            if cmd["op"] == "M":
                cursor = tuple(cmd["p"])
                start = cursor
            elif cmd["op"] == "L" and cursor and cmd.get("p"):
                end = tuple(cmd["p"])
                color = (18, 165, 106) if microscope else (20, 20, 23)
                width_px = 2 if microscope else max(1, int(round(path["stroke_width"])))
                draw.line([cursor, end], fill=color, width=width_px)
                cursor = end
            elif cmd["op"] == "C" and cursor and cmd.get("p"):
                samples = _sample_cubic(cursor, cmd["h1"], cmd["h2"], cmd["p"])
                color = (20, 52, 63) if microscope else (20, 20, 23)
                width_px = 2 if microscope else max(1, int(round(path["stroke_width"])))
                draw.line(samples, fill=color, width=width_px)
                cursor = tuple(cmd["p"])
            elif cmd["op"] == "Z" and cursor and start:
                draw.line([cursor, start], fill=(18, 165, 106) if microscope else (20, 20, 23), width=2)
                cursor = start
        if not microscope:
            continue
        for bridge in path["welded_bridges"]:
            samples = _sample_cubic(bridge["p0"], bridge["h1"], bridge["h2"], bridge["p1"], 10)
            draw.line(samples, fill=(255, 43, 214), width=2)
        for anchor in path["anchors"]:
            x, y = anchor["p"]
            if anchor.get("h_in"):
                draw.line([(x, y), tuple(anchor["h_in"])], fill=(126, 231, 255), width=1)
            if anchor.get("h_out"):
                draw.line([(x, y), tuple(anchor["h_out"])], fill=(126, 231, 255), width=1)
            if anchor["kind"] == "cusp":
                draw.rectangle([x - 3, y - 3, x + 3, y + 3], outline=(255, 138, 0), fill=(255, 179, 71))
            elif anchor["kind"] == "smooth":
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(11, 142, 171), fill=(62, 224, 255))
            else:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(102, 113, 124), fill=(231, 237, 242))
    return np.asarray(image)


def _check_svg(text: str, filename: str, failures: list[str]) -> None:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        failures.append(f"{filename} SVG 无法解析: {exc}")
        return
    if root.tag != f"{{{SVG_NS}}}svg":
        failures.append(f"{filename} 根节点不是 svg")
    ids = {node.attrib.get("id") for node in root.iter(f"{{{SVG_NS}}}g")}
    for required in ("strokes-primary", "strokes-secondary", "tapered-envelopes"):
        if required not in ids:
            failures.append(f"{filename} 缺少图层 {required}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = _font(18)
    failures: list[str] = []
    for sample in SAMPLES:
        path = ROOT / sample["filename"]
        rgb = load_rgb(path)
        token = "stage2:" + sample["id"]
        diag = get_diagnosis(rgb, token)
        print(f"\n[{sample['short']}] {sample['scene']}", flush=True)
        result = run_phase1(rgb, diag["recommended_params"], token=token, diagnosis=diag)
        if result.ink is None or result.oklab is None:
            failures.append(f"{sample['short']} 阶段一没有墨线或 OKLab")
            continue
        out = vectorize(
            result.ink,
            result.oklab,
            source_key="bench:" + sample["id"],
            display_size=(result.stats["width"], result.stats["height"]),
        )
        stats = out["vector_stats"]
        print(
            f"  stage2 {stats['stage2_ms']:7.1f} ms  paths {stats['total_paths']:3d}"
            f"  anchors {stats['total_anchors']:4d}  avg {stats['avg_anchors_per_path']:4.2f}"
            f"  straight {stats['straight_ratio'] * 100:5.1f}%"
            f"  L {stats['line_segments']:4d}  C {stats['cubic_segments']:3d}"
            f"  welds {stats['welded_count']:3d}  collapsed {stats['collapsed_double_lines']}",
            flush=True,
        )
        if stats["stage2_ms"] >= 1500:
            failures.append(f"{sample['short']} stage2_ms={stats['stage2_ms']} 超过 1500ms")
        if stats["total_paths"] < 1:
            failures.append(f"{sample['short']} 没有矢量路径")
        if not 2.0 <= stats["avg_anchors_per_path"] <= 12:
            failures.append(f"{sample['short']} 平均锚点 {stats['avg_anchors_per_path']} 不在 2~12")
        svg_name = f"{sample['short']}_vector_stage2.svg"
        svg_path = OUT_DIR / svg_name
        svg_path.write_text(out["svg_text"], encoding="utf-8")
        _check_svg(out["svg_text"], svg_name, failures)
        width = int(out["vector_data"]["width"])
        height = int(out["vector_data"]["height"])
        paths = out["vector_data"]["paths"]
        panels = [
            _panel(result.original, "原图", font),
            _panel(result.final, "阶段一位图线稿", font),
            _panel(_draw_vector(paths, width, height, True), "阶段二锚点显微镜", font),
            _panel(_draw_vector(paths, width, height, False), "阶段二最终 SVG 矢量线稿", font),
        ]
        gap = 8
        sheet_w = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
        sheet_h = max(panel.height for panel in panels)
        sheet = Image.new("RGB", (sheet_w, sheet_h), (12, 13, 16))
        x = 0
        for panel in panels:
            sheet.paste(panel, (x, 0))
            x += panel.width + gap
        compare = OUT_DIR / f"{sample['short']}_stage2_compare.png"
        sheet.save(compare, format="PNG", optimize=True)
        print(f"  svg {svg_path}", flush=True)
        print(f"  compare {compare}", flush=True)
    if failures:
        print("\n未通过：", flush=True)
        for item in failures:
            print(f"  - {item}", flush=True)
        raise SystemExit(1)
    print("\n阶段二基准通过", flush=True)


if __name__ == "__main__":
    main()
