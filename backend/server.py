"""阶段一实验台 API。默认 127.0.0.1:8765，存在 web/dist 时同时托管前端。"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import StageCache, get_diagnosis, load_rgb, public_diagnosis, resize_max, run_phase1
from presets import ROOT, SAMPLES, merge_params, preset_payloads
from stage2 import vectorize as run_vectorize
from stage3 import illustrate as run_illustrate

HOST = "127.0.0.1"
PORT = 8765
UPLOAD_DIR = ROOT / "outputs" / "uploads"
DIST_DIR = ROOT / "web" / "dist"

_LOCK = threading.Lock()
_STAGE_CACHE = StageCache(limit=3)
_FILE_CACHE: dict[str, tuple[float, np.ndarray]] = {}
_PNG_CACHE: OrderedDict[str, bytes] = OrderedDict()
_SVG_CACHE: OrderedDict[str, tuple[str, str]] = OrderedDict()
_SOURCE_CACHE: OrderedDict[str, dict] = OrderedDict()
_UPLOADS: dict[str, dict] = {}


class ExtractBody(BaseModel):
    image_id: str
    params: dict = Field(default_factory=dict)
    auto_mode: bool = False


class DiagnoseBody(BaseModel):
    image_id: str


class VectorizeBody(BaseModel):
    image_id: str
    params: dict = Field(default_factory=dict)
    auto_mode: bool = False
    stage2: dict = Field(default_factory=dict)


class IllustrateBody(BaseModel):
    image_id: str
    params: dict = Field(default_factory=dict)
    auto_mode: bool = False
    stage2: dict = Field(default_factory=dict)
    stage3: dict = Field(default_factory=dict)


def _builtin_path(image_id: str) -> Path | None:
    for sample in SAMPLES:
        if sample["id"] == image_id:
            return ROOT / sample["filename"]
    return None


def _resolve_path(image_id: str) -> Path:
    builtin = _builtin_path(image_id)
    if builtin is not None and builtin.exists():
        return builtin
    uploaded = _UPLOADS.get(image_id)
    if uploaded is not None:
        return Path(uploaded["path"])
    raise HTTPException(404, "找不到这张图")


def _load_cached(image_id: str) -> np.ndarray:
    path = _resolve_path(image_id)
    stamp = path.stat().st_mtime
    found = _FILE_CACHE.get(image_id)
    if found is not None and found[0] == stamp:
        return found[1]
    rgb = load_rgb(path)
    # 交互分辨率到 1024，原图先收到 2560，避免超大图把内存撑满。
    rgb = resize_max(rgb, 2560)
    _FILE_CACHE[image_id] = (stamp, rgb)
    return rgb


def _thumb(rgb: np.ndarray, longest: int = 160) -> bytes:
    small = resize_max(rgb, longest)
    return _encode_png(small)


def _encode_png(array: np.ndarray) -> bytes:
    if array.ndim == 2:
        image = array
    elif array.shape[2] == 4:
        image = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
    else:
        image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise HTTPException(500, "PNG 编码失败")
    return buf.tobytes()


def _png_url(key: str, array: np.ndarray) -> str:
    cached = _PNG_CACHE.get(key)
    if cached is None:
        cached = _encode_png(array)
        _PNG_CACHE[key] = cached
        while len(_PNG_CACHE) > 160:
            _PNG_CACHE.popitem(last=False)
    else:
        _PNG_CACHE.move_to_end(key)
    return f"/api/cache/{key}.png"


def _image_token(image_id: str, rgb: np.ndarray) -> str:
    path = _resolve_path(image_id)
    return f"{image_id}:{path.stat().st_mtime}:{rgb.shape}"


def _resolve_extract_params(body: ExtractBody, diag: dict) -> dict:
    raw = dict(body.params or {})
    preset = raw.get("preset")
    auto = bool(body.auto_mode) or preset == "auto"
    if not auto:
        return merge_params(raw)
    params = dict(diag["recommended_params"])
    for key, value in raw.items():
        if key == "preset" or key not in params:
            continue
        params[key] = value
    return merge_params(params)


def _sample_record(sample: dict, rgb: np.ndarray, builtin: bool) -> dict:
    return {
        "id": sample["id"],
        "filename": sample["filename"],
        "short": sample.get("short") or "上传",
        "title": sample.get("title") or sample["filename"],
        "scene": sample.get("scene") or "本地上传",
        "scene_preset": sample.get("scene_preset"),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "builtin": builtin,
        "thumb_url": f"/api/samples/{sample['id']}/thumb",
        "image_url": f"/api/samples/{sample['id']}/image",
    }


def _restore_uploads() -> None:
    if not UPLOAD_DIR.exists():
        return
    for path in sorted(UPLOAD_DIR.glob("up_*.png")):
        image_id = path.stem
        _UPLOADS[image_id] = {
            "id": image_id,
            "filename": path.name,
            "short": "上传",
            "title": "本地图片",
            "scene": "本地上传",
            "scene_preset": None,
            "path": str(path),
        }


app = FastAPI(title="Photo2Vector Phase 1", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/samples")
def list_samples():
    records = []
    for sample in SAMPLES:
        rgb = _load_cached(sample["id"])
        records.append(_sample_record(sample, rgb, True))
    for uploaded in _UPLOADS.values():
        rgb = _load_cached(uploaded["id"])
        records.append(_sample_record(uploaded, rgb, False))
    return {"samples": records}


@app.get("/api/samples/{image_id}/image")
def sample_image(image_id: str):
    path = _resolve_path(image_id)
    media = "image/webp" if path.suffix.lower() == ".webp" else "image/jpeg"
    if path.suffix.lower() == ".png":
        media = "image/png"
    return FileResponse(path, media_type=media)


@app.get("/api/samples/{image_id}/thumb")
def sample_thumb(image_id: str):
    rgb = _load_cached(image_id)
    return Response(content=_thumb(rgb), media_type="image/png")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "空文件")
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(413, "图片超过 30MB")
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(400, "无法解码这张图片")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    rgb = resize_max(rgb, 2560)
    image_id = "up_" + hashlib.sha1(raw).hexdigest()[:12]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{image_id}.png"
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    record = {
        "id": image_id,
        "filename": file.filename or f"{image_id}.png",
        "short": "上传",
        "title": file.filename or "本地图片",
        "scene": "本地上传",
        "scene_preset": None,
        "path": str(path),
    }
    with _LOCK:
        _UPLOADS[image_id] = record
        _FILE_CACHE[image_id] = (path.stat().st_mtime, rgb)
    return _sample_record(record, rgb, False)


@app.get("/api/presets")
def list_presets():
    return {"presets": preset_payloads()}


@app.post("/api/phase1/diagnose")
def diagnose(body: DiagnoseBody):
    with _LOCK:
        rgb = _load_cached(body.image_id)
        token = _image_token(body.image_id, rgb)
        diag = get_diagnosis(rgb, token)
        public = public_diagnosis(diag)
    return {
        "image_id": body.image_id,
        "diagnosis": public,
        "recommended_params": public["recommended_params"],
    }


@app.post("/api/phase1/extract")
def extract(body: ExtractBody):
    with _LOCK:
        rgb = _load_cached(body.image_id)
        token = _image_token(body.image_id, rgb)
        diag = get_diagnosis(rgb, token)
        params = _resolve_extract_params(body, diag)
        result = run_phase1(rgb, params, cache=_STAGE_CACHE, token=token, diagnosis=diag)
        images = {
            "original": _png_url(result.keys["original"], result.original),
            "smoothed": _png_url(result.keys["smoothed"], result.smoothed),
            "energy": _png_url(result.keys["energy"], result.energy),
            "soft_ink": _png_url(result.keys["soft_ink"], result.soft),
            "final_line": _png_url(result.keys["final_line"], result.final),
            "final_rgba": _png_url(result.keys["final_rgba"], result.rgba),
            "w_tex": _png_url(result.keys["w_tex"], result.w_tex),
        }
        if result.single_line is not None:
            images["single_line"] = _png_url(result.keys["single_line"], result.single_line)
        if result.band_images:
            for name, array in result.band_images.items():
                images[name] = _png_url(result.keys[name], array)
    return {
        "image_id": body.image_id,
        "timings_ms": result.timings_ms,
        "cache": result.cache_hits,
        "stats": result.stats,
        "params": params,
        "diagnosis": result.diagnosis,
        "images": images,
    }


def _source_key(token: str, params: dict) -> str:
    blob = json.dumps({"token": token, "params": params}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:20]


def _vector_filename(image_id: str) -> str:
    for sample in SAMPLES:
        if sample["id"] == image_id:
            return f"{sample['short']}_vector_stage2.svg"
    return "上传_vector_stage2.svg"


def _illustration_filename(image_id: str) -> str:
    for sample in SAMPLES:
        if sample["id"] == image_id:
            return f"{sample['short']}_illustration.svg"
    return "上传_illustration.svg"


@app.post("/api/stage2/vectorize")
def vectorize_api(body: VectorizeBody):
    with _LOCK:
        rgb = _load_cached(body.image_id)
        token = _image_token(body.image_id, rgb)
        diag = get_diagnosis(rgb, token)
        params = _resolve_extract_params(
            ExtractBody(image_id=body.image_id, params=body.params, auto_mode=body.auto_mode),
            diag,
        )
        key = _source_key(token, params)
        found = _SOURCE_CACHE.get(key)
        phase_ms = 0.0
        if found is None:
            started = time.perf_counter()
            result = run_phase1(rgb, params, cache=_STAGE_CACHE, token=token, diagnosis=diag)
            phase_ms = (time.perf_counter() - started) * 1000.0
            if result.ink is None or result.oklab is None:
                raise HTTPException(500, "阶段一没有留下可矢量化的墨线")
            found = {
                "ink": result.ink,
                "oklab": result.oklab,
                "energy_norm": result.energy_norm,
                "w_tex": diag.get("w_tex_256"),
                "width": int(result.stats["width"]),
                "height": int(result.stats["height"]),
            }
            _SOURCE_CACHE[key] = found
            _SOURCE_CACHE.move_to_end(key)
            while len(_SOURCE_CACHE) > 4:
                _SOURCE_CACHE.popitem(last=False)
        else:
            _SOURCE_CACHE.move_to_end(key)
        out = run_vectorize(
            found["ink"],
            found["oklab"],
            body.stage2,
            (found["width"], found["height"]),
            source_key=key,
        )
        filename = _vector_filename(body.image_id)
        svg_blob = json.dumps({"src": key, "stage2": out["params"]}, sort_keys=True, default=str)
        svg_key = hashlib.sha1(svg_blob.encode("utf-8")).hexdigest()[:20]
        _SVG_CACHE[svg_key] = (filename, out["svg_text"])
        _SVG_CACHE.move_to_end(svg_key)
        while len(_SVG_CACHE) > 24:
            _SVG_CACHE.popitem(last=False)
    return {
        "image_id": body.image_id,
        "svg_text": out["svg_text"],
        "svg_url": f"/api/cache/{svg_key}.svg",
        "filename": filename,
        "vector_data": out["vector_data"],
        "vector_stats": out["vector_stats"],
        "timings_ms": {
            "stage2": out["vector_stats"]["stage2_ms"],
            "phase1": round(phase_ms, 2),
        },
        "cache": {"phase1": phase_ms == 0.0},
        "params": out["params"],
    }


@app.post("/api/stage3/illustrate")
def illustrate_api(body: IllustrateBody):
    with _LOCK:
        rgb = _load_cached(body.image_id)
        token = _image_token(body.image_id, rgb)
        diag = get_diagnosis(rgb, token)
        params = _resolve_extract_params(
            ExtractBody(image_id=body.image_id, params=body.params, auto_mode=body.auto_mode),
            diag,
        )
        key = _source_key(token, params)
        found = _SOURCE_CACHE.get(key)
        phase_ms = 0.0
        if found is None:
            started = time.perf_counter()
            result = run_phase1(rgb, params, cache=_STAGE_CACHE, token=token, diagnosis=diag)
            phase_ms = (time.perf_counter() - started) * 1000.0
            if result.ink is None or result.oklab is None:
                raise HTTPException(500, "阶段一没有留下可矢量化的墨线")
            found = {
                "ink": result.ink,
                "oklab": result.oklab,
                "energy_norm": result.energy_norm,
                "w_tex": diag.get("w_tex_256"),
                "width": int(result.stats["width"]),
                "height": int(result.stats["height"]),
            }
            _SOURCE_CACHE[key] = found
            _SOURCE_CACHE.move_to_end(key)
            while len(_SOURCE_CACHE) > 4:
                _SOURCE_CACHE.popitem(last=False)
        else:
            _SOURCE_CACHE.move_to_end(key)
        stage2_started = time.perf_counter()
        vector = run_vectorize(
            found["ink"],
            found["oklab"],
            body.stage2,
            (found["width"], found["height"]),
            source_key=key,
        )
        stage2_ms = (time.perf_counter() - stage2_started) * 1000.0
        paths_blob = json.dumps({"src": key, "stage2": vector["params"]}, sort_keys=True, default=str)
        paths_key = hashlib.sha1(paths_blob.encode("utf-8")).hexdigest()[:20]
        illustrated = run_illustrate(
            found["oklab"],
            vector["vector_data"]["paths"],
            body.stage3,
            display_size=(found["width"], found["height"]),
            source_key=key,
            paths_key=paths_key,
            w_tex=found.get("w_tex"),
            energy_norm=found.get("energy_norm"),
        )
        filename = _illustration_filename(body.image_id)
        svg_blob = json.dumps(
            {"src": key, "paths": paths_key, "stage3": illustrated["params"]},
            sort_keys=True,
            default=str,
        )
        svg_key = hashlib.sha1(svg_blob.encode("utf-8")).hexdigest()[:20]
        _SVG_CACHE[svg_key] = (filename, illustrated["svg_text"])
        _SVG_CACHE.move_to_end(svg_key)
        while len(_SVG_CACHE) > 24:
            _SVG_CACHE.popitem(last=False)
        preview_key = hashlib.sha1((svg_key + ":preview").encode("utf-8")).hexdigest()[:20]
        images = {
            "saliency_map": _png_url(preview_key + "-sal", illustrated["previews"]["saliency"]),
            "flat_regions": _png_url(preview_key + "-flat", illustrated["previews"]["flat"]),
            "routed_strokes": _png_url(preview_key + "-strokes", illustrated["previews"]["strokes"]),
        }
    return {
        "image_id": body.image_id,
        "svg_text": illustrated["svg_text"],
        "svg_url": f"/api/cache/{svg_key}.svg",
        "filename": filename,
        "images": images,
        "palette": illustrated["palette"],
        "illustration_stats": illustrated["illustration_stats"],
        "shapes": illustrated["shapes"],
        "size": illustrated["size"],
        "timings_ms": {
            "stage3": illustrated["illustration_stats"]["stage3_ms"],
            "stage2": round(stage2_ms, 2),
            "phase1": round(phase_ms, 2),
            **illustrated["timings_ms"],
        },
        "cache": {"phase1": phase_ms == 0.0, **illustrated["cache"]},
        "params": illustrated["params"],
    }


@app.get("/api/stage2/export_svg")
def export_svg(key: str):
    return cached_svg(key)


@app.get("/api/cache/{key}.svg")
def cached_svg(key: str):
    item = _SVG_CACHE.get(key)
    if item is None:
        raise HTTPException(404, "SVG 缓存已过期，请重新概括")
    filename, text = item
    disposition = f"attachment; filename=\"vector_stage2.svg\"; filename*=UTF-8''{quote(filename)}"
    return Response(
        content=text.encode("utf-8"),
        media_type="image/svg+xml",
        headers={"Content-Disposition": disposition, "Cache-Control": "no-cache"},
    )


@app.get("/api/cache/{key}.png")
def cached_png(key: str):
    data = _PNG_CACHE.get(key)
    if data is None:
        raise HTTPException(404, "缓存已过期，请重新提取")
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def _mount_frontend() -> None:
    if not DIST_DIR.exists():
        return
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="web")


_restore_uploads()
_mount_frontend()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
