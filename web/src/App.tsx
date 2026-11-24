import { useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { extractLines, fetchPresets, fetchSamples, illustrateImage, uploadImage, vectorizeImage } from "./api"
import { DEFAULT_STAGE2, Microscope, Stage2Panel, SvgFrame, VectorWipe, statsLine, type MicroscopeHandle } from "./stage2"
import { DEFAULT_STAGE3, ShapeMicroscope, Stage3Panel, illustrationLine, paintSvg } from "./stage3"
import type {
  BandInk,
  Diagnosis,
  ExtractResult,
  IllustrationResult,
  InkTint,
  Phase1Params,
  Preset,
  Sample,
  Stage2Params,
  Stage2View,
  Stage3Params,
  Stage3View,
  StageMode,
  VectorizeResult,
  ViewMode,
  WipeSource,
} from "./types"
const VIEWS: { id: ViewMode; label: string }[] = [
  { id: "final", label: "最终线稿" },
  { id: "matrix", label: "三层通道矩阵" },
  { id: "color", label: "三色分带叠看" },
  { id: "wipe", label: "卷帘对比" },
  { id: "overlay", label: "原图叠底" },
  { id: "wtex", label: "纹理蒙版" },
]

const EYES_ON = { shadow: true, mid: true, highlight: true }

const SCORE_ROWS: { key: keyof Diagnosis["scores"]; label: string }[] = [
  { key: "periodicity", label: "周期网格" },
  { key: "roughness", label: "笔触肌理" },
  { key: "hdr_unevenness", label: "光比不均" },
  { key: "soft_ao", label: "柔和软影" },
  { key: "chroma_edge", label: "撞色边界" },
]

const TINTS: { id: InkTint; label: string; css: string }[] = [
  { id: "black", label: "黑", css: "#111111" },
  { id: "magenta", label: "品红", css: "#ff4ad8" },
  { id: "cyan", label: "青", css: "#3ee0ff" },
]

function format(value: number, step: number) {
  if (step >= 1) return String(Math.round(value))
  if (step >= 0.1) return value.toFixed(1)
  return value.toFixed(2)
}

export default function App() {
  const [samples, setSamples] = useState<Sample[]>([])
  const [presets, setPresets] = useState<Preset[]>([])
  const [imageId, setImageId] = useState<string>("")
  const [params, setParams] = useState<Phase1Params | null>(null)
  const [activePreset, setActivePreset] = useState("auto")
  const [mode, setMode] = useState<"auto" | "manual">("auto")
  const [fromAuto, setFromAuto] = useState(false)
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null)
  const [result, setResult] = useState<ExtractResult | null>(null)
  const [psLine, setPsLine] = useState<string>("")
  const [view, setView] = useState<ViewMode>("final")
  const [stage, setStage] = useState<StageMode>(3)
  const [view2, setView2] = useState<Stage2View>("micro")
  const [stage2, setStage2] = useState<Stage2Params>(DEFAULT_STAGE2)
  const [vector, setVector] = useState<VectorizeResult | null>(null)
  const [pending2, setPending2] = useState(false)
  const [view3, setView3] = useState<Stage3View>("svg")
  const [stage3, setStage3] = useState<Stage3Params>(DEFAULT_STAGE3)
  const [illustration, setIllustration] = useState<IllustrationResult | null>(null)
  const [pending3, setPending3] = useState(false)
  const [wipeBitmap, setWipeBitmap] = useState<"line" | "photo">("line")
  const [wipe, setWipe] = useState(0.5)
  const [wipeSource, setWipeSource] = useState<WipeSource>("original")
  const [underOpacity, setUnderOpacity] = useState(0.45)
  const [eyes, setEyes] = useState(EYES_ON)
  const [bandInk, setBandInk] = useState<BandInk>("gated")
  const [colorOnPhoto, setColorOnPhoto] = useState(false)
  const [colorMix, setColorMix] = useState(0.92)
  const [tint, setTint] = useState<InkTint>("magenta")
  const [transform, setTransform] = useState({ x: 40, y: 40, s: 1 })
  const [pending, setPending] = useState(false)
  const [error, setError] = useState("")
  const [dropping, setDropping] = useState(false)
  const [advanced, setAdvanced] = useState(false)
  const viewportRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null)
  const microRef = useRef<MicroscopeHandle>(null)
  const probeRef = useRef<HTMLSpanElement>(null)
  const psCache = useRef<Map<string, string>>(new Map())
  const heavyStamp = useRef("")

  const sample = samples.find((item) => item.id === imageId) ?? null
  const effectiveParams = useMemo(() => {
    if (!params) return null
    return {
      ...params,
      shadow_weight: eyes.shadow ? params.shadow_weight : 0,
      mid_weight: eyes.mid ? params.mid_weight : 0,
      highlight_weight: eyes.highlight ? params.highlight_weight : 0,
    }
  }, [params, eyes])

  useEffect(() => {
    let cancelled = false
    Promise.all([fetchSamples(), fetchPresets()])
      .then(([nextSamples, nextPresets]) => {
        if (cancelled) return
        setSamples(nextSamples)
        setPresets(nextPresets)
        if (nextSamples[0]) setImageId(nextSamples[0].id)
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "后端未连接")
      })
    return () => {
      cancelled = true
    }
  }, [])

  const requestKey =
    mode === "auto" ? `auto:${imageId}` : `${imageId}:${effectiveParams ? JSON.stringify(effectiveParams) : ""}`

  useEffect(() => {
    if (!imageId) return
    if (mode !== "auto" && !effectiveParams) return
    const heavy = mode === "auto" ? `auto:${imageId}` : heavySignature(effectiveParams as Phase1Params)
    const delay = heavyStamp.current === heavy ? 16 : 110
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      if (mode !== "auto") heavyStamp.current = heavy
      setPending(true)
      const sending = mode === "auto" ? { preset: "auto" } : (effectiveParams as Phase1Params)
      extractLines(imageId, sending, { autoMode: mode === "auto", signal: controller.signal })
        .then((payload) => {
          setResult(payload)
          if (payload.diagnosis) setDiagnosis(payload.diagnosis)
          if (mode === "auto") {
            setParams(payload.params)
            heavyStamp.current = heavySignature(payload.params)
          }
          setError("")
        })
        .catch((reason: unknown) => {
          if (reason instanceof DOMException && reason.name === "AbortError") return
          setError(reason instanceof Error ? reason.message : "提取失败")
        })
        .finally(() => {
          if (!controller.signal.aborted) setPending(false)
        })
    }, delay)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [requestKey, mode, effectiveParams, imageId])

  useEffect(() => {
    if (!imageId || presets.length === 0) return
    const cached = psCache.current.get(imageId)
    if (cached) {
      setPsLine(cached)
      return
    }
    const preset = presets.find((item) => item.id === "ps_baseline")
    if (!preset) return
    let cancelled = false
    extractLines(imageId, preset.params)
      .then((payload) => {
        psCache.current.set(imageId, payload.images.final_line)
        if (!cancelled) setPsLine(payload.images.final_line)
      })
      .catch(() => {
        if (!cancelled) setPsLine("")
      })
    return () => {
      cancelled = true
    }
  }, [imageId, presets])

  const phaseKey = result ? JSON.stringify(result.params) : ""
  const lineKey = result?.images.final_line ?? ""
  const stageKey = JSON.stringify(stage2)

  useEffect(() => {
    if (stage !== 2 || !imageId || !result || result.image_id !== imageId) return
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setPending2(true)
      vectorizeImage(imageId, result.params, stage2, controller.signal)
        .then((payload) => {
          setVector(payload)
          setError("")
        })
        .catch((reason: unknown) => {
          if (reason instanceof DOMException && reason.name === "AbortError") return
          setError(reason instanceof Error ? reason.message : "矢量化失败")
        })
        .finally(() => {
          if (!controller.signal.aborted) setPending2(false)
        })
    }, 80)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [stage, imageId, phaseKey, stageKey, lineKey])

  const stage3Key = JSON.stringify(stage3)

  useEffect(() => {
    if (stage !== 3 || !imageId || !result || result.image_id !== imageId) return
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setPending3(true)
      illustrateImage(imageId, result.params, stage2, stage3, controller.signal)
        .then((payload) => {
          setIllustration(payload)
          setError("")
        })
        .catch((reason: unknown) => {
          if (reason instanceof DOMException && reason.name === "AbortError") return
          setError(reason instanceof Error ? reason.message : "插画生成失败")
        })
        .finally(() => {
          if (!controller.signal.aborted) setPending3(false)
        })
    }, 90)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [stage, imageId, phaseKey, stageKey, stage3Key, lineKey, result, stage2, stage3])

  useEffect(() => {
    const node = viewportRef.current
    if (!node || !result) return
    const pad = 48
    const frame =
      stage === 3
        ? stage3Box(view3, result.stats.width, result.stats.height)
        : stage === 2
          ? { width: result.stats.width, height: result.stats.height }
          : stageBox(view, result.stats.width, result.stats.height)
    const scale = Math.min((node.clientWidth - pad) / frame.width, (node.clientHeight - pad) / frame.height, 1)
    setTransform({
      s: scale,
      x: (node.clientWidth - frame.width * scale) / 2,
      y: (node.clientHeight - frame.height * scale) / 2,
    })
  }, [imageId, view, view3, stage, result?.stats.width, result?.stats.height])

  useEffect(() => {
    const node = viewportRef.current
    if (!node) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const rect = node.getBoundingClientRect()
      const mx = event.clientX - rect.left
      const my = event.clientY - rect.top
      setTransform((current) => {
        const next = Math.min(6, Math.max(0.25, current.s * Math.exp(-event.deltaY * 0.0012)))
        const k = next / current.s
        return { s: next, x: mx - (mx - current.x) * k, y: my - (my - current.y) * k }
      })
    }
    node.addEventListener("wheel", onWheel, { passive: false })
    return () => node.removeEventListener("wheel", onWheel)
  }, [])

  useEffect(() => {
    setEyes((current) => (current.shadow && current.mid && current.highlight ? current : EYES_ON))
  }, [imageId])

  function setEye(band: keyof typeof EYES_ON, open: boolean) {
    setFromAuto((current) => current || mode === "auto" || activePreset === "auto")
    setMode("manual")
    setActivePreset("")
    setEyes((current) => ({ ...current, [band]: open }))
  }

  function patch(partial: Partial<Phase1Params>) {
    setFromAuto((current) => current || mode === "auto" || activePreset === "auto")
    setMode("manual")
    setActivePreset("")
    setParams((current) => (current ? { ...current, ...partial } : current))
  }

  function applyPreset(preset: Preset) {
    setEyes(EYES_ON)
    if (preset.id === "auto") {
      setFromAuto(false)
      setMode("auto")
      setActivePreset("auto")
      return
    }
    setFromAuto(false)
    setMode("manual")
    setActivePreset(preset.id)
    setParams(preset.params)
  }

  async function takeFile(file: File | undefined) {
    if (!file) return
    try {
      const created = await uploadImage(file)
      setSamples((current) => [created, ...current.filter((item) => item.id !== created.id)])
      setImageId(created.id)
      setError("")
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "上传失败")
    }
  }

  function exportSvg() {
    if (!vector) return
    const blob = new Blob([vector.svg_text], { type: "image/svg+xml" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = vector.filename
    link.click()
    URL.revokeObjectURL(url)
  }

  function exportIllustration() {
    if (!illustration) return
    const blob = new Blob([paintSvg(illustration.svg_text, stage3)], { type: "image/svg+xml" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = illustration.filename
    link.click()
    URL.revokeObjectURL(url)
  }

  function exportPng() {
    if (!result) return
    const link = document.createElement("a")
    link.href = result.images.final_line
    link.download = `${sample?.short ?? "line"}_phase1.png`
    link.click()
  }

  const content = useMemo(() => {
    if (!result) return null
    const width = result.stats.width
    const height = result.stats.height
    if (stage === 3) {
      const svg = illustration ? paintSvg(illustration.svg_text, stage3) : ""
      const shapes = illustration?.shapes ?? []
      if (view3 === "wipe") {
        return (
          <div className="stage" style={{ width, height }}>
            {svg ? <VectorWipe width={width} height={height} left={result.images.original} svg={svg} amount={wipe} onAmount={setWipe} /> : null}
          </div>
        )
      }
      if (view3 === "quad" && illustration) {
        const cells = [
          ["① 无AI主体分区场 S_fg", illustration.images.saliency_map],
          ["② 以线筑坝色块分割", illustration.images.flat_regions],
          ["③ 虚实分流深色线", illustration.images.routed_strokes],
        ] as const
        return (
          <div className="grid2" style={{ width: width * 2 + 8, height: height * 2 + 8 }}>
            {cells.map(([title, src]) => (
              <div className="cell" key={title} style={{ width, height }}>
                <img src={src} alt={title} draggable={false} />
                <span>{title}</span>
              </div>
            ))}
            <div className="cell" style={{ width, height }}>
              {svg ? <SvgFrame svg={svg} /> : null}
              <span>④ 最终矢量插画成稿</span>
            </div>
          </div>
        )
      }
      if (view3 === "micro") {
        return (
          <div className="stage" style={{ width, height }}>
            <ShapeMicroscope ref={microRef} shapes={shapes} width={width} height={height} />
          </div>
        )
      }
      return (
        <div className="stage" style={{ width, height }}>
          {svg ? <SvgFrame svg={svg} /> : null}
        </div>
      )
    }
    if (stage === 2) {
      const paths = vector?.vector_data.paths ?? []
      const svg = vector?.svg_text ?? ""
      if (view2 === "svg") {
        return (
          <div className="stage" style={{ width, height }}>
            {svg ? <SvgFrame svg={svg} /> : null}
          </div>
        )
      }
      if (view2 === "wipe") {
        const left = wipeBitmap === "photo" ? result.images.original : result.images.final_line
        return (
          <div className="stage" style={{ width, height }}>
            {svg ? <VectorWipe width={width} height={height} left={left} svg={svg} amount={wipe} onAmount={setWipe} /> : null}
          </div>
        )
      }
      if (view2 === "overlay") {
        return (
          <div className="stage" style={{ width, height }}>
            <img className="under" src={result.images.original} alt="" draggable={false} style={{ opacity: 0.5 }} />
            <Microscope ref={microRef} paths={paths} width={width} height={height} clear />
          </div>
        )
      }
      return (
        <div className="stage" style={{ width, height }}>
          <Microscope ref={microRef} paths={paths} width={width} height={height} />
        </div>
      )
    }
    const images = result.images
    if (view === "matrix") {
      const line = (gated?: string, raw?: string) => (bandInk === "raw" ? raw : gated) || ""
      const cells = [
        ["暗部显影 · 去纹理", images.shadow_exposure],
        ["中调 · 去纹理", images.mid_exposure],
        ["高光压暗 · 去纹理", images.highlight_exposure],
        ["暗部层线稿", line(images.shadow_line, images.shadow_raw_line)],
        ["中调层线稿", line(images.mid_line, images.mid_raw_line)],
        ["高光层线稿", line(images.highlight_line, images.highlight_raw_line)],
      ] as const
      if (!images.shadow_exposure) {
        return (
          <div className="stage" style={{ width, height }}>
            <img src={images.final_line} alt="单层线稿" draggable={false} />
          </div>
        )
      }
      return (
        <div className="grid3" style={{ width: width * 3 + 16, height: height * 2 + 8 }}>
          {cells.map(([title, src]) => (
            <div className="cell" key={title} style={{ width, height }}>
              {src ? <img src={src} alt={title} draggable={false} /> : null}
              <span>{title}</span>
            </div>
          ))}
        </div>
      )
    }
    const wipeLeft =
      wipeSource === "ps" && psLine
        ? psLine
        : wipeSource === "single" && images.single_line
          ? images.single_line
          : images.original
    return (
      <div className="stage" style={{ width, height }}>
        {view === "final" && <img src={images.final_line} alt="最终线稿" draggable={false} />}
        {view === "wtex" && <img src={images.w_tex} alt="纹理蒙版" draggable={false} />}
        {view === "color" && images.color_bands && (
          <>
            {colorOnPhoto && <img className="under" src={images.original} alt="" draggable={false} />}
            <img
              src={images.color_bands}
              alt="三色分带"
              draggable={false}
              style={colorOnPhoto ? { mixBlendMode: "multiply", opacity: colorMix } : undefined}
            />
          </>
        )}
        {view === "wipe" && (
          <Wipe width={width} height={height} right={images.final_line} left={wipeLeft} amount={wipe} onAmount={setWipe} />
        )}
        {view === "overlay" && (
          <>
            <img className="under" src={images.original} alt="" draggable={false} style={{ opacity: underOpacity }} />
            <Tint src={images.final_rgba} color={TINTS.find((item) => item.id === tint)?.css ?? "#111"} />
          </>
        )}
      </div>
    )
  }, [result, view, wipe, wipeSource, psLine, underOpacity, tint, bandInk, colorOnPhoto, colorMix, stage, view2, view3, vector, illustration, stage3, wipeBitmap])

  return (
    <div
      className="app"
      onDragOver={(event) => {
        event.preventDefault()
        setDropping(true)
      }}
      onDragLeave={() => setDropping(false)}
      onDrop={(event) => {
        event.preventDefault()
        setDropping(false)
        void takeFile(event.dataTransfer.files[0])
      }}
    >
      <header className="toolbar">
        <div className="brand">
          <strong>Photo2Vector</strong>
          <span>{stage === 1 ? "阶段一 · 线稿实验台" : stage === 2 ? "阶段二 · 矢量线稿" : "阶段三 · 矢量插画"}</span>
        </div>
        <div className="stage-switch" role="tablist">
          <button type="button" className={stage === 1 ? "on" : ""} onClick={() => setStage(1)}>
            ① 位图提线
          </button>
          <button type="button" className={stage === 2 ? "on" : ""} onClick={() => setStage(2)}>
            ② 矢量线稿
          </button>
          <button type="button" className={stage === 3 ? "on featured" : "featured"} onClick={() => setStage(3)}>
            ③ 矢量插画 · SVG
          </button>
        </div>
        <div className="samples">
          {samples.map((item) => (
            <button
              key={item.id}
              className={item.id === imageId ? "thumb on" : "thumb"}
              title={item.scene}
              onClick={() => setImageId(item.id)}
              type="button"
            >
              <img src={item.thumb_url} alt={item.short} />
              <small>{item.short}</small>
            </button>
          ))}
          <label className="upload" title="拖入或点选本地图片">
            <small>＋ 上传</small>
            <input
              type="file"
              accept="image/*"
              onChange={(event) => {
                void takeFile(event.target.files?.[0])
                event.target.value = ""
              }}
            />
          </label>
        </div>
        <div className="presets">
          {presets.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className={activePreset === preset.id ? "pill on" : "pill"}
              title={preset.description}
              onClick={() => applyPreset(preset)}
            >
              {preset.label}
              <b>{preset.badge}</b>
            </button>
          ))}
        </div>
        <div className="meta">
          <span className="badge">
            {result ? `${result.stats.width}×${result.stats.height}` : "—"}
          </span>
          <span className="badge">
            {stage === 1 ? "Total" : stage === 2 ? "矢量" : "插画"}{" "}
            <em>
              {stage === 3
                ? illustration
                  ? `${Math.round(illustration.timings_ms.stage3)} ms`
                  : "—"
                : stage === 2
                  ? vector
                    ? `${Math.round(vector.timings_ms.stage2)} ms`
                    : "—"
                  : result
                    ? `${Math.round(result.timings_ms.total)} ms`
                    : "—"}
            </em>
          </span>
          {stage === 2 && vector && <span className="badge vector-stats">{statsLine(vector.vector_stats)}</span>}
          {stage === 3 && illustration && <span className="badge vector-stats">{illustrationLine(illustration.illustration_stats)}</span>}
          {stage === 3 ? (
            <button className="export vector" type="button" disabled={!illustration} onClick={exportIllustration}>
              ⬇ 导出分层插画 .svg
            </button>
          ) : stage === 2 ? (
            <button className="export vector" type="button" disabled={!vector} onClick={exportSvg}>
              ⬇ 导出可编辑 .svg
            </button>
          ) : (
            <button className="export" type="button" disabled={!result} onClick={exportPng}>
              导出当前线稿 PNG
            </button>
          )}
        </div>
      </header>

      <div
        className={dropping ? "viewport dropping" : "viewport"}
        ref={viewportRef}
        onPointerDown={(event) => {
          if ((event.target as HTMLElement).dataset.handle === "wipe") return
          dragRef.current = { x: event.clientX, y: event.clientY, ox: transform.x, oy: transform.y }
          ;(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId)
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current
          if (drag) {
            setTransform((current) => ({
              ...current,
              x: drag.ox + event.clientX - drag.x,
              y: drag.oy + event.clientY - drag.y,
            }))
            return
          }
          if (stage === 3 && view3 !== "micro") return
          if (stage === 2 && view2 !== "micro" && view2 !== "overlay") return
          if (stage === 1) return
          const node = viewportRef.current
          if (!node) return
          const rect = node.getBoundingClientRect()
          const x = (event.clientX - rect.left - transform.x) / transform.s
          const y = (event.clientY - rect.top - transform.y) / transform.s
          microRef.current?.highlight(x, y, probeRef.current)
        }}
        onPointerUp={() => {
          dragRef.current = null
        }}
        onDoubleClick={() => {
          if (!result || !viewportRef.current) return
          const node = viewportRef.current
          const frame =
            stage === 3
              ? stage3Box(view3, result.stats.width, result.stats.height)
              : stage === 2
                ? { width: result.stats.width, height: result.stats.height }
                : stageBox(view, result.stats.width, result.stats.height)
          const scale = Math.min((node.clientWidth - 48) / frame.width, (node.clientHeight - 48) / frame.height, 1)
          setTransform({
            s: scale,
            x: (node.clientWidth - frame.width * scale) / 2,
            y: (node.clientHeight - frame.height * scale) / 2,
          })
        }}
      >
        {(pending || pending2 || pending3) && <div className="busy" />}
        {error && <div className="error">{error}</div>}
        <div className="float top">
          {stage === 3
            ? (
                [
                  ["svg", "插画成稿 (SVG)"],
                  ["wipe", "原图 vs 插画卷帘"],
                  ["quad", "叠纸解剖四宫格"],
                  ["micro", "色块锚点显微镜 (Paper.js)"],
                ] as const
              ).map(([id, label]) => (
                <button key={id} type="button" className={view3 === id ? "on" : ""} onClick={() => setView3(id)}>
                  {label}
                </button>
              ))
            : stage === 2
            ? (
                [
                  ["svg", "矢量成稿 (SVG)"],
                  ["micro", "锚点显微镜 (Paper.js)"],
                  ["wipe", "位图 vs 矢量卷帘"],
                  ["overlay", "原图叠底矢量"],
                ] as const
              ).map(([id, label]) => (
                <button key={id} type="button" className={view2 === id ? "on" : ""} onClick={() => setView2(id)}>
                  {label}
                </button>
              ))
            : VIEWS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={view === item.id ? "on" : ""}
              onClick={() => setView(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
        {stage === 3 && view3 === "svg" && (
          <div className="float bottom">
            <button type="button" className={stage3.show_bg_shapes ? "on" : ""} onClick={() => setStage3((current) => ({ ...current, show_bg_shapes: !current.show_bg_shapes }))}>
              👁 背景色块
            </button>
            <button type="button" className={stage3.show_fg_shapes ? "on" : ""} onClick={() => setStage3((current) => ({ ...current, show_fg_shapes: !current.show_fg_shapes }))}>
              👁 主体色块
            </button>
            <button type="button" className={stage3.show_strokes ? "on" : ""} onClick={() => setStage3((current) => ({ ...current, show_strokes: !current.show_strokes }))}>
              👁 插画线稿
            </button>
          </div>
        )}
        {stage === 3 && view3 === "micro" && (
          <div className="float bottom">
            <span><i className="legend-swatch curve" />色块轮廓</span>
            <span><i className="legend-swatch smooth" />平滑</span>
            <span><i className="legend-swatch cusp" />尖角</span>
            <span ref={probeRef}>悬停色块锚点可看类型与坐标</span>
          </div>
        )}
        {stage === 2 && (view2 === "micro" || view2 === "overlay") && (
          <div className="float bottom">
            <span><i className="legend-swatch line" />直线</span>
            <span><i className="legend-swatch curve" />曲线</span>
            <span><i className="legend-swatch smooth" />平滑</span>
            <span><i className="legend-swatch cusp" />尖角</span>
            <span><i className="legend-swatch weld" />焊口</span>
            <span ref={probeRef}>悬停锚点可检视类型与坐标</span>
          </div>
        )}
        {stage === 2 && view2 === "wipe" && (
          <div className="float bottom">
            <button type="button" className={wipeBitmap === "line" ? "on" : ""} onClick={() => setWipeBitmap("line")}>
              左侧位图线稿
            </button>
            <button type="button" className={wipeBitmap === "photo" ? "on" : ""} onClick={() => setWipeBitmap("photo")}>
              左侧原图
            </button>
          </div>
        )}
        {stage === 2 && view2 === "svg" && (
          <div className="float bottom">
            <button type="button" className={stage2.render_style === "center" ? "on" : ""} onClick={() => setStage2((current) => ({ ...current, render_style: "center" }))}>
              钢笔中心线
            </button>
            <button type="button" className={stage2.render_style === "taper" ? "on" : ""} onClick={() => setStage2((current) => ({ ...current, render_style: "taper" }))}>
              手绘笔锋包络
            </button>
            <button type="button" className={stage2.show_secondary ? "on" : ""} onClick={() => setStage2((current) => ({ ...current, show_secondary: !current.show_secondary }))}>
              二级细节
            </button>
          </div>
        )}
        {stage === 1 && view === "wipe" && (
          <div className="float bottom">
            <button type="button" className={wipeSource === "original" ? "on" : ""} onClick={() => setWipeSource("original")}>
              左侧原图
            </button>
            <button type="button" className={wipeSource === "single" ? "on" : ""} onClick={() => setWipeSource("single")}>
              左侧单层线稿
            </button>
            <button type="button" className={wipeSource === "ps" ? "on" : ""} onClick={() => setWipeSource("ps")}>
              左侧 PS 基线
            </button>
          </div>
        )}
        {stage === 1 && view === "matrix" && (
          <div className="float bottom">
            <button type="button" className={bandInk === "gated" ? "on" : ""} onClick={() => setBandInk("gated")}>
              本亮度带选通后
            </button>
            <button type="button" className={bandInk === "raw" ? "on" : ""} onClick={() => setBandInk("raw")}>
              全图重曝原始线稿
            </button>
          </div>
        )}
        {stage === 1 && view === "color" && (
          <div className="float bottom">
            <span>
              <i className="legend-dot" style={{ background: "#1d4ed8" }} />
              暗部
            </span>
            <span>
              <i className="legend-dot" style={{ background: "#18181b" }} />
              中调
            </span>
            <span>
              <i className="legend-dot" style={{ background: "#db2777" }} />
              高光
            </span>
            <button type="button" className={colorOnPhoto ? "on" : ""} onClick={() => setColorOnPhoto((value) => !value)}>
              叠在原图上
            </button>
            {colorOnPhoto && (
              <input
                type="range"
                min={0.35}
                max={1}
                step={0.02}
                value={colorMix}
                onChange={(event) => setColorMix(Number(event.target.value))}
              />
            )}
          </div>
        )}
        {stage === 1 && view === "wtex" && (
          <div className="float bottom">
            <span>暗色保护主体 · 亮色抹平纹理</span>
          </div>
        )}
        {stage === 1 && view === "overlay" && (
          <div className="float bottom">
            <span>原图 {Math.round(underOpacity * 100)}%</span>
            <input
              type="range"
              min={0}
              max={0.8}
              step={0.02}
              value={underOpacity}
              onChange={(event) => setUnderOpacity(Number(event.target.value))}
            />
            {TINTS.map((item) => (
              <button key={item.id} type="button" className={tint === item.id ? "chip on" : "chip"} onClick={() => setTint(item.id)}>
                {item.label}
              </button>
            ))}
          </div>
        )}
        <div className="zoom">{Math.round(transform.s * 100)}%</div>
        {content && (
          <div
            className="stage-wrap"
            style={{ transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.s})` }}
          >
            {content}
          </div>
        )}
      </div>

      <aside className="panel">
        {sample && <p className="hint">{sample.scene}</p>}
        {stage === 3 ? (
          <Stage3Panel
            params={stage3}
            palette={illustration?.palette ?? []}
            stats={illustration?.illustration_stats ?? null}
            onChange={(partial) => setStage3((current) => ({ ...current, ...partial }))}
            onBack={() => setStage(2)}
          />
        ) : stage === 2 ? (
          <Stage2Panel
            params={stage2}
            stats={vector?.vector_stats ?? null}
            onChange={(partial) => setStage2((current) => ({ ...current, ...partial }))}
            onBack={() => setStage(1)}
          />
        ) : (
          <>
        {diagnosis && <DiagnosisCard diagnosis={diagnosis} manual={mode === "manual" && fromAuto} />}
        {params && (
          <>
            <Section title="① 去纹理保结构">
              <label className="check" title="纹理区走重度去纹理，五官、文字和硬边留在轻度基底上。">
                <b>分区自适应保护主体</b>
                <input
                  type="checkbox"
                  checked={params.spatial_adaptive}
                  onChange={(event) => patch({ spatial_adaptive: event.target.checked })}
                />
              </label>
              <Slider
                label="主体保护强度"
                hint="0 退回整图去纹理。接近 1 时，蒙版暗处几乎不抹，亮处仍然重度压纹理。"
                min={0}
                max={1}
                step={0.01}
                value={params.spatial_strength}
                onChange={(value) => patch({ spatial_strength: value })}
              />
              <Select
                label="方法"
                hint="滚动引导保留硬边；L0 烫平厚涂笔触；RTV 压周期网格、格纹和网点。"
                value={params.texture_method}
                options={[
                  ["none", "不滤波"],
                  ["bilateral", "双边（PS 基线）"],
                  ["rgf", "滚动引导 RGF"],
                  ["l0", "L0 梯度最小化"],
                  ["rtv", "相对总变分 RTV"],
                ]}
                onChange={(value) => patch({ texture_method: value as Phase1Params["texture_method"] })}
              />
              <Slider
                label="纹理尺度"
                hint="0 表示跳过。越大越烫平笔触、锁子甲和网点，过大也会伤到细结构。"
                min={0}
                max={10}
                step={0.1}
                value={params.texture_scale}
                onChange={(value) => patch({ texture_scale: value })}
              />
            </Section>
            <Section title="⚡ 多曝光亮度分层（暗 / 中 / 亮）">
              <Select
                label="合成方式"
                hint="软窗极值并集保住任一亮度层看清的边缘。概率互补是能量叠加。关闭分层则退回单层对照。"
                value={params.band_blend_mode}
                options={[
                  ["weighted_max", "软窗极值并集 (推荐)"],
                  ["soft_sum", "概率互补叠加"],
                  ["single", "关闭分层 (单层对照)"],
                ]}
                onChange={(value) => patch({ band_blend_mode: value })}
              />
              <BandRow
                color="#1d4ed8"
                open={eyes.shadow}
                onToggle={() => setEye("shadow", !eyes.shadow)}
                label="暗部层权重"
                hint="关掉等于权重 0。拖动只重合成缓存里的软墨线。"
                min={0}
                max={1.4}
                step={0.05}
                value={params.shadow_weight}
                onChange={(value) => patch({ shadow_weight: value })}
              >
                <Slider
                  label="暗部撑开幅度"
                  hint="把挤在暗部的盔甲和战靴色阶拉开，让去纹理不再把它们当成笔触抹平。"
                  min={0}
                  max={1}
                  step={0.01}
                  value={params.shadow_push}
                  onChange={(value) => patch({ shadow_push: value })}
                />
              </BandRow>
              <BandRow
                color="#18181b"
                open={eyes.mid}
                onToggle={() => setEye("mid", !eyes.mid)}
                label="中调层权重"
                hint="常规主体结构。权重 0 即关闭这一层。"
                min={0}
                max={1.4}
                step={0.05}
                value={params.mid_weight}
                onChange={(value) => patch({ mid_weight: value })}
              />
              <BandRow
                color="#db2777"
                open={eyes.highlight}
                onToggle={() => setEye("highlight", !eyes.highlight)}
                label="高光层权重"
                hint="受光手臂、白罩袍和帽顶靠这一层拉回来。"
                min={0}
                max={1.4}
                step={0.05}
                value={params.highlight_weight}
                onChange={(value) => patch({ highlight_weight: value })}
              >
                <Slider
                  label="高光压暗幅度"
                  hint="把挤在强光里的浅色轮廓拉开。越大，受光侧的弱边越明显。"
                  min={0}
                  max={1}
                  step={0.01}
                  value={params.highlight_pull}
                  onChange={(value) => patch({ highlight_pull: value })}
                />
              </BandRow>
            </Section>
            <Section title="② 感知色彩与亮暗归一化">
              <Slider
                label="局部对比度归一化"
                hint="拉起亮部白墨镜与暗部黑靴的弱轮廓。0 时退回全局能量，亮部容易丢线。"
                min={0}
                max={1}
                step={0.01}
                value={params.contrast_norm}
                onChange={(value) => patch({ contrast_norm: value })}
              />
              <Slider
                label="画面安静度"
                hint="只按死接近底噪的平坦天空和白底。不再拿全图最强边缘去砍受光侧的弱轮廓。"
                min={0}
                max={1}
                step={0.01}
                value={params.quietness}
                onChange={(value) => patch({ quietness: value })}
              />
            </Section>
            <Section title="③ 提线算子与流向连贯">
              <Select
                label="提线模式"
                hint="混合模式同时保住撞色单线和暗槽实心墨。PS 减淡是未归一化的对照基线。"
                value={params.extract_mode}
                options={[
                  ["hybrid", "混合（推荐）"],
                  ["ridge", "结构单脊线"],
                  ["fdog", "流向墨线 FDoG"],
                  ["ps_dodge", "PS 颜色减淡"],
                ]}
                onChange={(value) => patch({ extract_mode: value as Phase1Params["extract_mode"] })}
              />
              <Slider
                label="线宽尺度"
                hint="法向采样距离。宽缓软影要用稍大的尺度才能和真正的阶跃边分开。"
                min={0.6}
                max={6}
                step={0.1}
                value={params.line_width}
                onChange={(value) => patch({ line_width: value })}
              />
              <Slider
                label="软影 / AO 抑制"
                hint="过滤宽缓渐变阴影。柔和接触阴影上中心和两侧能量几乎一样，会被减成空白。"
                min={0}
                max={1}
                step={0.01}
                value={params.ao_suppression}
                onChange={(value) => patch({ ao_suppression: value })}
              />
              <Slider
                label="流向连贯"
                hint="沿边缘切向做 RK2 积分，把断开的弱线串起来，同时把东倒西歪的噪点平均掉。"
                min={0}
                max={1}
                step={0.01}
                value={params.flow_coherence}
                onChange={(value) => patch({ flow_coherence: value })}
              />
            </Section>
            <Section title="④ 双阈值连接与碎点清扫">
              <Select
                label="阈值"
                hint="滞后连接会保住连在主干上的暗线，孤立灰点直接丢掉。硬阈值是 PS 那种一刀切。"
                value={params.threshold_mode}
                options={[
                  ["hysteresis", "滞后连接"],
                  ["hard", "硬阈值"],
                  ["soft", "连续灰阶"],
                ]}
                onChange={(value) => patch({ threshold_mode: value as Phase1Params["threshold_mode"] })}
              />
              {params.threshold_mode !== "hard" && (
                <Slider
                  label="低阈值"
                  hint="连在高阈值主干上的弱线会被救回来。PS 基线的减淡比值没有归一化，这个数要低得多。"
                  min={0.01}
                  max={0.8}
                  step={0.01}
                  value={params.thresh_low}
                  onChange={(value) => patch({ thresh_low: value })}
                />
              )}
              <Slider
                label="高阈值"
                hint="单独就能留下的主干。抬高会更干净，弱轮廓会先消失。"
                min={0.02}
                max={0.9}
                step={0.01}
                value={params.thresh_high}
                onChange={(value) => patch({ thresh_high: value })}
              />
            </Section>
            <details open={advanced} onToggle={(event) => setAdvanced((event.target as HTMLDetailsElement).open)}>
              <summary>高级微调</summary>
              <Slider label="色度权重" hint="0 只看明度。1 附近让等明度撞色和明度边缘权重接近。" min={0} max={2} step={0.05} value={params.chroma_weight} onChange={(value) => patch({ chroma_weight: value })} />
              <Slider label="张量平滑 σ" hint="结构张量的空间尺度。稍大更连贯，过大会糊掉近距离双线。" min={0.4} max={4} step={0.05} value={params.sigma_g} onChange={(value) => patch({ sigma_g: value })} />
              <Slider label="归一化窗口" hint="局部能量标准差的模糊半径，大约 12 到 20 像素。" min={6} max={32} step={0.5} value={params.sigma_norm} onChange={(value) => patch({ sigma_norm: value })} />
              <Slider label="相干度次方" hint="压低杂乱交叉点。拐角的相干度本来就低，次方太高会把它们掐断。" min={0} max={3} step={0.05} value={params.coh_power} onChange={(value) => patch({ coh_power: value })} />
              <Slider label="ETF 邻域" hint="沿切向平滑方向场的半径。" min={1} max={4} step={1} value={params.etf_radius} onChange={(value) => patch({ etf_radius: value })} />
              <Slider label="ETF 迭代" hint="方向场传播次数。" min={1} max={4} step={1} value={params.etf_iters} onChange={(value) => patch({ etf_iters: value })} />
              <Slider label="对齐强度 η" hint="弱边缘切向顺从旁边强主干的程度。" min={0} max={6} step={0.1} value={params.eta} onChange={(value) => patch({ eta: value })} />
              <Slider label="FDoG σ" hint="法向差分高斯的中心尺度。" min={0.6} max={2.5} step={0.05} value={params.fdog_sigma} onChange={(value) => patch({ fdog_sigma: value })} />
              <Slider label="FDoG ρ" hint="外围高斯的权重，越接近 1 越只留细线。" min={0.9} max={0.99} step={0.005} value={params.fdog_rho} onChange={(value) => patch({ fdog_rho: value })} />
              <Slider label="FDoG ε" hint="XDoG 软阈值。更负则只留更重的暗线。" min={-0.5} max={0.05} step={0.01} value={params.fdog_eps} onChange={(value) => patch({ fdog_eps: value })} />
              <Slider label="FDoG φ" hint="软阈值的陡峭程度。" min={1} max={24} step={0.5} value={params.fdog_phi} onChange={(value) => patch({ fdog_phi: value })} />
              <Slider label="碎点面积" hint="面积和跨度都小于这里的孤立墨点会被抹掉。长细线因为跨度够，会留下。" min={0} max={200} step={1} value={params.min_component_size} onChange={(value) => patch({ min_component_size: value })} />
              <Slider label="碎点跨度" hint="包围盒对角线短于这个值，且面积也小，才清掉。" min={0} max={60} step={1} value={params.min_span} onChange={(value) => patch({ min_span: value })} />
              <label className="check" title="Zhang-Suen 单像素中心线，给下一阶段的曲线拟合做预览。">
                <b>1px 中心骨架预览</b>
                <input type="checkbox" checked={params.thin_1px} onChange={(event) => patch({ thin_1px: event.target.checked })} />
              </label>
            </details>
            {result && <Timings result={result} />}
          </>
        )}
          </>
        )}
      </aside>
    </div>
  )
}

function DiagnosisCard({ diagnosis, manual }: { diagnosis: Diagnosis; manual: boolean }) {
  return (
    <section className="diag">
      <div className="diag-head">
        <b>{diagnosis.strategy_badge}</b>
        <span>{Math.round(diagnosis.diagnose_ms)} ms</span>
      </div>
      <p className="diag-label">{diagnosis.strategy_label}</p>
      {manual && <p className="diag-note">手动微调（基于 Auto）</p>}
      <div className="meters">
        {SCORE_ROWS.map((row) => {
          const value = Number(diagnosis.scores[row.key] ?? 0)
          return (
            <div className="meter" key={row.key}>
              <span>{row.label}</span>
              <div className="bar">
                <i style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }} />
              </div>
              <em>{Math.round(value * 100)}</em>
            </div>
          )
        })}
      </div>
      <ul className="reasons">
        {diagnosis.reasons.map((reason) => (
          <li key={reason}>{reason}</li>
        ))}
      </ul>
    </section>
  )
}

function heavySignature(params: Phase1Params) {
  const rest: Partial<Phase1Params> = { ...params }
  delete rest.shadow_weight
  delete rest.mid_weight
  delete rest.highlight_weight
  delete rest.band_blend_mode
  return JSON.stringify(rest)
}

function stageBox(view: ViewMode, width: number, height: number) {
  if (view === "matrix") return { width: width * 3 + 16, height: height * 2 + 8 }
  return { width, height }
}

function stage3Box(view: Stage3View, width: number, height: number) {
  if (view === "quad") return { width: width * 2 + 8, height: height * 2 + 8 }
  return { width, height }
}

function BandRow({
  color,
  open,
  onToggle,
  label,
  hint,
  min,
  max,
  step,
  value,
  onChange,
  children,
}: {
  color: string
  open: boolean
  onToggle: () => void
  label: string
  hint: string
  min: number
  max: number
  step: number
  value: number
  onChange: (value: number) => void
  children?: ReactNode
}) {
  return (
    <div className={open ? "band-line" : "band-line off"}>
      <button type="button" className={open ? "eye on" : "eye"} title={open ? "关闭这一层" : "打开这一层"} onClick={onToggle}>
        {open ? "👁" : "–"}
      </button>
      <i className="swatch" style={{ background: color }} />
      <div>
        <Slider label={label} hint={hint} min={min} max={max} step={step} value={value} onChange={onChange} />
        {children}
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="group">
      <h3>{title}</h3>
      {children}
    </section>
  )
}

function Slider({
  label,
  hint,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string
  hint: string
  min: number
  max: number
  step: number
  value: number
  onChange: (value: number) => void
}) {
  return (
    <label className="field" title={hint}>
      <span className="field-top">
        <b>{label}</b>
        <span className="val">{format(value, step)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  )
}

function Select<T extends string>({
  label,
  hint,
  value,
  options,
  onChange,
}: {
  label: string
  hint: string
  value: T
  options: [T, string][]
  onChange: (value: T) => void
}) {
  return (
    <label className="field" title={hint}>
      <span className="field-top">
        <b>{label}</b>
      </span>
      <select value={value} onChange={(event) => onChange(event.target.value as T)}>
        {options.map(([id, name]) => (
          <option key={id} value={id}>
            {name}
          </option>
        ))}
      </select>
    </label>
  )
}

function Timings({ result }: { result: ExtractResult }) {
  const rows = [
    ["去纹理", result.timings_ms.texture, result.cache.texture],
    ["结构张量", result.timings_ms.tensor, result.cache.tensor],
    ["提线", result.timings_ms.extract, result.cache.soft_ink],
    ["净网", result.timings_ms.post, false],
    [
      "分层",
      result.cache.bands ? (result.timings_ms.compose ?? 0) : (result.timings_ms.bands ?? 0),
      false,
    ],
  ] as const
  const max = Math.max(...rows.map((row) => row[1]), 1)
  return (
    <div className="group">
      <h3>耗时</h3>
      <div className="times">
        {rows.map(([name, ms, cached]) => (
          <div className="time-row" key={name}>
            <span>{name}</span>
            <div className="bar">
              <i style={{ width: `${Math.max(4, (ms / max) * 100)}%` }} />
            </div>
            <span>{cached ? "cache" : `${Math.round(ms)} ms`}</span>
          </div>
        ))}
      </div>
      <p className="stats">
        线条 {(result.stats.line_ratio * 100).toFixed(1)}% · {result.stats.components} 条连通
      </p>
    </div>
  )
}

function Wipe({
  width,
  height,
  left,
  right,
  amount,
  onAmount,
}: {
  width: number
  height: number
  left: string
  right: string
  amount: number
  onAmount: (value: number) => void
}) {
  return (
    <>
      <img src={right} alt="阶段一线稿" draggable={false} />
      <div className="wipe-clip" style={{ width: `${amount * 100}%` }}>
        <img src={left} alt="" draggable={false} style={{ width, height }} />
      </div>
      <div
        className="wipe-bar"
        data-handle="wipe"
        style={{ left: `${amount * 100}%` }}
        onPointerDown={(event) => {
          event.stopPropagation()
          const parent = event.currentTarget.parentElement
          if (!parent) return
          const move = (ev: PointerEvent) => {
            const rect = parent.getBoundingClientRect()
            const next = (ev.clientX - rect.left) / rect.width
            onAmount(Math.min(0.98, Math.max(0.02, next)))
          }
          const up = () => {
            window.removeEventListener("pointermove", move)
            window.removeEventListener("pointerup", up)
          }
          window.addEventListener("pointermove", move)
          window.addEventListener("pointerup", up)
        }}
      />
    </>
  )
}

function Tint({ src, color }: { src: string; color: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const image = new Image()
    image.onload = () => {
      canvas.width = image.naturalWidth
      canvas.height = image.naturalHeight
      const ctx = canvas.getContext("2d")
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(image, 0, 0)
      ctx.globalCompositeOperation = "source-in"
      ctx.fillStyle = color
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
    image.src = src
  }, [src, color])
  return <canvas ref={ref} />
}
