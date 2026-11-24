import { forwardRef, useEffect, useImperativeHandle, useRef } from "react"
import paper from "paper"
import type { IllustrationShape, IllustrationStats, LineColorMode, Stage3Params, VectorAnchor } from "./types"
import type { MicroscopeHandle } from "./stage2"

export const DEFAULT_STAGE3: Stage3Params = {
  palette_k: 14,
  bg_simplify: 0.095,
  fg_detail: 0.032,
  shape_fit_tol: 2.8,
  solid_line_gate: 0.28,
  line_color_mode: "colored",
  show_bg_shapes: true,
  show_fg_shapes: true,
  show_strokes: true,
  dust_scale: 1,
  albedo_group: 0.4,
  shade_threshold: 0.13,
}

export function illustrationLine(stats: IllustrationStats) {
  const kept = Math.round(stats.retained_stroke_ratio * 100)
  return `色块 ${stats.shape_count.bg + stats.shape_count.fg} · 锚点 ${stats.shape_anchors} · 实线 ${kept}% · 覆盖 ${Math.round(stats.coverage_ratio * 100)}%`
}

export function paintSvg(svg: string, params: Pick<Stage3Params, "show_bg_shapes" | "show_fg_shapes" | "show_strokes" | "line_color_mode">) {
  const strokes = params.show_strokes && params.line_color_mode !== "none"
  return svg
    .replace(/(<g id="bg-shapes"[^>]*?)display="(?:inline|none)"/, `$1display="${params.show_bg_shapes ? "inline" : "none"}"`)
    .replace(/(<g id="fg-base-silhouettes"[^>]*?)display="(?:inline|none)"/, `$1display="${params.show_fg_shapes ? "inline" : "none"}"`)
    .replace(/(<g id="fg-detail-shapes"[^>]*?)display="(?:inline|none)"/, `$1display="${params.show_fg_shapes ? "inline" : "none"}"`)
    .replace(/(<g id="strokes-secondary"[^>]*?)display="(?:inline|none)"/, `$1display="${strokes ? "inline" : "none"}"`)
    .replace(/(<g id="strokes-primary"[^>]*?)display="(?:inline|none)"/, `$1display="${strokes ? "inline" : "none"}"`)
}

export const ShapeMicroscope = forwardRef<
  MicroscopeHandle,
  { shapes: IllustrationShape[]; width: number; height: number }
>(function ShapeMicroscope({ shapes, width, height }, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const scopeRef = useRef<paper.PaperScope | null>(null)
  const anchorsRef = useRef<{ x: number; y: number; kind: string }[]>([])
  const hiRef = useRef<paper.Layer | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.width = width
    canvas.height = height
    const scope = new paper.PaperScope()
    scope.setup(canvas)
    scope.activate()
    scope.view.viewSize = new scope.Size(width, height)
    scope.project.clear()
    const anchors: { x: number; y: number; kind: string }[] = []
    for (const shape of shapes) {
      drawShape(scope, shape)
      for (const anchor of shape.anchors) {
        drawHandle(scope, anchor)
        anchors.push({ x: anchor.p[0], y: anchor.p[1], kind: anchor.kind })
      }
    }
    const highlight = new scope.Layer()
    scopeRef.current = scope
    anchorsRef.current = anchors
    hiRef.current = highlight
    scope.view.update()
    return () => {
      scope.project.clear()
      scopeRef.current = null
    }
  }, [shapes, width, height])

  useImperativeHandle(ref, () => ({
    highlight(x: number, y: number, label: HTMLElement | null) {
      const scope = scopeRef.current
      const layer = hiRef.current
      if (!scope || !layer) return
      scope.activate()
      layer.activate()
      layer.removeChildren()
      let best: { x: number; y: number; kind: string } | null = null
      let bestD = 14
      for (const anchor of anchorsRef.current) {
        const dist = Math.hypot(anchor.x - x, anchor.y - y)
        if (dist < bestD) {
          best = anchor
          bestD = dist
        }
      }
      if (!best) {
        if (label) label.textContent = "悬停色块锚点可看类型与坐标"
        scope.view.update()
        return
      }
      const ring = new scope.Path.Circle(new scope.Point(best.x, best.y), 7)
      ring.strokeColor = new scope.Color("#ffe08a")
      ring.strokeWidth = 1.2
      if (label) {
        const name = best.kind === "cusp" ? "尖角" : best.kind === "smooth" ? "平滑锚点" : "端点"
        label.textContent = `${name} · ${best.x.toFixed(1)}, ${best.y.toFixed(1)}`
      }
      scope.view.update()
    },
  }))

  return <canvas ref={canvasRef} />
})

function drawShape(scope: paper.PaperScope, shape: IllustrationShape) {
  const path = new scope.Path()
  path.fillColor = new scope.Color(shape.fill)
  path.opacity = 0.92
  path.strokeColor = new scope.Color("#14343f")
  path.strokeWidth = 0.6
  let started = false
  for (const cmd of shape.commands) {
    if (cmd.op === "M" && cmd.p) {
      path.moveTo(new scope.Point(cmd.p[0], cmd.p[1]))
      started = true
    } else if (cmd.op === "L" && cmd.p && started) {
      path.lineTo(new scope.Point(cmd.p[0], cmd.p[1]))
    } else if (cmd.op === "C" && cmd.p && cmd.h1 && cmd.h2 && started) {
      path.cubicCurveTo(
        new scope.Point(cmd.h1[0], cmd.h1[1]),
        new scope.Point(cmd.h2[0], cmd.h2[1]),
        new scope.Point(cmd.p[0], cmd.p[1]),
      )
    } else if (cmd.op === "Z") {
      path.closePath()
    }
  }
}

function drawHandle(scope: paper.PaperScope, anchor: VectorAnchor) {
  const point = new scope.Point(anchor.p[0], anchor.p[1])
  const arm = (tip?: [number, number]) => {
    if (!tip) return
    const end = new scope.Point(tip[0], tip[1])
    const line = new scope.Path.Line(point, end)
    line.strokeColor = new scope.Color("#7ee7ff")
    line.strokeWidth = 0.7
    const knob = new scope.Path.Circle(end, 1.6)
    knob.fillColor = new scope.Color("#bff6ff")
  }
  if (anchor.kind !== "endpoint") {
    arm(anchor.h_in)
    arm(anchor.h_out)
  }
  if (anchor.kind === "cusp") {
    const box = new scope.Path.Rectangle(new scope.Rectangle(point.x - 2.4, point.y - 2.4, 4.8, 4.8))
    box.fillColor = new scope.Color("#ffb347")
    return
  }
  const dot = new scope.Path.Circle(point, 2.2)
  dot.fillColor = new scope.Color(anchor.kind === "smooth" ? "#3ee0ff" : "#e7edf2")
}

export function Stage3Panel({
  params,
  palette,
  stats,
  onChange,
  onBack,
}: {
  params: Stage3Params
  palette: { hex: string; ratio: number }[]
  stats: IllustrationStats | null
  onChange: (partial: Partial<Stage3Params>) => void
  onBack: () => void
}) {
  return (
    <>
      <section className="group">
        <h3>矢量插画成稿</h3>
        <p className="hint">形态学平涂后再按 L0 剪影、L1 固有色、L2 点睛与单层暗面递归取块。图层开关只改显示，不重新计算。</p>
        <div className="palette-row" title="当前插画调色盘">
          {palette.length === 0 && <span className="hint">色卡会在成稿完成后出现</span>}
          {palette.map((swatch) => (
            <i
              key={`${swatch.hex}-${swatch.ratio}`}
              className="palette-chip"
              style={{ background: swatch.hex }}
              title={`${swatch.hex} · ${Math.round(swatch.ratio * 100)}%`}
            />
          ))}
        </div>
        <label className="field" title="同色系深色线跟着衬底走。炭黑是经典线稿。无描边只留叠纸色块。">
          <span className="field-top">
            <b>线稿配色风格</b>
          </span>
          <select
            value={params.line_color_mode}
            onChange={(event) => onChange({ line_color_mode: event.target.value as LineColorMode })}
          >
            <option value="colored">同色系深色线 [推荐]</option>
            <option value="black">经典炭黑线</option>
            <option value="none">纯色块无描边</option>
          </select>
        </label>
        <Range
          label="蒙尘拔除力度"
          hint="P0 面积开闭的力度。越大，水泥麻点、反光白点和网点拔得越干净。"
          min={0.3}
          max={2.2}
          step={0.05}
          value={params.dust_scale}
          onChange={(value) => onChange({ dust_scale: value })}
        />
        <Range
          label="抗光影聚拢度"
          hint="越高越无视明暗，把受光和阴影收成同一块固有色。默认 0.40，对应明度权重 0.20。"
          min={0.1}
          max={0.5}
          step={0.01}
          value={params.albedo_group}
          onChange={(value) => onChange({ albedo_group: value })}
        />
        <Range
          label="块内二分阴影门槛"
          hint="L2 在每个大部件里最多切一块暗面。拉到最大就变成纯平涂、不再分阴影。"
          min={0.06}
          max={0.3}
          step={0.01}
          value={params.shade_threshold}
          onChange={(value) => onChange({ shade_threshold: value })}
        />
        <Range
          label="调色盘色数"
          hint="主体加权归纳出的插画色卡数量。"
          min={6}
          max={22}
          step={1}
          value={params.palette_k}
          onChange={(value) => onChange({ palette_k: value })}
        />
        <Range
          label="背景概括力度"
          hint="越大，背景越少、越大块。砖墙、天空和海面会收成几块干净平涂。"
          min={0.04}
          max={0.18}
          step={0.005}
          value={params.bg_simplify}
          onChange={(value) => onChange({ bg_simplify: value })}
        />
        <Range
          label="主体部件精细度"
          hint="越小，主体里的固有色部件越多。脸、衣服、手臂会各自保持完整，而不是按光影切开。"
          min={0.012}
          max={0.07}
          step={0.002}
          value={params.fg_detail}
          onChange={(value) => onChange({ fg_detail: value })}
        />
        <Range
          label="色块曲线概括容差"
          hint="贝塞尔拟合容差。越大，每个色块的锚点越少、边缘越利落。"
          min={0.8}
          max={6}
          step={0.1}
          value={params.shape_fit_tol}
          onChange={(value) => onChange({ shape_fit_tol: value })}
        />
        <Range
          label="虚边退线力度"
          hint="实度低于这里的虚边会撤掉，交给平涂色块自己显形。"
          min={0}
          max={0.75}
          step={0.01}
          value={params.solid_line_gate}
          onChange={(value) => onChange({ solid_line_gate: value })}
        />
      </section>
      {stats && (
        <p className="stats">
          背景块 {stats.shape_count.bg} · 主体块 {stats.shape_count.fg} · 保留实线 {stats.stroke_paths} · 主线{" "}
          {stats.primary_strokes}
        </p>
      )}
      <button className="back-stage" type="button" onClick={onBack}>
        返回阶段二微调线稿
      </button>
    </>
  )
}

function Range({
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
  const text = step >= 1 ? String(Math.round(value)) : step >= 0.1 ? value.toFixed(1) : value.toFixed(3)
  return (
    <label className="field" title={hint}>
      <span className="field-top">
        <b>{label}</b>
        <span className="val">{text}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  )
}
