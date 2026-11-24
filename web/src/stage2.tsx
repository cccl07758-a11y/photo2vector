import { forwardRef, useEffect, useImperativeHandle, useRef } from "react"
import paper from "paper"
import type { Stage2Params, VectorAnchor, VectorPath, VectorStats } from "./types"

export const DEFAULT_STAGE2: Stage2Params = {
  fit_tol: 1.8,
  line_snap_scale: 1,
  gap_max_px: 22,
  detail_keep: 0.7,
  min_stroke_len: 12,
  stroke_scale: 1,
  cusp_angle_deg: 42,
  max_turn_deg: 45,
  highlight_collapse: true,
  render_style: "center",
  show_secondary: true,
}

export function statsLine(stats: VectorStats) {
  const straight = Math.round(stats.straight_ratio * 100)
  return `路径 ${stats.total_paths} · 锚点 ${stats.total_anchors} (均 ${stats.avg_anchors_per_path.toFixed(1)}/条) · 直线 ${straight}% · 焊口 ${stats.welded_count}`
}

export type MicroscopeHandle = {
  highlight: (x: number, y: number, label: HTMLElement | null) => void
}

type DrawAnchor = { x: number; y: number; kind: string }

export const Microscope = forwardRef<
  MicroscopeHandle,
  { paths: VectorPath[]; width: number; height: number; clear?: boolean }
>(function Microscope({ paths, width, height, clear }, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const scopeRef = useRef<paper.PaperScope | null>(null)
  const anchorsRef = useRef<DrawAnchor[]>([])
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
    if (!clear) {
      const board = new scope.Path.Rectangle(new scope.Rectangle(0, 0, width, height))
      board.fillColor = new scope.Color("#f6f3ec")
    }
    const anchors: DrawAnchor[] = []
    for (const item of paths) {
      drawGeometry(scope, item)
    }
    for (const item of paths) {
      for (const bridge of item.welded_bridges) drawWeld(scope, bridge)
    }
    for (const item of paths) {
      for (const anchor of item.anchors) {
        drawAnchor(scope, anchor)
        anchors.push({ x: anchor.p[0], y: anchor.p[1], kind: anchor.kind })
      }
    }
    const highlight = new scope.Layer()
    highlight.name = "hover"
    scopeRef.current = scope
    anchorsRef.current = anchors
    hiRef.current = highlight
    scope.view.update()
    return () => {
      scope.project.clear()
      scopeRef.current = null
    }
  }, [paths, width, height, clear])

  useImperativeHandle(ref, () => ({
    highlight(x: number, y: number, label: HTMLElement | null) {
      const scope = scopeRef.current
      const layer = hiRef.current
      if (!scope || !layer) return
      scope.activate()
      layer.activate()
      layer.removeChildren()
      let best: DrawAnchor | null = null
      let bestD = 12
      for (const anchor of anchorsRef.current) {
        const dx = anchor.x - x
        const dy = anchor.y - y
        const dist = Math.hypot(dx, dy)
        if (dist < bestD) {
          best = anchor
          bestD = dist
        }
      }
      if (!best) {
        if (label) label.textContent = "悬停锚点可检视类型与坐标"
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

function drawGeometry(scope: paper.PaperScope, item: VectorPath) {
  let cursor: paper.Point | null = null
  let start: paper.Point | null = null
  for (const cmd of item.commands) {
    if (cmd.op === "M" && cmd.p) {
      cursor = new scope.Point(cmd.p[0], cmd.p[1])
      start = cursor
      continue
    }
    if (cmd.op === "L" && cursor && cmd.p) {
      const end = new scope.Point(cmd.p[0], cmd.p[1])
      const line = new scope.Path.Line(cursor, end)
      line.strokeColor = new scope.Color("#12a56a")
      line.strokeWidth = 1.45
      line.strokeCap = "round"
      cursor = end
      continue
    }
    if (cmd.op === "C" && cursor && cmd.p && cmd.h1 && cmd.h2) {
      const curve = new scope.Path()
      curve.strokeColor = new scope.Color("#14343f")
      curve.strokeWidth = 1.45
      curve.strokeCap = "round"
      curve.moveTo(cursor)
      curve.cubicCurveTo(
        new scope.Point(cmd.h1[0], cmd.h1[1]),
        new scope.Point(cmd.h2[0], cmd.h2[1]),
        new scope.Point(cmd.p[0], cmd.p[1]),
      )
      cursor = new scope.Point(cmd.p[0], cmd.p[1])
      continue
    }
    if (cmd.op === "Z" && cursor && start) {
      const line = new scope.Path.Line(cursor, start)
      line.strokeColor = new scope.Color("#12a56a")
      line.strokeWidth = 1.45
      cursor = start
    }
  }
}

function drawWeld(
  scope: paper.PaperScope,
  bridge: { p0: [number, number]; h1: [number, number]; h2: [number, number]; p1: [number, number] },
) {
  const weld = new scope.Path()
  weld.strokeColor = new scope.Color("#ff2bd6")
  weld.strokeWidth = 1.8
  weld.dashArray = [5, 3.5]
  weld.moveTo(new scope.Point(bridge.p0[0], bridge.p0[1]))
  weld.cubicCurveTo(
    new scope.Point(bridge.h1[0], bridge.h1[1]),
    new scope.Point(bridge.h2[0], bridge.h2[1]),
    new scope.Point(bridge.p1[0], bridge.p1[1]),
  )
}

function drawAnchor(scope: paper.PaperScope, anchor: VectorAnchor) {
  const point = new scope.Point(anchor.p[0], anchor.p[1])
  const handle = (tip: [number, number] | undefined) => {
    if (!tip) return
    const end = new scope.Point(tip[0], tip[1])
    const arm = new scope.Path.Line(point, end)
    arm.strokeColor = new scope.Color("#7ee7ff")
    arm.strokeWidth = 0.7
    const knob = new scope.Path.Circle(end, 1.7)
    knob.fillColor = new scope.Color("#bff6ff")
    knob.strokeColor = new scope.Color("#1aa8c7")
    knob.strokeWidth = 0.6
  }
  if (anchor.kind === "smooth" || anchor.kind === "cusp") {
    handle(anchor.h_in)
    handle(anchor.h_out)
  }
  if (anchor.kind === "cusp") {
    const box = new scope.Path.Rectangle(new scope.Rectangle(point.x - 2.5, point.y - 2.5, 5, 5))
    box.fillColor = new scope.Color("#ffb347")
    box.strokeColor = new scope.Color("#ff8a00")
    box.strokeWidth = 0.8
    return
  }
  const dot = new scope.Path.Circle(point, anchor.kind === "smooth" ? 2.35 : 2.15)
  dot.fillColor = new scope.Color(anchor.kind === "smooth" ? "#3ee0ff" : "#e7edf2")
  dot.strokeColor = new scope.Color(anchor.kind === "smooth" ? "#0b8eab" : "#66717c")
  dot.strokeWidth = 0.7
}

export function SvgFrame({ svg }: { svg: string }) {
  return <div className="svg-stage" dangerouslySetInnerHTML={{ __html: svg }} />
}

export function VectorWipe({
  width,
  height,
  left,
  svg,
  amount,
  onAmount,
}: {
  width: number
  height: number
  left: string
  svg: string
  amount: number
  onAmount: (value: number) => void
}) {
  return (
    <>
      <SvgFrame svg={svg} />
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
            onAmount(Math.min(0.98, Math.max(0.02, (ev.clientX - rect.left) / rect.width)))
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

export function Stage2Panel({
  params,
  onChange,
  onBack,
  stats,
}: {
  params: Stage2Params
  onChange: (partial: Partial<Stage2Params>) => void
  onBack: () => void
  stats: VectorStats | null
}) {
  return (
    <>
      <section className="group">
        <h3>矢量概括与几何规则化</h3>
        <p className="hint">后端写出权威直线和贝塞尔。这里的滑块只重算阶段二，不重跑提线。</p>
        <label className="field" title="钢笔中心线适合再进 Illustrator 编辑。笔锋包络用同一条中心线生成首尾收尖的填充轮廓。">
          <span className="field-top">
            <b>线条渲染风格</b>
          </span>
          <select
            value={params.render_style}
            onChange={(event) => onChange({ render_style: event.target.value as Stage2Params["render_style"] })}
          >
            <option value="center">钢笔分级中心线 (单线可编辑)</option>
            <option value="taper">手绘可变笔锋包络</option>
          </select>
        </label>
        <Range
          label="曲线概括容差"
          hint="拉大后，相邻的小弧会被合成更少的锚点，线条更利落。"
          min={0.6}
          max={5}
          step={0.1}
          value={params.fit_tol}
          onChange={(value) => onChange({ fit_tol: value })}
        />
        <Range
          label="直线拉直灵敏度"
          hint="越大，越接近直线的笔画会被强制写成 L，不再留软弯。"
          min={0.4}
          max={2.2}
          step={0.05}
          value={params.line_snap_scale}
          onChange={(value) => onChange({ line_snap_scale: value })}
        />
        <Range
          label="断口焊接半径"
          hint="0 关闭焊接。只补沿切向对齐的断口，平行线不会被连成 U 形。"
          min={0}
          max={36}
          step={1}
          value={params.gap_max_px}
          onChange={(value) => onChange({ gap_max_px: value })}
        />
        <Range
          label="二级细节保留度"
          hint="降低后，内部细碎折痕会从成稿里滤掉，只留大轮廓。"
          min={0}
          max={1}
          step={0.01}
          value={params.detail_keep}
          onChange={(value) => onChange({ detail_keep: value })}
        />
        <label className="check" title="二级线是色差和跨度都较弱的内部结构。关掉后只画一级主轮廓。">
          <b>显示二级细节线</b>
          <input
            type="checkbox"
            checked={params.show_secondary}
            onChange={(event) => onChange({ show_secondary: event.target.checked })}
          />
        </label>
        <Range
          label="最短笔画过滤"
          hint="短于这个长度、又没有接到主干上的碎线会被丢掉。"
          min={4}
          max={40}
          step={1}
          value={params.min_stroke_len}
          onChange={(value) => onChange({ min_stroke_len: value })}
        />
        <Range
          label="线宽缩放系数"
          hint="三档插画线宽再乘上这个系数。1 是默认的 1.4 / 2.2 / 3.2。"
          min={0.5}
          max={2.4}
          step={0.05}
          value={params.stroke_scale}
          onChange={(value) => onChange({ stroke_scale: value })}
        />
      </section>
      <details>
        <summary>高级几何</summary>
        <Range
          label="尖角阈值"
          hint="转角超过这里就硬切成尖角锚点，左右手柄不再共线。"
          min={28}
          max={70}
          step={1}
          value={params.cusp_angle_deg}
          onChange={(value) => onChange({ cusp_angle_deg: value })}
        />
        <Range
          label="跨交点最大转角"
          hint="小于这个转角的两支会被接成同一条长笔画。更大的岔口保持 T 形。"
          min={20}
          max={70}
          step={1}
          value={params.max_turn_deg}
          onChange={(value) => onChange({ max_turn_deg: value })}
        />
        <label className="check" title="只合并中间确实更亮的窄高光双脊。剑刃和镜框这种真双边会留下。">
          <b>合并高光伪双线</b>
          <input
            type="checkbox"
            checked={params.highlight_collapse}
            onChange={(event) => onChange({ highlight_collapse: event.target.checked })}
          />
        </label>
      </details>
      {stats && (
        <p className="stats">
          主轮廓 {stats.primary_paths} · 二级 {stats.secondary_paths} · 尖角 {stats.cusp_anchors} · 高光双线{" "}
          {stats.collapsed_double_lines}
        </p>
      )}
      <button className="back-stage" type="button" onClick={onBack}>
        返回阶段一微调底稿
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
  const text = step >= 1 ? String(Math.round(value)) : step >= 0.1 ? value.toFixed(1) : value.toFixed(2)
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
