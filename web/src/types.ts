export type TextureMethod = "none" | "bilateral" | "rgf" | "l0" | "rtv"
export type ExtractMode = "ps_dodge" | "ridge" | "fdog" | "hybrid"
export type ThresholdMode = "hysteresis" | "hard" | "soft"
export type ViewMode = "final" | "matrix" | "color" | "wipe" | "overlay" | "wtex"
export type InkTint = "black" | "magenta" | "cyan"
export type WipeSource = "original" | "single" | "ps"
export type BandBlendMode = "weighted_max" | "soft_sum" | "single"
export type BandInk = "gated" | "raw"

export interface Phase1Params {
  texture_method: TextureMethod
  texture_scale: number
  chroma_weight: number
  sigma_g: number
  sigma_norm: number
  quietness: number
  contrast_norm: number
  extract_mode: ExtractMode
  line_width: number
  coh_power: number
  ao_suppression: number
  flow_coherence: number
  fdog_sigma: number
  fdog_rho: number
  fdog_eps: number
  fdog_phi: number
  eta: number
  etf_iters: number
  etf_radius: number
  threshold_mode: ThresholdMode
  thresh_low: number
  thresh_high: number
  min_component_size: number
  min_span: number
  thin_1px: boolean
  max_side: number
  spatial_adaptive: boolean
  spatial_strength: number
  band_blend_mode: BandBlendMode
  shadow_weight: number
  mid_weight: number
  highlight_weight: number
  shadow_push: number
  highlight_pull: number
}

export interface Sample {
  id: string
  filename: string
  short: string
  title: string
  scene: string
  scene_preset: string | null
  width: number
  height: number
  builtin: boolean
  thumb_url: string
  image_url: string
}

export interface Preset {
  id: string
  label: string
  badge: string
  description: string
  params: Phase1Params
}

export interface DiagnosisScores {
  periodicity: number
  roughness: number
  hdr_unevenness: number
  soft_ao: number
  chroma_edge: number
  noise_sigma: number
  diagnose_ms: number
}

export interface Diagnosis {
  strategy_label: string
  strategy_badge: string
  reasons: string[]
  scores: DiagnosisScores
  diagnose_ms: number
  texture_method: string
}

export interface ExtractImages {
  original: string
  smoothed: string
  energy: string
  soft_ink: string
  final_line: string
  final_rgba: string
  w_tex: string
  single_line?: string
  shadow_exposure?: string
  mid_exposure?: string
  highlight_exposure?: string
  shadow_line?: string
  mid_line?: string
  highlight_line?: string
  shadow_raw_line?: string
  mid_raw_line?: string
  highlight_raw_line?: string
  color_bands?: string
}

export type StageMode = 1 | 2 | 3
export type Stage3View = "svg" | "wipe" | "quad" | "micro"
export type LineColorMode = "colored" | "black" | "none"
export type Stage2View = "svg" | "micro" | "wipe" | "overlay"
export type RenderStyle = "center" | "taper"
export type AnchorKind = "endpoint" | "smooth" | "cusp"

export interface Stage2Params {
  fit_tol: number
  line_snap_scale: number
  gap_max_px: number
  detail_keep: number
  min_stroke_len: number
  stroke_scale: number
  cusp_angle_deg: number
  max_turn_deg: number
  highlight_collapse: boolean
  render_style: RenderStyle
  show_secondary: boolean
}

export interface VectorCommand {
  op: "M" | "L" | "C" | "Z"
  p?: [number, number]
  h1?: [number, number]
  h2?: [number, number]
}

export interface VectorAnchor {
  p: [number, number]
  kind: AnchorKind
  h_in?: [number, number]
  h_out?: [number, number]
}

export interface WeldBridge {
  p0: [number, number]
  h1: [number, number]
  h2: [number, number]
  p1: [number, number]
  kind?: string
}

export interface VectorPath {
  id: number
  tier: "primary" | "secondary"
  score: number
  stroke_width: number
  closed: boolean
  commands: VectorCommand[]
  anchors: VectorAnchor[]
  welded_bridges: WeldBridge[]
  width_profile?: number[]
}

export interface VectorStats {
  total_paths: number
  primary_paths: number
  secondary_paths: number
  total_anchors: number
  cusp_anchors: number
  smooth_anchors: number
  avg_anchors_per_path: number
  line_segments: number
  cubic_segments: number
  straight_ratio: number
  welded_count: number
  collapsed_double_lines: number
  stage2_ms: number
}

export interface VectorizeResult {
  image_id: string
  svg_text: string
  svg_url: string
  filename: string
  vector_data: { width: number; height: number; paths: VectorPath[] }
  vector_stats: VectorStats
  timings_ms: { stage2: number; phase1: number }
  cache: { phase1: boolean }
  params: Stage2Params
}

export interface Stage3Params {
  palette_k: number
  bg_simplify: number
  fg_detail: number
  shape_fit_tol: number
  solid_line_gate: number
  line_color_mode: LineColorMode
  show_bg_shapes: boolean
  show_fg_shapes: boolean
  show_strokes: boolean
}

export interface PaletteSwatch {
  hex: string
  ratio: number
}

export interface IllustrationStats {
  shape_count: { bg: number; fg: number }
  palette_hex: string[]
  shape_anchors: number
  stroke_paths: number
  retained_stroke_ratio: number
  coverage_ratio: number
  shape_coverage?: number
  stage3_ms: number
  primary_strokes: number
  secondary_strokes: number
}

export interface IllustrationShape {
  id: number
  region: number
  group: "bg" | "base" | "detail"
  fill: string
  z: number
  commands: VectorCommand[]
  anchors: VectorAnchor[]
}

export interface IllustrationResult {
  image_id: string
  svg_text: string
  svg_url: string
  filename: string
  images: {
    saliency_map: string
    flat_regions: string
    routed_strokes: string
  }
  palette: PaletteSwatch[]
  illustration_stats: IllustrationStats
  shapes: IllustrationShape[]
  size: { width: number; height: number }
  timings_ms: { stage3: number; stage2: number; phase1: number }
  cache: { phase1: boolean; saliency?: boolean; routing?: boolean; regions?: boolean; geometry?: boolean }
  params: Stage3Params
}

export interface ExtractResult {
  image_id: string
  timings_ms: {
    texture: number
    tensor: number
    extract: number
    post: number
    total: number
    diagnose?: number
    bands?: number
    compose?: number
  }
  cache: {
    texture: boolean
    tensor: boolean
    soft_ink: boolean
    bands?: boolean
  }
  stats: {
    width: number
    height: number
    components: number
    line_ratio: number
    soft_p50: number
    soft_p99: number
  }
  params: Phase1Params
  diagnosis?: Diagnosis
  images: ExtractImages
}
