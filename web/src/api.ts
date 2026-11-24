import type {
  Diagnosis,
  ExtractResult,
  IllustrationResult,
  Phase1Params,
  Preset,
  Sample,
  Stage2Params,
  Stage3Params,
  VectorizeResult,
} from "./types"

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    if (typeof body.detail === "string") return body.detail
  } catch {
    /* 非 JSON 错误页 */
  }
  return `请求失败 ${response.status}`
}

export async function fetchSamples(): Promise<Sample[]> {
  const response = await fetch("/api/samples")
  if (!response.ok) throw new Error(await readError(response))
  const body = (await response.json()) as { samples: Sample[] }
  return body.samples
}

export async function fetchPresets(): Promise<Preset[]> {
  const response = await fetch("/api/presets")
  if (!response.ok) throw new Error(await readError(response))
  const body = (await response.json()) as { presets: Preset[] }
  return body.presets
}

export async function uploadImage(file: File): Promise<Sample> {
  const form = new FormData()
  form.append("file", file)
  const response = await fetch("/api/upload", { method: "POST", body: form })
  if (!response.ok) throw new Error(await readError(response))
  return (await response.json()) as Sample
}

export async function fetchDiagnosis(imageId: string, signal?: AbortSignal): Promise<Diagnosis & { recommended_params: Phase1Params }> {
  const response = await fetch("/api/phase1/diagnose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: imageId }),
    signal,
  })
  if (!response.ok) throw new Error(await readError(response))
  const body = (await response.json()) as {
    diagnosis: Diagnosis
    recommended_params: Phase1Params
  }
  return { ...body.diagnosis, recommended_params: body.recommended_params }
}

export async function extractLines(
  imageId: string,
  params: Partial<Phase1Params> & { preset?: string },
  options?: { autoMode?: boolean; signal?: AbortSignal },
): Promise<ExtractResult> {
  const response = await fetch("/api/phase1/extract", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_id: imageId,
      params,
      auto_mode: Boolean(options?.autoMode),
    }),
    signal: options?.signal,
  })
  if (!response.ok) throw new Error(await readError(response))
  return (await response.json()) as ExtractResult
}

export async function vectorizeImage(
  imageId: string,
  phase1: Partial<Phase1Params>,
  stage2: Stage2Params,
  signal?: AbortSignal,
): Promise<VectorizeResult> {
  const response = await fetch("/api/stage2/vectorize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_id: imageId,
      params: phase1,
      auto_mode: false,
      stage2,
    }),
    signal,
  })
  if (!response.ok) throw new Error(await readError(response))
  return (await response.json()) as VectorizeResult
}

export async function illustrateImage(
  imageId: string,
  phase1: Partial<Phase1Params>,
  stage2: Stage2Params,
  stage3: Stage3Params,
  signal?: AbortSignal,
): Promise<IllustrationResult> {
  const response = await fetch("/api/stage3/illustrate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_id: imageId,
      params: phase1,
      auto_mode: false,
      stage2,
      stage3,
    }),
    signal,
  })
  if (!response.ok) throw new Error(await readError(response))
  return (await response.json()) as IllustrationResult
}
