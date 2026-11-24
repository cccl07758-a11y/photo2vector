import paper from 'paper'

export function createPaperScope(canvas: HTMLCanvasElement) {
  const scope = new paper.PaperScope()
  scope.setup(canvas)
  return scope
}

export function strokeCircle(
  scope: paper.PaperScope,
  center: paper.Point,
  radius: number,
) {
  const path = new scope.Path.Circle(center, radius)
  path.strokeColor = new scope.Color('#111111')
  return path
}
