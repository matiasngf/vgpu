export interface OrbitInput {
  /** Camera yaw/pitch in radians, advanced by `advance()` and by pointer drags. */
  readonly yaw: number;
  readonly pitch: number;
  advance(deltaTime: number): void;
  dispose(): void;
}

const DRIFT_SPEED = 0.09;
const DRAG_SENSITIVITY = 0.006;

/** Drag to look around the environment; the camera drifts on its own while idle. */
export function installOrbitInput(canvas: HTMLCanvasElement, yaw = 0.6, pitch = 0.12): OrbitInput {
  let currentYaw = yaw;
  let currentPitch = pitch;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const previousTouchAction = canvas.style.touchAction;
  canvas.style.touchAction = 'none';
  const parent = canvas.parentElement;
  const previousParentPosition = parent?.style.position ?? '';
  let changedParentPosition = false;

  const hint = document.createElement('div');
  hint.textContent = 'drag to look around';
  Object.assign(hint.style, {
    position: 'absolute', left: '50%', bottom: '18px', transform: 'translateX(-50%)',
    color: 'rgba(255,255,255,.8)', font: '500 12px system-ui', letterSpacing: '.08em',
    textTransform: 'uppercase', pointerEvents: 'none', transition: 'opacity 400ms', zIndex: '2',
  });
  if (parent) {
    if (getComputedStyle(parent).position === 'static') {
      parent.style.position = 'relative';
      changedParentPosition = true;
    }
    parent.append(hint);
  }

  const down = (event: PointerEvent) => {
    if (!event.isPrimary) return;
    canvas.setPointerCapture?.(event.pointerId);
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    hint.style.opacity = '0';
  };
  const move = (event: PointerEvent) => {
    if (!dragging || !event.isPrimary) return;
    currentYaw -= (event.clientX - lastX) * DRAG_SENSITIVITY;
    currentPitch = Math.max(-1.2, Math.min(1.2, currentPitch + (event.clientY - lastY) * DRAG_SENSITIVITY));
    lastX = event.clientX;
    lastY = event.clientY;
  };
  const up = (event: PointerEvent) => { if (event.isPrimary) dragging = false; };

  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('pointerleave', up);

  return {
    get yaw() { return currentYaw; },
    get pitch() { return currentPitch; },
    advance(deltaTime: number) {
      if (!dragging) currentYaw += deltaTime * DRIFT_SPEED;
    },
    dispose() {
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', up);
      canvas.removeEventListener('pointercancel', up);
      canvas.removeEventListener('pointerleave', up);
      hint.remove();
      canvas.style.touchAction = previousTouchAction;
      if (changedParentPosition && parent) parent.style.position = previousParentPosition;
    },
  };
}
