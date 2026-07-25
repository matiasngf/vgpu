// Live controls: drag-to-orbit / wheel-to-zoom in place of drei's `<OrbitControls>`,
// plus the sun-rotation slider and auto-rotate toggle the experiment exposed through
// leva. Every element is created with `document.createElement`, so the runner smoke
// test can boot this against a stubbed DOM.

import { EARTH_TUNING, type OrbitState } from './planet';

export interface EarthControls {
  /** Current orbit state, damped toward the drag target. */
  step(deltaTime: number): OrbitState;
  /** Sun rotation in degrees; advances on its own while auto-rotate is on. */
  sunDegrees(deltaTime: number): number;
  dispose(): void;
}

const PANEL_STYLE =
  'position:absolute;top:12px;right:12px;display:flex;flex-direction:column;gap:6px;padding:8px 10px;' +
  'border:1px solid #24405f;border-radius:8px;background:rgba(6,14,26,.78);color:#d8efff;' +
  'font:12px system-ui;z-index:2;backdrop-filter:blur(6px)';

export function installControls(canvas: HTMLCanvasElement, initial: OrbitState): EarthControls {
  let yaw = initial.yaw;
  let pitch = initial.pitch;
  let radius = initial.radius;
  let targetYaw = yaw;
  let targetPitch = pitch;
  let targetRadius = radius;
  let sun = 0;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const previousTouchAction = canvas.style.touchAction;
  canvas.style.touchAction = 'none';

  const parent = canvas.parentElement;
  const previousParentPosition = parent?.style.position ?? '';
  let changedParentPosition = false;
  const panel = document.createElement('div');
  panel.style.cssText = PANEL_STYLE;
  const sunRow = document.createElement('label');
  sunRow.style.cssText = 'display:flex;align-items:center;gap:8px';
  const sunLabel = document.createElement('span');
  sunLabel.textContent = 'Sun';
  const sunSlider = document.createElement('input');
  sunSlider.type = 'range';
  sunSlider.min = '0';
  sunSlider.max = '360';
  sunSlider.step = '0.5';
  sunSlider.value = '0';
  sunSlider.style.width = '110px';
  sunRow.append(sunLabel, sunSlider);
  const autoRow = document.createElement('label');
  autoRow.style.cssText = 'display:flex;align-items:center;gap:8px';
  const autoBox = document.createElement('input');
  autoBox.type = 'checkbox';
  autoBox.checked = true;
  const autoLabel = document.createElement('span');
  autoLabel.textContent = 'Auto rotate';
  autoRow.append(autoBox, autoLabel);
  panel.append(sunRow, autoRow);
  if (parent) {
    if (getComputedStyle(parent).position === 'static') {
      parent.style.position = 'relative';
      changedParentPosition = true;
    }
    parent.append(panel);
  }

  const onSlider = () => {
    sun = Number(sunSlider.value) || 0;
    autoBox.checked = false;
  };
  sunSlider.addEventListener('input', onSlider);

  const down = (event: PointerEvent) => {
    if (!event.isPrimary) return;
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture?.(event.pointerId);
  };
  const move = (event: PointerEvent) => {
    if (!dragging || !event.isPrimary) return;
    const rect = canvas.getBoundingClientRect();
    targetYaw -= ((event.clientX - lastX) / Math.max(1, rect.width)) * Math.PI * 2;
    targetPitch += ((event.clientY - lastY) / Math.max(1, rect.height)) * Math.PI;
    targetPitch = Math.max(-1.45, Math.min(1.45, targetPitch));
    lastX = event.clientX;
    lastY = event.clientY;
  };
  const up = (event: PointerEvent) => {
    if (event.isPrimary) dragging = false;
  };
  const wheel = (event: WheelEvent) => {
    event.preventDefault();
    const { minRadius, maxRadius } = EARTH_TUNING.camera;
    targetRadius = Math.max(minRadius, Math.min(maxRadius, targetRadius * Math.exp(event.deltaY * 0.0012)));
  };
  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('wheel', wheel, { passive: false });

  return {
    step(deltaTime) {
      // Critically-damped-ish smoothing, frame-rate independent.
      const blend = 1 - Math.exp(-deltaTime * 9);
      yaw += (targetYaw - yaw) * blend;
      pitch += (targetPitch - pitch) * blend;
      radius += (targetRadius - radius) * blend;
      return { yaw, pitch, radius };
    },
    sunDegrees(deltaTime) {
      if (autoBox.checked) {
        sun = (sun + deltaTime * EARTH_TUNING.sun.degreesPerSecond) % 360;
        sunSlider.value = sun.toFixed(1);
      }
      return sun;
    },
    dispose() {
      sunSlider.removeEventListener('input', onSlider);
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', up);
      canvas.removeEventListener('pointercancel', up);
      canvas.removeEventListener('wheel', wheel);
      panel.remove();
      canvas.style.touchAction = previousTouchAction;
      if (changedParentPosition && parent) parent.style.position = previousParentPosition;
    },
  };
}
