import type { Effect, Frame, Gpu, Surface, Target } from 'vgpu';

import blurWgsl from './blur.wgsl';
import brightPassWgsl from './bright-pass.wgsl';
import postWgsl from './post.wgsl';
import sceneWgsl from './scene.wgsl';

type Output = Surface | Target;

interface ThumbOptions {
  warmupFrames?: number;
  dt?: number;
  /** Seconds into the day/night cycle to freeze. */
  time?: number;
}

export interface DreamcoreFrameOptions {
  /** 0 = night with the door glowing, 1 = full day. */
  phase: number;
  /** Seconds, only drives film grain. */
  time?: number;
  /** 1 sample per pixel for live rendering, 4 for stills. */
  samples?: 1 | 4;
  /** Exact blade shadow rays toward the door and sun; off for a cheaper live frame. */
  grassShadows?: boolean;
  /** Wind amplitude for the blades (0 for stills). */
  wind?: number;
}

interface Effects {
  scene: Effect;
  brightPass: Effect;
  blurH1: Effect;
  blurV1: Effect;
  blurH2: Effect;
  blurV2: Effect;
  post: Effect;
  sampler: GPUSampler;
}

interface Targets {
  scene: Target;
  bloomA: Target;
  bloomB: Target;
}

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const CLEAR: readonly [number, number, number, number] = [0, 0, 0, 1];

/** Scene constants measured against the reference photos; see scene.wgsl for the units. */
export const LOOK = {
  /** Elevated camera, matching the reference field photo; same vertical FOV for any aspect. */
  camera: { height: 5.5, pitch: -0.037, fovY: 0.733 },
  door: { x: 0, z: 14.5, yaw: 0, leaf: 2.5 },
  sun: { azimuth: -1.15, elevation: 0.72 },
  texture: 1,
  doorLight: 12,
  /** Blade patch around the door (radius in metres) and blade height. */
  grass: { radius: 9, height: 0.09 },
  post: { exposure: 1, bloomStrength: 0.6, grain: 0.035, vignette: 0.3, nightThreshold: 0.16, dayThreshold: 0.7, knee: 0.1 },
} as const;

/** Night holds, the day sweeps out of the door, holds, then the night flows back in. */
export const CYCLE_SECONDS = 16;

export function phaseAt(seconds: number): number {
  const t = ((seconds % CYCLE_SECONDS) + CYCLE_SECONDS) % CYCLE_SECONDS;
  if (t < 2.5) return 0;
  if (t < 9) return ease((t - 2.5) / 6.5);
  if (t < 12.5) return 1;
  return 1 - ease((t - 12.5) / 3.5);
}

function ease(x: number): number {
  const c = Math.min(1, Math.max(0, x));
  return c * c * (3 - 2 * c);
}

export async function run(canvas: HTMLCanvasElement): Promise<() => void> {
  const { init } = await import('vgpu');
  const gpu = await init();
  const surface = gpu.surface(canvas, { dpr: [1, 1.5] });
  const effects = createEffects(gpu, 'dreamcore-live');
  const targets = createTargets(gpu, surface.size, 'dreamcore-live');
  let disposed = false;

  setConstants(effects);
  setBindings(effects, targets);
  await prewarm(effects, targets, surface);

  let sawInitialResize = false;
  const unsubscribeResize = surface.onResize(() => {
    if (!sawInitialResize) {
      sawInitialResize = true;
      return;
    }
    if (disposed) return;
    resizeTargets(targets, surface.size);
    setBindings(effects, targets);
  });

  const handle = gpu.frame.loop((frame) => {
    // Only the two animated values are written each frame; everything else stays as set.
    setFrame(effects, { phase: phaseAt(gpu.time), time: gpu.time, samples: 1, grassShadows: false, wind: 1 });
    renderChain(frame, effects, targets, surface);
  });

  return () => {
    if (disposed) return;
    disposed = true;
    handle.stop();
    unsubscribeResize();
    surface.dispose();
    gpu.dispose();
  };
}

export async function renderThumb(gpu: Gpu, target: Target, opts: ThumbOptions = {}): Promise<void> {
  const effects = createEffects(gpu, 'dreamcore-thumb');
  const targets = createTargets(gpu, target.size, 'dreamcore-thumb');
  setConstants(effects);
  setBindings(effects, targets);
  await prewarm(effects, targets, target);
  const time = opts.time ?? 4.6;
  renderFrame(gpu, effects, targets, target, { phase: phaseAt(time), time, samples: 4 });
  await gpu.gpu.queue.onSubmittedWorkDone();
  await gpu.settled();
}

/** Renders one still into `target`. Used by the headless keyframe script. */
export async function renderStill(gpu: Gpu, target: Target, frameOpts: DreamcoreFrameOptions): Promise<void> {
  const effects = createEffects(gpu, 'dreamcore-still');
  const targets = createTargets(gpu, target.size, 'dreamcore-still');
  setConstants(effects);
  setBindings(effects, targets);
  await prewarm(effects, targets, target);
  renderFrame(gpu, effects, targets, target, frameOpts);
  await gpu.gpu.queue.onSubmittedWorkDone();
  await gpu.settled();
}

function createEffects(gpu: Gpu, label: string): Effects {
  return {
    scene: gpu.effect(sceneWgsl, { label: `${label}-scene` }),
    brightPass: gpu.effect(brightPassWgsl, { label: `${label}-bright-pass` }),
    // Each blur pass owns its uniform buffer; sharing one effect would make every pass
    // observe the last direction written in the frame.
    blurH1: gpu.effect(blurWgsl, { label: `${label}-blur-h1` }),
    blurV1: gpu.effect(blurWgsl, { label: `${label}-blur-v1` }),
    blurH2: gpu.effect(blurWgsl, { label: `${label}-blur-h2` }),
    blurV2: gpu.effect(blurWgsl, { label: `${label}-blur-v2` }),
    post: gpu.effect(postWgsl, { label: `${label}-post` }),
    sampler: gpu.sampler({ minFilter: 'linear', magFilter: 'linear' }),
  };
}

function createTargets(gpu: Gpu, size: readonly [number, number], label: string): Targets {
  const full = normalizeSize(size);
  const half = halfSize(full);
  return {
    scene: gpu.target({ size: full, format: HDR_FORMAT, label: `${label}-scene` }),
    bloomA: gpu.target({ size: half, format: HDR_FORMAT, label: `${label}-bloom-a` }),
    bloomB: gpu.target({ size: half, format: HDR_FORMAT, label: `${label}-bloom-b` }),
  };
}

function setConstants(effects: Effects): void {
  const { camera, door, sun, post, grass } = LOOK;
  effects.scene.set({
    params: {
      time: 0,
      phase: 0,
      camera: [camera.height, camera.pitch, camera.fovY, 1],
      door: [door.x, door.z, door.yaw, door.leaf],
      look: [sun.azimuth, sun.elevation, LOOK.texture, LOOK.doorLight],
      grass: [grass.radius, grass.height, 1, 0],
    },
  });
  effects.brightPass.set({ samp: effects.sampler, bright: { threshold: post.nightThreshold, knee: post.knee } });
  effects.blurH1.set({ samp: effects.sampler, blur: { direction: [1, 0], radius: 1 } });
  effects.blurV1.set({ samp: effects.sampler, blur: { direction: [0, 1], radius: 1 } });
  effects.blurH2.set({ samp: effects.sampler, blur: { direction: [1, 0], radius: 2.4 } });
  effects.blurV2.set({ samp: effects.sampler, blur: { direction: [0, 1], radius: 2.4 } });
  effects.post.set({
    samp: effects.sampler,
    post: { exposure: post.exposure, bloomStrength: post.bloomStrength, grain: post.grain, vignette: post.vignette, seed: 0.37, _pad: 0 },
  });
}

function setBindings(effects: Effects, targets: Targets): void {
  effects.scene.set({ params: { resolution: targets.scene.size } });
  effects.brightPass.set({ src: targets.scene });
  effects.blurH1.set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  effects.blurV1.set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  effects.blurH2.set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  effects.blurV2.set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  effects.post.set({ scene: targets.scene, bloom: targets.bloomA, post: { resolution: targets.scene.size } });
}

function setFrame(effects: Effects, frame: DreamcoreFrameOptions): void {
  const phase = Math.min(1, Math.max(0, frame.phase));
  const { camera, post, grass } = LOOK;
  effects.scene.set({
    params: {
      time: frame.time ?? 0,
      phase,
      camera: [camera.height, camera.pitch, camera.fovY, frame.samples ?? 1],
      grass: [grass.radius, grass.height, frame.grassShadows === false ? 0 : 1, frame.wind ?? 0],
    },
  });
  // The door only needs to bloom at night; by day the threshold rises so the field stays crisp.
  effects.brightPass.set({ bright: { threshold: post.nightThreshold + (post.dayThreshold - post.nightThreshold) * phase } });
  effects.post.set({ post: { seed: 0.37 + (frame.time ?? 0) * 0.01 } });
}

async function prewarm(effects: Effects, targets: Targets, output: Output): Promise<void> {
  await Promise.all([
    effects.scene.compile(targets.scene), effects.brightPass.compile(targets.bloomA),
    effects.blurH1.compile(targets.bloomB), effects.blurV1.compile(targets.bloomA),
    effects.blurH2.compile(targets.bloomB), effects.blurV2.compile(targets.bloomA),
    effects.post.compile({ colors: [output.format] }),
  ]);
}

function renderChain(frame: Frame, effects: Effects, targets: Targets, output: Output): void {
  frame.pass({ target: targets.scene, clear: CLEAR }, (pass) => pass.draw(effects.scene));
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.brightPass));
  frame.pass({ target: targets.bloomB, clear: CLEAR }, (pass) => pass.draw(effects.blurH1));
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.blurV1));
  frame.pass({ target: targets.bloomB, clear: CLEAR }, (pass) => pass.draw(effects.blurH2));
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.blurV2));
  frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(effects.post));
}

function renderFrame(gpu: Gpu, effects: Effects, targets: Targets, output: Target, frameOpts: DreamcoreFrameOptions): void {
  setFrame(effects, frameOpts);
  gpu.frame((frame) => renderChain(frame, effects, targets, output));
}

function resizeTargets(targets: Targets, size: readonly [number, number]): void {
  const full = normalizeSize(size);
  targets.scene.resize(full);
  targets.bloomA.resize(halfSize(full));
  targets.bloomB.resize(halfSize(full));
}

function normalizeSize(size: readonly [number, number]): [number, number] {
  return [Math.max(1, Math.floor(size[0])), Math.max(1, Math.floor(size[1]))];
}

function halfSize(size: readonly [number, number]): [number, number] {
  return [Math.max(1, Math.ceil(size[0] / 2)), Math.max(1, Math.ceil(size[1] / 2))];
}
