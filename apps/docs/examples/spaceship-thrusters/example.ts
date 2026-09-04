import type { Effect, Frame, Gpu, Surface, Target } from 'vgpu';

import bakeDetailWgsl from './bake-detail.wgsl';
import bakeNoiseWgsl from './bake-noise.wgsl';
import blurWgsl from './blur.wgsl';
import brightPassWgsl from './bright-pass.wgsl';
import compositeWgsl from './composite.wgsl';
import debugPreviewWgsl from './debug-preview.wgsl';
import fireWgsl from './fire.wgsl';

type Output = Surface | Target;

export type ThrusterIntermediate = 'noise-atlas' | 'detail' | 'fire-hdr' | 'bloom';

export interface ThrusterThumbOptions {
  time?: number;
  /** Receives every internal render target so headless runs can inspect the graph. */
  onIntermediateRendered?: (
    kind: ThrusterIntermediate,
    pixels: Uint8Array,
    size: readonly [number, number],
  ) => void | Promise<void>;
}

interface Effects {
  bakeNoise: Effect;
  bakeDetail: Effect;
  fire: Effect;
  brightPass: Effect;
  blurH1: Effect;
  blurV1: Effect;
  blurH2: Effect;
  blurV2: Effect;
  composite: Effect;
  clampSampler: GPUSampler;
  repeatSampler: GPUSampler;
}

interface Targets {
  /** Tileable 3D noise packed as 64 slices of 128² (+1 texel periodic border). Baked once. */
  noiseAtlas: Target;
  /** Tileable 2D high-frequency detail. Baked once. */
  detail: Target;
  /** Half-resolution HDR fire pass. */
  fire: Target;
  bloomA: Target;
  bloomB: Target;
}

// Must match the constants in thruster-common.wgsl.
const NOISE_ATLAS_SIZE = (128 + 2) * 8;
const DETAIL_SIZE = 512;
const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const FIRE_SCALE = 0.5;
const BLOOM_HEIGHT = 240;
const CLEAR: readonly [number, number, number, number] = [0, 0, 0, 1];

export async function run(canvas: HTMLCanvasElement): Promise<() => void> {
  const { init } = await import('vgpu');
  const gpu = await init();
  const surface = gpu.surface(canvas, { dpr: [1, 1.5] });
  const effects = createEffects(gpu, 'thrusters-live');
  const targets = createTargets(gpu, surface.size, 'thrusters-live');
  let disposed = false;

  setConstants(effects, targets);
  setBindings(effects, targets);
  await prewarm(effects, targets, surface);
  bakeNoise(gpu, effects, targets);

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
    // Only the clock changes per frame; every other binding is stable.
    effects.fire.set({ params: { time: gpu.time } });
    effects.composite.set({ composite: { time: gpu.time } });
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

export async function renderThumb(gpu: Gpu, target: Target, opts: ThrusterThumbOptions = {}): Promise<void> {
  const effects = createEffects(gpu, 'thrusters-thumb');
  const targets = createTargets(gpu, target.size, 'thrusters-thumb');
  const time = opts.time ?? 6.2;
  setConstants(effects, targets);
  setBindings(effects, targets);
  await prewarm(effects, targets, target);
  bakeNoise(gpu, effects, targets);

  effects.fire.set({ params: { time } });
  effects.composite.set({ composite: { time } });
  gpu.frame((frame) => renderChain(frame, effects, targets, target));
  await gpu.gpu.queue.onSubmittedWorkDone();

  if (opts.onIntermediateRendered) {
    await dumpIntermediates(gpu, effects, targets, opts.onIntermediateRendered);
  }
  await gpu.settled();
}

/**
 * Reads every internal target back for headless inspection. 8-bit targets are
 * read directly; HDR targets go through a tonemapping preview pass first
 * because readback only supports 8-bit formats.
 */
async function dumpIntermediates(
  gpu: Gpu,
  effects: Effects,
  targets: Targets,
  report: NonNullable<ThrusterThumbOptions['onIntermediateRendered']>,
): Promise<void> {
  await report('noise-atlas', await targets.noiseAtlas.read(), targets.noiseAtlas.size);
  await report('detail', await targets.detail.read(), targets.detail.size);
  const preview = gpu.effect(debugPreviewWgsl, { label: 'thrusters-debug-preview' });
  preview.set({ samp: effects.clampSampler, preview: { exposure: 1, mode: 0 } });
  for (const [kind, source] of [['fire-hdr', targets.fire], ['bloom', targets.bloomA]] as const) {
    const previewTarget = gpu.target({ size: source.size, format: 'rgba8unorm', label: `thrusters-preview-${kind}` });
    preview.set({ src: source });
    await preview.compile(previewTarget);
    gpu.frame((frame) => frame.pass({ target: previewTarget, clear: CLEAR }, (pass) => pass.draw(preview)));
    await gpu.gpu.queue.onSubmittedWorkDone();
    await report(kind, await previewTarget.read(), previewTarget.size);
    previewTarget.color.destroy();
  }
}

function createEffects(gpu: Gpu, label: string): Effects {
  return {
    bakeNoise: gpu.effect(bakeNoiseWgsl, { label: `${label}-bake-noise` }),
    bakeDetail: gpu.effect(bakeDetailWgsl, { label: `${label}-bake-detail` }),
    fire: gpu.effect(fireWgsl, { label: `${label}-fire` }),
    brightPass: gpu.effect(brightPassWgsl, { label: `${label}-bright-pass` }),
    // Each blur pass owns its uniform buffer so the encoded direction/radius stay distinct.
    blurH1: gpu.effect(blurWgsl, { label: `${label}-blur-h1` }),
    blurV1: gpu.effect(blurWgsl, { label: `${label}-blur-v1` }),
    blurH2: gpu.effect(blurWgsl, { label: `${label}-blur-h2` }),
    blurV2: gpu.effect(blurWgsl, { label: `${label}-blur-v2` }),
    composite: gpu.effect(compositeWgsl, { label: `${label}-composite` }),
    // The atlas must clamp: tiles carry their own periodic border.
    clampSampler: gpu.sampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }),
    repeatSampler: gpu.sampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' }),
  };
}

function createTargets(gpu: Gpu, size: readonly [number, number], label: string): Targets {
  const full = normalizeSize(size);
  return {
    noiseAtlas: gpu.target({ size: [NOISE_ATLAS_SIZE, NOISE_ATLAS_SIZE], format: 'rgba8unorm', label: `${label}-noise-atlas` }),
    detail: gpu.target({ size: [DETAIL_SIZE, DETAIL_SIZE], format: 'rgba8unorm', label: `${label}-detail` }),
    fire: gpu.target({ size: fireSize(full), format: HDR_FORMAT, label: `${label}-fire` }),
    bloomA: gpu.target({ size: bloomSize(full), format: HDR_FORMAT, label: `${label}-bloom-a` }),
    bloomB: gpu.target({ size: bloomSize(full), format: HDR_FORMAT, label: `${label}-bloom-b` }),
  };
}

function setConstants(effects: Effects, targets: Targets): void {
  effects.fire.set({
    params: { time: 0, motion: 1 },
    atlas: targets.noiseAtlas,
    detail: targets.detail,
    atlasSamp: effects.clampSampler,
    detailSamp: effects.repeatSampler,
  });
  effects.brightPass.set({ samp: effects.clampSampler, bright: { threshold: 0.7, knee: 0.5 } });
  effects.blurH1.set({ samp: effects.clampSampler, blur: { direction: [1, 0], radius: 1 } });
  effects.blurV1.set({ samp: effects.clampSampler, blur: { direction: [0, 1], radius: 1 } });
  effects.blurH2.set({ samp: effects.clampSampler, blur: { direction: [1, 0], radius: 2.6 } });
  effects.blurV2.set({ samp: effects.clampSampler, blur: { direction: [0, 1], radius: 2.6 } });
  effects.composite.set({ samp: effects.clampSampler, composite: { exposure: 1.05, bloomStrength: 0.8, grain: 0.015, time: 0 } });
}

function setBindings(effects: Effects, targets: Targets): void {
  effects.fire.set({ params: { resolution: targets.fire.size } });
  effects.brightPass.set({ src: targets.fire });
  effects.blurH1.set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  effects.blurV1.set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  effects.blurH2.set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  effects.blurV2.set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  effects.composite.set({ scene: targets.fire, bloom: targets.bloomA });
}

async function prewarm(effects: Effects, targets: Targets, output: Output): Promise<void> {
  await Promise.all([
    effects.bakeNoise.compile(targets.noiseAtlas), effects.bakeDetail.compile(targets.detail),
    effects.fire.compile(targets.fire), effects.brightPass.compile(targets.bloomA),
    effects.blurH1.compile(targets.bloomB), effects.blurV1.compile(targets.bloomA),
    effects.blurH2.compile(targets.bloomB), effects.blurV2.compile(targets.bloomA),
    effects.composite.compile({ colors: [output.format] }),
  ]);
}

/** Bakes the noise textures. Called once; the fire pass only ever reads them. */
function bakeNoise(gpu: Gpu, effects: Effects, targets: Targets): void {
  gpu.frame((frame) => {
    frame.pass({ target: targets.noiseAtlas, clear: CLEAR }, (pass) => pass.draw(effects.bakeNoise));
    frame.pass({ target: targets.detail, clear: CLEAR }, (pass) => pass.draw(effects.bakeDetail));
  });
}

function renderChain(frame: Frame, effects: Effects, targets: Targets, output: Output): void {
  frame.pass({ target: targets.fire, clear: CLEAR }, (pass) => pass.draw(effects.fire));
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.brightPass));
  frame.pass({ target: targets.bloomB, clear: CLEAR }, (pass) => pass.draw(effects.blurH1));
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.blurV1));
  frame.pass({ target: targets.bloomB, clear: CLEAR }, (pass) => pass.draw(effects.blurH2));
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.blurV2));
  frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(effects.composite));
}

function resizeTargets(targets: Targets, size: readonly [number, number]): void {
  const full = normalizeSize(size);
  targets.fire.resize(fireSize(full));
  targets.bloomA.resize(bloomSize(full));
  targets.bloomB.resize(bloomSize(full));
}

function normalizeSize(size: readonly [number, number]): [number, number] {
  return [Math.max(1, Math.floor(size[0])), Math.max(1, Math.floor(size[1]))];
}

function fireSize(size: readonly [number, number]): [number, number] {
  return [Math.max(1, Math.round(size[0] * FIRE_SCALE)), Math.max(1, Math.round(size[1] * FIRE_SCALE))];
}

function bloomSize(size: readonly [number, number]): [number, number] {
  const height = Math.max(1, Math.min(BLOOM_HEIGHT, size[1]));
  return [Math.max(1, Math.round(height * size[0] / size[1])), height];
}
