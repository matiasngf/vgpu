import type { Draw, Effect, Frame, Gpu, PingPongTargets, Surface, Target } from 'vgpu';

import aoApplyWgsl from './ao-apply.wgsl';
import aoBlurWgsl from './ao-blur.wgsl';
import aoWgsl from './ao.wgsl';
import bakeDetailWgsl from './bake-detail.wgsl';
import bakeMaterialAtlasWgsl from './bake-material-atlas.wgsl';
import bakeMaterialFinishWgsl from './bake-material-finish.wgsl';
import bakeMaterialHeightWgsl from './bake-material-height.wgsl';
import bakeNoiseWgsl from './bake-noise.wgsl';
import blurWgsl from './blur.wgsl';
import brightPassWgsl from './bright-pass.wgsl';
import compositeWgsl from './composite.wgsl';
import debugPreviewWgsl from './debug-preview.wgsl';
import fireDirectWgsl from './fire-direct.wgsl';
import fireWgsl from './fire.wgsl';
import gridWgsl from './grid.wgsl';
import postWgsl from './post.wgsl';
import resolveWgsl from './resolve.wgsl';
import sceneWgsl from './scene.wgsl';
import shadowWgsl from './shadow.wgsl';
import { invert, lookAt, merge, multiply, pack, perspective, type Vec3 } from './cad';
import { buildEngine, buildGround, buildStand, DEFAULT_ENGINE, engineToStand, FILL_LIGHT, KEY_LIGHT } from './engine';

type Output = Surface | Target;

export type ThrusterIntermediate =
  | 'noise-atlas' | 'detail' | 'concrete' | 'concrete-normal' | 'gravel' | 'gravel-normal'
  | 'shadow-map' | 'scene-color' | 'scene-depth' | 'scene-normal' | 'plume-grid' | 'fire-hdr' | 'bloom'
  | 'ao' | 'scene-lit' | 'composite';

/**
 * 'fast': the interactive pipeline (what the docs page runs).
 * 'social': the same graph plus screen-space ambient occlusion on the
 *           geometry, a higher-resolution bloom chain and a final lens pass
 *           (edge softness, vignette, photographic grain) for renders meant
 *           to be posted rather than played. The plume itself is untouched.
 */
export type ThrusterQuality = 'fast' | 'social';
export const RENDER_QUALITY: ThrusterQuality = 'fast';

export interface ThrusterCamera {
  position: Vec3;
  target: Vec3;
  fovDeg?: number;
}

export interface ThrusterThumbOptions {
  time?: number;
  /** Pipeline variant; defaults to `RENDER_QUALITY`. */
  quality?: ThrusterQuality;
  /** Override the camera (headless artifact hunting from other angles). */
  camera?: ThrusterCamera;
  /** Receives every internal render target so headless runs can inspect the graph. */
  onIntermediateRendered?: (
    kind: ThrusterIntermediate,
    pixels: Uint8Array,
    size: readonly [number, number],
  ) => void | Promise<void>;
}

// --- Scene layout (units: nozzle exit radius = 1) ------------------------------

/** Height of the engine axis above the pad. */
const AXIS_HEIGHT = 1.7;
/** Exhaust direction: horizontal +X (nozzle exit at the origin). */
const PLUME_AXIS: Vec3 = [1, 0, 0];
const PLUME = { nozzle: [0, AXIS_HEIGHT, 0] as Vec3, r0: 0.93, spread: 0.03, length: 45, sootGain: 0.2, glowGain: 5, exitGain: 5 };
const CAMERA = { position: [-10, 15, 10] as Vec3, target: [0.8, 0.8, -1.2] as Vec3, fovDeg: 40, near: 0.5, far: 400 };
/** Named camera presets, also reachable from the headless scripts. */
export const CAMERA_PRESETS: Record<string, ThrusterCamera> = {
  default: { position: CAMERA.position, target: CAMERA.target, fovDeg: CAMERA.fovDeg },
  behind: { position: [-14, 4, 2], target: [6, 1.5, 0], fovDeg: 45 },
  front: { position: [26, 5, 6], target: [0, 1.7, 0], fovDeg: 40 },
  top: { position: [4, 24, 0.5], target: [4, 0, 0], fovDeg: 45 },
  closeup: { position: [-1.5, 4.5, 6], target: [2.5, 1.7, 0], fovDeg: 35 },
  side: { position: [6, 3, 16], target: [6, 1.7, 0], fovDeg: 40 },
};
/** Where the two floodlights are aimed. */
const KEY_TARGET: Vec3 = [1.5, 1.2, 0];
const FILL_TARGET: Vec3 = [1, 1.5, 0];
/** Perspective shadow camera at the key floodlight. */
const SHADOW = { size: 2048, fovDeg: 95, near: 1, far: 60 };
// Night: no sun. A white metal-halide key floodlight by the stand (shadowed),
// a warmer fill floodlight across the pad, a faint night-sky ambient, and the
// plume itself as the dominant light on everything.
const LIGHTING = {
  ambient: 0.12,
  skyColor: [0.03, 0.045, 0.09],
  groundColor: [0.02, 0.015, 0.02],
  fogColor: [0.004, 0.005, 0.01],
  fogDensity: 0.006,
  keyLight: [...KEY_LIGHT, 150] as [number, number, number, number],
  keyColor: [0.95, 1.0, 0.93],
  keySpot: [...normalize3(sub3(KEY_TARGET, KEY_LIGHT)), Math.cos((58 * Math.PI) / 180)] as [number, number, number, number],
  fillLight: [...FILL_LIGHT, 70] as [number, number, number, number],
  fillColor: [1.0, 0.9, 0.72],
  fillSpot: [...normalize3(sub3(FILL_TARGET, FILL_LIGHT)), Math.cos((55 * Math.PI) / 180)] as [number, number, number, number],
  /** World units, on top of the normal offset. */
  shadowBias: 0.03,
};
/** Segment light that stands in for the plume's glow on the geometry. */
const PLUME_LIGHT = { length: 32, intensity: 85 };

interface Effects {
  quality: ThrusterQuality;
  bakeNoise: Effect;
  bakeDetail: Effect;
  /** Ground materials: height + masks, then normal / albedo / roughness, once per material. */
  bakeConcreteHeight: Effect;
  bakeConcrete: Effect;
  bakeConcreteAtlas: Effect;
  bakeGravelHeight: Effect;
  bakeGravel: Effect;
  bakeGravelAtlas: Effect;
  grid: Effect;
  fire: Effect;
  resolve: Effect;
  brightPass: Effect;
  blurH1: Effect;
  blurV1: Effect;
  blurH2: Effect;
  blurV2: Effect;
  composite: Effect;
  /** Social pipeline only. */
  ao?: Effect;
  aoBlur?: Effect;
  aoApply?: Effect;
  post?: Effect;
  clampSampler: GPUSampler;
  repeatSampler: GPUSampler;
}

type Mesh = ReturnType<Gpu['mesh']>;

interface Geometry {
  meshes: Mesh[];
  draws: Draw[];
  shadowDraws: Draw[];
}

interface Targets {
  /** Tileable 3D noise packed as 64 slices of 128² (+1 texel periodic border). Baked once. */
  noiseAtlas: Target;
  /** Tileable 2D high-frequency detail. Baked once. */
  detail: Target;
  /** Scratch targets for the material bakes: height + masks (rgba16float), then the finished 1024² tile. */
  materialHeight: Target;
  materialTile: Target;
  /** Baked ground materials as level atlases: colors[0] albedo + roughness, colors[1] tangent normal + height + cavity. */
  concrete: Target;
  gravel: Target;
  /** Sun shadow map: light-space depth in r32float. */
  shadow: Target;
  /** Lit geometry: radiance in colors[0], camera distance in colors[1] (r32float), normal + occludable share in colors[2], plus depth. */
  scene: Target;
  /** Social pipeline only: raw and half-blurred occlusion, the occluded scene, and the composite before the lens pass. */
  ao?: Target;
  aoBlur?: Target;
  sceneLit?: Target;
  ldr?: Target;
  /** Plume grid: cone-fitted slice atlas of emission + extinction, refilled every frame. */
  plumeGrid: Target;
  /** One 2x2 phase of the plume per frame, quarter resolution (fire + aux). */
  march: Target;
  /** Half-resolution plume history (fire + aux), interleaved from `march`. Rebuilt on resize. */
  fireHistory: PingPongTargets;
  bloomA: Target;
  bloomB: Target;
}

// Must match the constants in thruster-common.wgsl.
const NOISE_ATLAS_SIZE = (128 + 2) * 8;
const DETAIL_SIZE = 512;
/** Must match material-common.wgsl: 1024² tiles, packed as a 4-level atlas with periodic borders. */
const MATERIAL_SIZE = 1024;
const MATERIAL_ATLAS: [number, number] = [1368, 1026];
/** 16 x 16 slices of (64 + 2 border)². Must match plume-volume.wgsl. */
const PLUME_GRID_SIZE = (64 + 2) * 16;
const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const FIRE_SCALE = 0.5; // the plume is soft; the composite upsamples it depth-aware
/**
 * 'grid': evaluate the volume once per frame into the plume grid and march
 *         two fetches per step (cost independent of resolution and steps).
 * 'direct': evaluate the volume at every march step (cheaper at low
 *         resolution; see `profile` to compare on your GPU).
 */
export const PLUME_MODE: 'grid' | 'direct' = 'grid';
/** Fresh-sample weight in the temporal resolve (1 = no history blending). */
const TEMPORAL_BLEND = 0.75;
/** How much a stale 2x2 phase leans on its block's fresh sample each frame. */
const TEMPORAL_NEIGHBOR = 0.3;
/** Per-variant knobs. Everything not listed here is shared between the two pipelines. */
const QUALITY: Record<ThrusterQuality, {
  bloomHeight: number;
  /** Screen-space ambient occlusion on the geometry (world-space radius in nozzle radii). */
  ao: { radius: number; intensity: number; bias: number } | null;
  /** Final lens pass; null renders the composite straight to the output. */
  post: { grain: number; vignette: number; edgeBlur: number; edgeStart: number } | null;
  /** Composite-side vignette and grain (the social variant moves both to the post pass). */
  composite: { vignette: number; grain: number };
}> = {
  fast: { bloomHeight: 240, ao: null, post: null, composite: { vignette: 0.28, grain: 0.02 } },
  social: {
    bloomHeight: 480,
    ao: { radius: 2.0, intensity: 4.5, bias: 0.08 },
    post: { grain: 0.085, vignette: 0.45, edgeBlur: 0.009, edgeStart: 0.5 },
    composite: { vignette: 0, grain: 0 },
  },
};
const CLEAR: readonly [number, number, number, number] = [0, 0, 0, 1];

export async function run(canvas: HTMLCanvasElement): Promise<() => void> {
  const { init } = await import('vgpu');
  const gpu = await init();
  const surface = gpu.surface(canvas, { dpr: [1, 1.5] });
  const effects = createEffects(gpu, 'thrusters-live', RENDER_QUALITY);
  const targets = createTargets(gpu, surface.size, 'thrusters-live', RENDER_QUALITY);
  const geometry = createGeometry(gpu, effects, targets, 'thrusters-live');
  let disposed = false;

  setConstants(effects, targets);
  setBindings(effects, geometry, targets);
  await prewarm(effects, geometry, targets, surface);
  // Subscribe before the first frame: the bake frame applies any auto-resize
  // that happened during prewarm, and target.resize() is a no-op for equal
  // sizes, so handling the initial event is cheap and never misses a resize.
  const unsubscribeResize = surface.onResize(() => {
    if (disposed) return;
    resizeTargets(gpu, targets, surface.size);
    setBindings(effects, geometry, targets);
  });
  bakeStatic(gpu, effects, geometry, targets);

  let frameIndex = 0;
  const handle = gpu.frame.loop((frame) => {
    // Only the clock and the temporal phase change per frame.
    setFrame(effects, gpu.time, frameIndex++);
    renderChain(frame, effects, geometry, targets, surface);
  });

  return () => {
    if (disposed) return;
    disposed = true;
    handle.stop();
    unsubscribeResize();
    for (const mesh of geometry.meshes) mesh.destroy();
    destroyTargets(targets);
    surface.dispose();
    gpu.dispose();
  };
}

export async function renderThumb(gpu: Gpu, target: Target, opts: ThrusterThumbOptions = {}): Promise<void> {
  const quality = opts.quality ?? RENDER_QUALITY;
  const effects = createEffects(gpu, 'thrusters-thumb', quality);
  const targets = createTargets(gpu, target.size, 'thrusters-thumb', quality);
  const geometry = createGeometry(gpu, effects, targets, 'thrusters-thumb');
  const time = opts.time ?? 6.2;
  setConstants(effects, targets);
  setBindings(effects, geometry, targets, opts.camera);
  await prewarm(effects, geometry, targets, target);
  bakeStatic(gpu, effects, geometry, targets);

  // Four frames at a fixed time fill all four phases of the history, so the
  // still is a complete, deterministic plume.
  for (let phase = 0; phase < 4; phase++) {
    setFrame(effects, time, phase);
    gpu.frame((frame) => renderChain(frame, effects, geometry, targets, target));
  }
  await gpu.gpu.queue.onSubmittedWorkDone();

  if (opts.onIntermediateRendered) {
    await dumpIntermediates(gpu, targets, opts.onIntermediateRendered);
  }
  await gpu.settled();
  for (const mesh of geometry.meshes) mesh.destroy();
  destroyTargets(targets);
}

/** Offscreen targets own their textures; release them when the graph is torn down. */
function destroyTargets(targets: Targets): void {
  for (const target of Object.values(targets)) (target as { destroy?: () => void }).destroy?.();
  destroyHistory(targets.fireHistory);
}

function destroyHistory(history: PingPongTargets): void {
  for (const target of [history.read, history.write]) (target as { destroy?: () => void }).destroy?.();
}

/**
 * Renders `frames` consecutive animated frames (dt apart) and hands each one
 * back, for checking the temporal interleave on moving fire headlessly.
 */
export async function renderSequence(
  gpu: Gpu,
  target: Target,
  frames: number,
  dt: number,
  onFrame: (index: number, pixels: Uint8Array, size: readonly [number, number]) => void | Promise<void>,
  startTime = 6.2,
  quality: ThrusterQuality = RENDER_QUALITY,
): Promise<void> {
  const effects = createEffects(gpu, 'thrusters-sequence', quality);
  const targets = createTargets(gpu, target.size, 'thrusters-sequence', quality);
  const geometry = createGeometry(gpu, effects, targets, 'thrusters-sequence');
  setConstants(effects, targets);
  setBindings(effects, geometry, targets);
  await prewarm(effects, geometry, targets, target);
  bakeStatic(gpu, effects, geometry, targets);
  for (let i = 0; i < frames; i++) {
    setFrame(effects, startTime + i * dt, i);
    gpu.frame((frame) => renderChain(frame, effects, geometry, targets, target));
    await gpu.gpu.queue.onSubmittedWorkDone();
    await onFrame(i, await target.read(), target.size);
  }
  await gpu.settled();
  for (const mesh of geometry.meshes) mesh.destroy();
  destroyTargets(targets);
}

export interface ThrusterProfile {
  /** Milliseconds per pass, median over the measured frames (GPU wall clock). */
  passes: Record<ProfileStage, number>;
  quality: ThrusterQuality;
  frames: number;
  size: readonly [number, number];
  fireSize: readonly [number, number];
}

/**
 * Times each pass of the per-frame chain separately (submit + wait), so the
 * headless harness can compare optimizations. Baking is excluded.
 */
type ProfileStage = 'scene' | 'ao' | 'grid' | 'fire' | 'bloom' | 'composite' | 'post';

export async function profile(gpu: Gpu, target: Target, frames = 20, time = 6.2, quality: ThrusterQuality = RENDER_QUALITY): Promise<ThrusterProfile> {
  const effects = createEffects(gpu, 'thrusters-profile', quality);
  const targets = createTargets(gpu, target.size, 'thrusters-profile', quality);
  const geometry = createGeometry(gpu, effects, targets, 'thrusters-profile');
  setConstants(effects, targets);
  setBindings(effects, geometry, targets);
  await prewarm(effects, geometry, targets, target);
  bakeStatic(gpu, effects, geometry, targets);
  await gpu.gpu.queue.onSubmittedWorkDone();

  const stages: Record<ProfileStage, (frame: Frame) => void> = {
    scene: (frame) => frame.pass({ target: targets.scene, clear: [0, 0, 0, 0] }, (pass) => { for (const draw of geometry.draws) pass.draw(draw); }),
    ao: (frame) => renderOcclusion(frame, effects, targets),
    grid: (frame) => { if (PLUME_MODE === 'grid') frame.pass({ target: targets.plumeGrid, clear: false }, (pass) => pass.draw(effects.grid)); },
    fire: (frame) => {
      frame.pass({ target: targets.march, clear: CLEAR }, (pass) => pass.draw(effects.fire));
      frame.pass({ target: targets.fireHistory.write, clear: CLEAR }, (pass) => pass.draw(effects.resolve));
      targets.fireHistory.swap();
      setHistoryReaders(effects, targets);
    },
    bloom: (frame) => {
      frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.brightPass));
      frame.pass({ target: targets.bloomB, clear: CLEAR }, (pass) => pass.draw(effects.blurH1));
      frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.blurV1));
      frame.pass({ target: targets.bloomB, clear: CLEAR }, (pass) => pass.draw(effects.blurH2));
      frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.blurV2));
    },
    composite: (frame) => frame.pass({ target: targets.ldr ?? target, clear: CLEAR }, (pass) => pass.draw(effects.composite)),
    post: (frame) => { const post = effects.post; if (post) frame.pass({ target, clear: CLEAR }, (pass) => pass.draw(post)); },
  };
  const samples = Object.fromEntries(Object.keys(stages).map((name) => [name, [] as number[]])) as Record<ProfileStage, number[]>;
  for (let i = 0; i < frames + 2; i++) {
    setFrame(effects, time + i / 60, i);
    for (const [name, stage] of Object.entries(stages)) {
      const started = performance.now();
      gpu.frame(stage);
      await gpu.gpu.queue.onSubmittedWorkDone();
      if (i >= 2) samples[name as ProfileStage].push(performance.now() - started); // skip warm-up frames
    }
  }
  const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] ?? 0;
  const result: ThrusterProfile = {
    passes: Object.fromEntries(Object.entries(samples).map(([name, values]) => [name, median(values)])) as Record<ProfileStage, number>,
    quality,
    frames,
    size: target.size,
    fireSize: targets.march.size,
  };
  for (const mesh of geometry.meshes) mesh.destroy();
  destroyTargets(targets);
  return result;
}

/**
 * Reads every internal target back for headless inspection. 8-bit targets are
 * read directly; HDR and depth targets go through a preview pass first
 * because readback only supports 8-bit formats.
 */
async function dumpIntermediates(
  gpu: Gpu,
  targets: Targets,
  report: NonNullable<ThrusterThumbOptions['onIntermediateRendered']>,
): Promise<void> {
  await report('noise-atlas', await targets.noiseAtlas.read(), targets.noiseAtlas.size);
  await report('detail', await targets.detail.read(), targets.detail.size);
  await report('concrete', await targets.concrete.read(), targets.concrete.size);
  await report('gravel', await targets.gravel.read(), targets.gravel.size);
  const preview = gpu.effect(debugPreviewWgsl, { label: 'thrusters-debug-preview' });
  type Job = [ThrusterIntermediate, Target['color'], readonly [number, number], { exposure: number; mode: number }];
  const jobs: Job[] = [
    ['shadow-map', targets.shadow.color, targets.shadow.size, { exposure: SHADOW.far, mode: 2 }],
    ['scene-color', targets.scene.color, targets.scene.size, { exposure: 1, mode: 0 }],
    ['scene-depth', targets.scene.colors[1], targets.scene.size, { exposure: 60, mode: 2 }],
    ['scene-normal', targets.scene.colors[2], targets.scene.size, { exposure: 1, mode: 1 }],
    ['concrete-normal', targets.concrete.colors[1], targets.concrete.size, { exposure: 1, mode: 1 }],
    ['gravel-normal', targets.gravel.colors[1], targets.gravel.size, { exposure: 1, mode: 1 }],
    ['plume-grid', targets.plumeGrid.color, targets.plumeGrid.size, { exposure: 0.25, mode: 0 }],
    ['fire-hdr', targets.fireHistory.read.color, targets.fireHistory.read.size, { exposure: 1, mode: 0 }],
    ['bloom', targets.bloomA.color, targets.bloomA.size, { exposure: 1, mode: 0 }],
  ];
  if (targets.ao) jobs.push(['ao', targets.ao.color, targets.ao.size, { exposure: 1, mode: 2 }]);
  if (targets.sceneLit) jobs.push(['scene-lit', targets.sceneLit.color, targets.sceneLit.size, { exposure: 1, mode: 0 }]);
  if (targets.ldr) await report('composite', await targets.ldr.read(), targets.ldr.size);
  for (const [kind, source, size, params] of jobs) {
    const previewTarget = gpu.target({ size, format: 'rgba8unorm', label: `thrusters-preview-${kind}` });
    preview.set({ src: source, preview: params });
    await preview.compile(previewTarget);
    gpu.frame((frame) => frame.pass({ target: previewTarget, clear: CLEAR }, (pass) => pass.draw(preview)));
    await gpu.gpu.queue.onSubmittedWorkDone();
    await report(kind, await previewTarget.read(), previewTarget.size);
    previewTarget.color.destroy();
  }
}

function createEffects(gpu: Gpu, label: string, quality: ThrusterQuality): Effects {
  const variant = QUALITY[quality];
  return {
    quality,
    bakeNoise: gpu.effect(bakeNoiseWgsl, { label: `${label}-bake-noise` }),
    bakeDetail: gpu.effect(bakeDetailWgsl, { label: `${label}-bake-detail` }),
    bakeConcreteHeight: gpu.effect(bakeMaterialHeightWgsl, { label: `${label}-bake-concrete-height` }),
    bakeConcrete: gpu.effect(bakeMaterialFinishWgsl, { label: `${label}-bake-concrete` }),
    bakeConcreteAtlas: gpu.effect(bakeMaterialAtlasWgsl, { label: `${label}-bake-concrete-atlas` }),
    bakeGravelHeight: gpu.effect(bakeMaterialHeightWgsl, { label: `${label}-bake-gravel-height` }),
    bakeGravel: gpu.effect(bakeMaterialFinishWgsl, { label: `${label}-bake-gravel` }),
    bakeGravelAtlas: gpu.effect(bakeMaterialAtlasWgsl, { label: `${label}-bake-gravel-atlas` }),
    grid: gpu.effect(gridWgsl, { label: `${label}-grid` }),
    fire: gpu.effect(PLUME_MODE === 'grid' ? fireWgsl : fireDirectWgsl, { label: `${label}-fire` }),
    resolve: gpu.effect(resolveWgsl, { label: `${label}-resolve` }),
    brightPass: gpu.effect(brightPassWgsl, { label: `${label}-bright-pass` }),
    // Each blur pass owns its uniform buffer so the encoded direction/radius stay distinct.
    blurH1: gpu.effect(blurWgsl, { label: `${label}-blur-h1` }),
    blurV1: gpu.effect(blurWgsl, { label: `${label}-blur-v1` }),
    blurH2: gpu.effect(blurWgsl, { label: `${label}-blur-h2` }),
    blurV2: gpu.effect(blurWgsl, { label: `${label}-blur-v2` }),
    composite: gpu.effect(compositeWgsl, { label: `${label}-composite` }),
    ...(variant.ao ? {
      ao: gpu.effect(aoWgsl, { label: `${label}-ao` }),
      aoBlur: gpu.effect(aoBlurWgsl, { label: `${label}-ao-blur` }),
      aoApply: gpu.effect(aoApplyWgsl, { label: `${label}-ao-apply` }),
    } : {}),
    ...(variant.post ? { post: gpu.effect(postWgsl, { label: `${label}-post` }) } : {}),
    // The atlas must clamp: tiles carry their own periodic border.
    clampSampler: gpu.sampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }),
    repeatSampler: gpu.sampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' }),
  };
}

function createTargets(gpu: Gpu, size: readonly [number, number], label: string, quality: ThrusterQuality): Targets {
  const full = normalizeSize(size);
  const variant = QUALITY[quality];
  const bloom = bloomSize(full, variant.bloomHeight);
  return {
    noiseAtlas: gpu.target({ size: [NOISE_ATLAS_SIZE, NOISE_ATLAS_SIZE], format: 'rgba8unorm', label: `${label}-noise-atlas` }),
    detail: gpu.target({ size: [DETAIL_SIZE, DETAIL_SIZE], format: 'rgba8unorm', label: `${label}-detail` }),
    materialHeight: gpu.target({ size: [MATERIAL_SIZE, MATERIAL_SIZE], format: 'rgba16float', label: `${label}-material-height` }),
    materialTile: gpu.target({ size: [MATERIAL_SIZE, MATERIAL_SIZE], colors: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }], label: `${label}-material-tile` }),
    concrete: gpu.target({ size: MATERIAL_ATLAS, colors: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }], label: `${label}-concrete` }),
    gravel: gpu.target({ size: MATERIAL_ATLAS, colors: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }], label: `${label}-gravel` }),
    shadow: gpu.target({ size: [SHADOW.size, SHADOW.size], format: 'r32float', depth: true, label: `${label}-shadow` }),
    scene: gpu.target({ size: full, colors: [{ format: HDR_FORMAT }, { format: 'r32float' }, { format: 'rgba8unorm' }], depth: true, label: `${label}-scene` }),
    ...(variant.ao ? {
      ao: gpu.target({ size: full, format: 'r8unorm', label: `${label}-ao` }),
      aoBlur: gpu.target({ size: full, format: 'r8unorm', label: `${label}-ao-blur` }),
      sceneLit: gpu.target({ size: full, format: HDR_FORMAT, label: `${label}-scene-lit` }),
    } : {}),
    ...(variant.post ? { ldr: gpu.target({ size: full, format: 'rgba8unorm', label: `${label}-ldr` }) } : {}),
    plumeGrid: gpu.target({ size: [PLUME_GRID_SIZE, PLUME_GRID_SIZE], format: HDR_FORMAT, label: `${label}-plume-grid` }),
    march: gpu.target({ size: marchSize(full), colors: [{ format: HDR_FORMAT }, { format: HDR_FORMAT }], label: `${label}-march` }),
    fireHistory: createHistory(gpu, full, label),
    bloomA: gpu.target({ size: bloom, format: HDR_FORMAT, label: `${label}-bloom-a` }),
    bloomB: gpu.target({ size: bloom, format: HDR_FORMAT, label: `${label}-bloom-b` }),
  };
}

/** Builds the parametric engine, stand and pad and uploads them as draws. Both floodlights shine from off frame, so no fixtures. */
function createGeometry(gpu: Gpu, effects: Effects, targets: Targets, label: string): Geometry {
  const parts = [
    ['engine', engineToStand(buildEngine(DEFAULT_ENGINE), AXIS_HEIGHT)],
    ['stand', buildStand(DEFAULT_ENGINE, AXIS_HEIGHT)],
    ['ground', buildGround()],
  ] as const;
  const meshes: Mesh[] = [];
  const draws: Draw[] = [];
  const shadowDraws: Draw[] = [];
  for (const [name, cad] of parts) {
    const data = pack(cad);
    const mesh = gpu.mesh({
      label: `${label}-${name}`,
      buffers: [
        { data: data.positions, attributes: { position: 'float32x3' } },
        { data: data.normals, attributes: { normal: 'float32x3' } },
        { data: data.uvs, attributes: { uv: 'float32x2' } },
        { data: data.materials, attributes: { material: 'float32' } },
      ],
      indices: data.indices,
    });
    meshes.push(mesh);
    draws.push(gpu.draw({
      shader: sceneWgsl,
      mesh,
      label: `${label}-${name}`,
      set: {
        detail: targets.detail, detailSamp: effects.repeatSampler, shadowMap: targets.shadow,
        concreteColor: targets.concrete, concreteNormal: targets.concrete.colors[1],
        gravelColor: targets.gravel, gravelNormal: targets.gravel.colors[1],
        atlasSamp: effects.clampSampler,
      },
    }));
    shadowDraws.push(gpu.draw({ shader: shadowWgsl, mesh, label: `${label}-${name}-shadow` }));
  }
  return { meshes, draws, shadowDraws };
}

function setConstants(effects: Effects, targets: Targets): void {
  const plume = { ...PLUME, axis: PLUME_AXIS };
  effects.bakeConcreteHeight.set({ mat: { kind: 0 } });
  effects.bakeConcrete.set({ mat: { kind: 0 }, height: targets.materialHeight });
  effects.bakeGravelHeight.set({ mat: { kind: 1 } });
  effects.bakeGravel.set({ mat: { kind: 1 }, height: targets.materialHeight });
  effects.bakeConcreteAtlas.set({ color: targets.materialTile, normal: targets.materialTile.colors[1] });
  effects.bakeGravelAtlas.set({ color: targets.materialTile, normal: targets.materialTile.colors[1] });
  effects.grid.set({
    params: { time: 0, motion: 1, frame: -1 },
    atlas: targets.noiseAtlas,
    detail: targets.detail,
    atlasSamp: effects.clampSampler,
    detailSamp: effects.repeatSampler,
    plume,
  });
  effects.fire.set({
    params: { time: 0, motion: 1, phase: 0, frame: 0 },
    detail: targets.detail,
    detailSamp: effects.repeatSampler,
    plume,
  });
  if (PLUME_MODE === 'grid') effects.fire.set({ plumeGrid: targets.plumeGrid, gridSamp: effects.clampSampler });
  else effects.fire.set({ atlas: targets.noiseAtlas, atlasSamp: effects.clampSampler });
  effects.resolve.set({ resolve: { phase: 0, blend: TEMPORAL_BLEND, neighbor: TEMPORAL_NEIGHBOR } });
  effects.brightPass.set({ samp: effects.clampSampler, bright: { threshold: 1.0, knee: 0.6 } });
  effects.blurH1.set({ samp: effects.clampSampler, blur: { direction: [1, 0], radius: 1 } });
  effects.blurV1.set({ samp: effects.clampSampler, blur: { direction: [0, 1], radius: 1 } });
  effects.blurH2.set({ samp: effects.clampSampler, blur: { direction: [1, 0], radius: 2.6 } });
  effects.blurV2.set({ samp: effects.clampSampler, blur: { direction: [0, 1], radius: 2.6 } });
  const variant = QUALITY[effects.quality];
  effects.composite.set({ samp: effects.clampSampler, composite: { exposure: 1.35, bloomStrength: 0.8, time: 0, skyColor: [0.004, 0.005, 0.01], ...variant.composite } });
  if (variant.ao) {
    effects.ao!.set({ ao: variant.ao });
    effects.aoBlur!.set({ blur: { direction: [1, 0] } });
  }
  if (variant.post) effects.post!.set({ samp: effects.clampSampler, post: { time: 0, ...variant.post } });
}

function setBindings(effects: Effects, geometry: Geometry, targets: Targets, camera: ThrusterCamera = CAMERA_PRESETS.default!): void {
  const [width, height] = targets.scene.size;
  const fov = ((camera.fovDeg ?? CAMERA.fovDeg) * Math.PI) / 180;
  const view = lookAt(camera.position, camera.target);
  const projection = perspective(fov, width / height, CAMERA.near, CAMERA.far);
  const viewProj = multiply(projection, view);
  const shadowViewProj = keyLightCamera();
  const shadowFov = Math.tan((SHADOW.fovDeg * Math.PI) / 360);
  for (const draw of geometry.draws) {
    draw.set({
      camera: { viewProj, position: camera.position, time: 0, pixelAngle: (2 * Math.tan(fov / 2)) / height },
      lighting: { ...LIGHTING, shadowTexel: 1 / SHADOW.size, shadowViewProj, shadowFov },
      plumeLight: { nozzle: PLUME.nozzle, axis: PLUME_AXIS, ...PLUME_LIGHT },
    });
  }
  for (const draw of geometry.shadowDraws) draw.set({ light: { viewProj: shadowViewProj, position: KEY_LIGHT } });
  const history = targets.fireHistory.read.size;
  effects.fire.set({
    params: { resolution: history, sceneScale: [width / history[0], height / history[1]] },
    camera: { invViewProj: invert(viewProj), position: camera.position },
    sceneDepth: targets.scene.colors[1],
  });
  effects.resolve.set({ marchFire: targets.march, marchAux: targets.march.colors[1] });
  // With occlusion, everything downstream of the scene reads the occluded copy.
  const lit = targets.sceneLit ?? targets.scene;
  if (effects.ao) {
    effects.ao.set({
      camera: { invViewProj: invert(viewProj), position: camera.position },
      ao: { projScale: height / (2 * Math.tan(fov / 2)) },
      sceneDepth: targets.scene.colors[1],
      sceneAux: targets.scene.colors[2],
    });
    effects.aoBlur!.set({ src: targets.ao!, sceneDepth: targets.scene.colors[1] });
    effects.aoApply!.set({ src: targets.aoBlur!, sceneDepth: targets.scene.colors[1], scene: targets.scene, sceneAux: targets.scene.colors[2] });
  }
  if (effects.post) effects.post.set({ src: targets.ldr! });
  effects.brightPass.set({ scene: lit });
  effects.blurH1.set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  effects.blurV1.set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  effects.blurH2.set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  effects.blurV2.set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  effects.composite.set({ scene: lit, sceneDepth: targets.scene.colors[1], bloom: targets.bloomA });
  setHistoryReaders(effects, targets);
}

/** Perspective camera at the key floodlight, looking where the lamp points. */
function keyLightCamera() {
  return multiply(perspective((SHADOW.fovDeg * Math.PI) / 180, 1, SHADOW.near, SHADOW.far), lookAt(KEY_LIGHT, KEY_TARGET));
}

function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function createHistory(gpu: Gpu, full: readonly [number, number], label: string): PingPongTargets {
  const [width, height] = fireSize(full);
  return gpu.pingPong(width, height, { colors: [{ format: HDR_FORMAT }, { format: HDR_FORMAT }], label: `${label}-fire-history` });
}

/** Rebinds everything that reads the plume history after a ping-pong swap. */
function setHistoryReaders(effects: Effects, targets: Targets): void {
  const history = targets.fireHistory;
  effects.resolve.set({ historyFire: history.read, historyAux: history.read.colors[1] });
  effects.brightPass.set({ fire: history.read });
  effects.composite.set({ fire: history.read, fireAux: history.read.colors[1] });
}

/** Per-frame uniforms: the clock, and which 2x2 phase this frame marches. */
function setFrame(effects: Effects, time: number, frameIndex: number): void {
  const phase = frameIndex & 3;
  effects.grid.set({ params: { time, frame: frameIndex } });
  effects.fire.set({ params: { time, phase, frame: frameIndex } });
  effects.resolve.set({ resolve: { phase } });
  effects.composite.set({ composite: { time } });
  effects.post?.set({ post: { time } });
}

async function prewarm(effects: Effects, geometry: Geometry, targets: Targets, output: Output): Promise<void> {
  await Promise.all([
    effects.bakeNoise.compile(targets.noiseAtlas), effects.bakeDetail.compile(targets.detail),
    effects.bakeConcreteHeight.compile(targets.materialHeight), effects.bakeConcrete.compile(targets.materialTile), effects.bakeConcreteAtlas.compile(targets.concrete),
    effects.bakeGravelHeight.compile(targets.materialHeight), effects.bakeGravel.compile(targets.materialTile), effects.bakeGravelAtlas.compile(targets.gravel),
    ...geometry.draws.map((draw) => draw.compile(targets.scene)),
    ...geometry.shadowDraws.map((draw) => draw.compile(targets.shadow)),
    effects.grid.compile(targets.plumeGrid), effects.fire.compile(targets.march), effects.resolve.compile(targets.fireHistory.write), effects.brightPass.compile(targets.bloomA),
    effects.blurH1.compile(targets.bloomB), effects.blurV1.compile(targets.bloomA),
    effects.blurH2.compile(targets.bloomB), effects.blurV2.compile(targets.bloomA),
    effects.composite.compile(targets.ldr ?? { colors: [output.format] }),
    ...(effects.ao ? [effects.ao.compile(targets.ao!), effects.aoBlur!.compile(targets.aoBlur!), effects.aoApply!.compile(targets.sceneLit!)] : []),
    ...(effects.post ? [effects.post.compile({ colors: [output.format] })] : []),
  ]);
}

/**
 * Bakes everything that never changes: the noise textures and the sun shadow
 * map (static light, static geometry). Called once after prewarm; the per-frame
 * chain only reads these targets.
 */
function bakeStatic(gpu: Gpu, effects: Effects, geometry: Geometry, targets: Targets): void {
  gpu.frame((frame) => {
    frame.pass({ target: targets.noiseAtlas, clear: CLEAR }, (pass) => pass.draw(effects.bakeNoise));
    frame.pass({ target: targets.detail, clear: CLEAR }, (pass) => pass.draw(effects.bakeDetail));
    // Ground materials: height + masks into the scratch target, then the finish
    // pass derives normals, cavity, albedo and roughness into a tile, and the
    // atlas pass packs the tile with its coarser levels. Once per material.
    frame.pass({ target: targets.materialHeight, clear: CLEAR }, (pass) => pass.draw(effects.bakeConcreteHeight));
    frame.pass({ target: targets.materialTile, clear: CLEAR }, (pass) => pass.draw(effects.bakeConcrete));
    frame.pass({ target: targets.concrete, clear: CLEAR }, (pass) => pass.draw(effects.bakeConcreteAtlas));
    frame.pass({ target: targets.materialHeight, clear: CLEAR }, (pass) => pass.draw(effects.bakeGravelHeight));
    frame.pass({ target: targets.materialTile, clear: CLEAR }, (pass) => pass.draw(effects.bakeGravel));
    frame.pass({ target: targets.gravel, clear: CLEAR }, (pass) => pass.draw(effects.bakeGravelAtlas));
    frame.pass({ target: targets.shadow, clear: [1, 0, 0, 1] }, (pass) => {
      for (const draw of geometry.shadowDraws) pass.draw(draw);
    });
    // First full fill of the plume grid; per-frame passes then refresh half
    // of the slices each and preserve the other half.
    frame.pass({ target: targets.plumeGrid, clear: [0, 0, 0, 0] }, (pass) => pass.draw(effects.grid));
  });
}

function renderChain(frame: Frame, effects: Effects, geometry: Geometry, targets: Targets, output: Output): void {
  // Depth attachment cleared to 0 in colors[1] means "no surface" for the plume.
  frame.pass({ target: targets.scene, clear: [0, 0, 0, 0] }, (pass) => {
    for (const draw of geometry.draws) pass.draw(draw);
  });
  renderOcclusion(frame, effects, targets);
  // Plume: evaluate the volume once into the grid, march one phase at quarter
  // resolution over it, interleave that into the half-resolution history, then
  // everything downstream reads the history.
  if (PLUME_MODE === 'grid') frame.pass({ target: targets.plumeGrid, clear: false }, (pass) => pass.draw(effects.grid));
  frame.pass({ target: targets.march, clear: CLEAR }, (pass) => pass.draw(effects.fire));
  frame.pass({ target: targets.fireHistory.write, clear: CLEAR }, (pass) => pass.draw(effects.resolve));
  targets.fireHistory.swap();
  setHistoryReaders(effects, targets);
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.brightPass));
  frame.pass({ target: targets.bloomB, clear: CLEAR }, (pass) => pass.draw(effects.blurH1));
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.blurV1));
  frame.pass({ target: targets.bloomB, clear: CLEAR }, (pass) => pass.draw(effects.blurH2));
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.blurV2));
  const post = effects.post;
  if (post) {
    frame.pass({ target: targets.ldr!, clear: CLEAR }, (pass) => pass.draw(effects.composite));
    frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(post));
  } else {
    frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(effects.composite));
  }
}

/**
 * Social pipeline: screen-space occlusion from the scene's distance and
 * normal attachments, blurred depth-aware, then multiplied into the lit
 * scene's occludable share. No-op in the fast pipeline.
 */
function renderOcclusion(frame: Frame, effects: Effects, targets: Targets): void {
  if (!effects.ao) return;
  frame.pass({ target: targets.ao!, clear: CLEAR }, (pass) => pass.draw(effects.ao!));
  frame.pass({ target: targets.aoBlur!, clear: CLEAR }, (pass) => pass.draw(effects.aoBlur!));
  frame.pass({ target: targets.sceneLit!, clear: CLEAR }, (pass) => pass.draw(effects.aoApply!));
}

function resizeTargets(gpu: Gpu, targets: Targets, size: readonly [number, number]): void {
  const full = normalizeSize(size);
  targets.scene.resize(full);
  for (const target of [targets.ao, targets.aoBlur, targets.sceneLit, targets.ldr]) target?.resize(full);
  targets.march.resize(marchSize(full));
  // Ping-pong targets do not resize: rebuild the history at the new size.
  destroyHistory(targets.fireHistory);
  targets.fireHistory = createHistory(gpu, full, 'thrusters-live');
  const bloom = bloomSize(full, QUALITY[RENDER_QUALITY].bloomHeight);
  targets.bloomA.resize(bloom);
  targets.bloomB.resize(bloom);
}

function normalizeSize(size: readonly [number, number]): [number, number] {
  return [Math.max(1, Math.floor(size[0])), Math.max(1, Math.floor(size[1]))];
}

function fireSize(size: readonly [number, number]): [number, number] {
  return [Math.max(1, Math.round(size[0] * FIRE_SCALE)), Math.max(1, Math.round(size[1] * FIRE_SCALE))];
}

/** Quarter of the history: one 2x2 phase per frame. */
function marchSize(size: readonly [number, number]): [number, number] {
  const history = fireSize(size);
  return [Math.max(1, Math.ceil(history[0] / 2)), Math.max(1, Math.ceil(history[1] / 2))];
}

function bloomSize(size: readonly [number, number], bloomHeight: number): [number, number] {
  const height = Math.max(1, Math.min(bloomHeight, size[1]));
  return [Math.max(1, Math.round(height * size[0] / size[1])), height];
}

function normalize3(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
