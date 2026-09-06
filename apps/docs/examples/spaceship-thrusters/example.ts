import type { Draw, Effect, Frame, Gpu, PingPongTargets, Surface, Target } from 'vgpu';

import bakeDetailWgsl from './bake-detail.wgsl';
import bakeNoiseWgsl from './bake-noise.wgsl';
import blurWgsl from './blur.wgsl';
import brightPassWgsl from './bright-pass.wgsl';
import compositeWgsl from './composite.wgsl';
import debugPreviewWgsl from './debug-preview.wgsl';
import fireWgsl from './fire.wgsl';
import resolveWgsl from './resolve.wgsl';
import sceneWgsl from './scene.wgsl';
import shadowWgsl from './shadow.wgsl';
import { invert, lookAt, multiply, orthographic, pack, perspective, type Vec3 } from './cad';
import { buildEngine, buildFloodlight, buildGantry, buildGround, buildStand, DEFAULT_ENGINE, engineToStand, WORK_LIGHT } from './engine';

type Output = Surface | Target;

export type ThrusterIntermediate = 'noise-atlas' | 'detail' | 'shadow-map' | 'scene-color' | 'scene-depth' | 'fire-hdr' | 'bloom';

export interface ThrusterThumbOptions {
  time?: number;
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
const PLUME = { nozzle: [0, AXIS_HEIGHT, 0] as Vec3, r0: 0.93, spread: 0.03, length: 45, sootGain: 0.2, glowGain: 10, exitGain: 5 };
const CAMERA = { position: [-10, 15, 10] as Vec3, target: [0.8, 0.8, -1.2] as Vec3, fovDeg: 40, near: 0.5, far: 400 };
/** Orthographic sun camera covering the stand and the near plume. */
const SHADOW = { size: 2048, halfExtent: 9, center: [-2.5, 1, 0.5] as Vec3, distance: 60 };
// Late dusk: a low, warm sun grazing in from behind the stand as a rim light,
// a dim blue sky, and the plume as the key light.
const LIGHTING = {
  sunDir: normalize3([0.55, 0.24, -0.7]),
  sunIntensity: 2.2,
  sunColor: [1.0, 0.5, 0.25],
  ambient: 0.2,
  skyColor: [0.18, 0.28, 0.62],
  groundColor: [0.1, 0.08, 0.09],
  fogColor: [0.05, 0.055, 0.1],
  fogDensity: 0.005,
  workLight: [...WORK_LIGHT, 130] as [number, number, number, number],
  workLightColor: [1.0, 0.85, 0.65],
  /** In light-space NDC depth; the span is 4 * halfExtent world units, so this is ~0.03 units. */
  shadowBias: 0.0008,
};
/** Segment light that stands in for the plume's glow on the geometry. */
const PLUME_LIGHT = { length: 32, intensity: 85 };

interface Effects {
  bakeNoise: Effect;
  bakeDetail: Effect;
  fire: Effect;
  resolve: Effect;
  brightPass: Effect;
  blurH1: Effect;
  blurV1: Effect;
  blurH2: Effect;
  blurV2: Effect;
  composite: Effect;
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
  /** Sun shadow map: light-space depth in r32float. */
  shadow: Target;
  /** Lit geometry: radiance in colors[0], camera distance in colors[1] (r32float), plus depth. */
  scene: Target;
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
const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const FIRE_SCALE = 0.5; // the plume is soft; the composite upsamples it depth-aware
/** Fresh-sample weight in the temporal resolve (1 = no history blending). */
const TEMPORAL_BLEND = 0.75;
/** How much a stale 2x2 phase leans on its block's fresh sample each frame. */
const TEMPORAL_NEIGHBOR = 0.3;
const BLOOM_HEIGHT = 240;
const CLEAR: readonly [number, number, number, number] = [0, 0, 0, 1];

export async function run(canvas: HTMLCanvasElement): Promise<() => void> {
  const { init } = await import('vgpu');
  const gpu = await init();
  const surface = gpu.surface(canvas, { dpr: [1, 1.5] });
  const effects = createEffects(gpu, 'thrusters-live');
  const targets = createTargets(gpu, surface.size, 'thrusters-live');
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
  const effects = createEffects(gpu, 'thrusters-thumb');
  const targets = createTargets(gpu, target.size, 'thrusters-thumb');
  const geometry = createGeometry(gpu, effects, targets, 'thrusters-thumb');
  const time = opts.time ?? 6.2;
  setConstants(effects, targets);
  setBindings(effects, geometry, targets);
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
): Promise<void> {
  const effects = createEffects(gpu, 'thrusters-sequence');
  const targets = createTargets(gpu, target.size, 'thrusters-sequence');
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
  passes: Record<'scene' | 'fire' | 'bloom' | 'composite', number>;
  frames: number;
  size: readonly [number, number];
  fireSize: readonly [number, number];
}

/**
 * Times each pass of the per-frame chain separately (submit + wait), so the
 * headless harness can compare optimizations. Baking is excluded.
 */
export async function profile(gpu: Gpu, target: Target, frames = 20, time = 6.2): Promise<ThrusterProfile> {
  const effects = createEffects(gpu, 'thrusters-profile');
  const targets = createTargets(gpu, target.size, 'thrusters-profile');
  const geometry = createGeometry(gpu, effects, targets, 'thrusters-profile');
  setConstants(effects, targets);
  setBindings(effects, geometry, targets);
  await prewarm(effects, geometry, targets, target);
  bakeStatic(gpu, effects, geometry, targets);
  await gpu.gpu.queue.onSubmittedWorkDone();

  const stages: Record<'scene' | 'fire' | 'bloom' | 'composite', (frame: Frame) => void> = {
    scene: (frame) => frame.pass({ target: targets.scene, clear: [0, 0, 0, 0] }, (pass) => { for (const draw of geometry.draws) pass.draw(draw); }),
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
    composite: (frame) => frame.pass({ target, clear: CLEAR }, (pass) => pass.draw(effects.composite)),
  };
  const samples: Record<string, number[]> = { scene: [], fire: [], bloom: [], composite: [] };
  for (let i = 0; i < frames + 2; i++) {
    setFrame(effects, time + i / 60, i);
    for (const [name, stage] of Object.entries(stages)) {
      const started = performance.now();
      gpu.frame(stage);
      await gpu.gpu.queue.onSubmittedWorkDone();
      if (i >= 2) samples[name]!.push(performance.now() - started); // skip warm-up frames
    }
  }
  const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] ?? 0;
  const result: ThrusterProfile = {
    passes: { scene: median(samples.scene!), fire: median(samples.fire!), bloom: median(samples.bloom!), composite: median(samples.composite!) },
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
  const preview = gpu.effect(debugPreviewWgsl, { label: 'thrusters-debug-preview' });
  const jobs = [
    ['shadow-map', targets.shadow.color, targets.shadow.size, { exposure: 1, mode: 2 }],
    ['scene-color', targets.scene.color, targets.scene.size, { exposure: 1, mode: 0 }],
    ['scene-depth', targets.scene.colors[1], targets.scene.size, { exposure: 60, mode: 2 }],
    ['fire-hdr', targets.fireHistory.read.color, targets.fireHistory.read.size, { exposure: 1, mode: 0 }],
    ['bloom', targets.bloomA.color, targets.bloomA.size, { exposure: 1, mode: 0 }],
  ] as const;
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

function createEffects(gpu: Gpu, label: string): Effects {
  return {
    bakeNoise: gpu.effect(bakeNoiseWgsl, { label: `${label}-bake-noise` }),
    bakeDetail: gpu.effect(bakeDetailWgsl, { label: `${label}-bake-detail` }),
    fire: gpu.effect(fireWgsl, { label: `${label}-fire` }),
    resolve: gpu.effect(resolveWgsl, { label: `${label}-resolve` }),
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
    shadow: gpu.target({ size: [SHADOW.size, SHADOW.size], format: 'r32float', depth: true, label: `${label}-shadow` }),
    scene: gpu.target({ size: full, colors: [{ format: HDR_FORMAT }, { format: 'r32float' }], depth: true, label: `${label}-scene` }),
    march: gpu.target({ size: marchSize(full), colors: [{ format: HDR_FORMAT }, { format: HDR_FORMAT }], label: `${label}-march` }),
    fireHistory: createHistory(gpu, full, label),
    bloomA: gpu.target({ size: bloomSize(full), format: HDR_FORMAT, label: `${label}-bloom-a` }),
    bloomB: gpu.target({ size: bloomSize(full), format: HDR_FORMAT, label: `${label}-bloom-b` }),
  };
}

/** Builds the parametric engine, stand and pad and uploads them as three draws. */
function createGeometry(gpu: Gpu, effects: Effects, targets: Targets, label: string): Geometry {
  const parts = [
    ['engine', engineToStand(buildEngine(DEFAULT_ENGINE), AXIS_HEIGHT)],
    ['stand', buildStand(DEFAULT_ENGINE, AXIS_HEIGHT)],
    ['gantry', buildGantry()],
    ['floodlight', buildFloodlight()],
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
      set: { detail: targets.detail, detailSamp: effects.repeatSampler, shadowMap: targets.shadow },
    }));
    shadowDraws.push(gpu.draw({ shader: shadowWgsl, mesh, label: `${label}-${name}-shadow` }));
  }
  return { meshes, draws, shadowDraws };
}

function setConstants(effects: Effects, targets: Targets): void {
  effects.fire.set({
    params: { time: 0, motion: 1, phase: 0, frame: 0 },
    atlas: targets.noiseAtlas,
    detail: targets.detail,
    atlasSamp: effects.clampSampler,
    detailSamp: effects.repeatSampler,
    plume: { ...PLUME, axis: PLUME_AXIS },
  });
  effects.resolve.set({ resolve: { phase: 0, blend: TEMPORAL_BLEND, neighbor: TEMPORAL_NEIGHBOR } });
  effects.brightPass.set({ samp: effects.clampSampler, bright: { threshold: 1.0, knee: 0.6 } });
  effects.blurH1.set({ samp: effects.clampSampler, blur: { direction: [1, 0], radius: 1 } });
  effects.blurV1.set({ samp: effects.clampSampler, blur: { direction: [0, 1], radius: 1 } });
  effects.blurH2.set({ samp: effects.clampSampler, blur: { direction: [1, 0], radius: 2.6 } });
  effects.blurV2.set({ samp: effects.clampSampler, blur: { direction: [0, 1], radius: 2.6 } });
  effects.composite.set({ samp: effects.clampSampler, composite: { exposure: 1.35, bloomStrength: 0.8, grain: 0.02, time: 0, skyColor: [0.05, 0.055, 0.1] } });
}

function setBindings(effects: Effects, geometry: Geometry, targets: Targets): void {
  const [width, height] = targets.scene.size;
  const view = lookAt(CAMERA.position, CAMERA.target);
  const projection = perspective((CAMERA.fovDeg * Math.PI) / 180, width / height, CAMERA.near, CAMERA.far);
  const viewProj = multiply(projection, view);
  const sunViewProj = sunCamera();
  for (const draw of geometry.draws) {
    draw.set({
      camera: { viewProj, position: CAMERA.position, time: 0 },
      lighting: { ...LIGHTING, shadowTexel: 1 / SHADOW.size, shadowExtent: 2 * SHADOW.halfExtent, sunViewProj },
      plumeLight: { nozzle: PLUME.nozzle, axis: PLUME_AXIS, ...PLUME_LIGHT },
    });
  }
  for (const draw of geometry.shadowDraws) draw.set({ light: { viewProj: sunViewProj } });
  const history = targets.fireHistory.read.size;
  effects.fire.set({
    params: { resolution: history, sceneScale: [width / history[0], height / history[1]] },
    camera: { invViewProj: invert(viewProj), position: CAMERA.position },
    sceneDepth: targets.scene.colors[1],
  });
  effects.resolve.set({ marchFire: targets.march, marchAux: targets.march.colors[1] });
  effects.brightPass.set({ scene: targets.scene });
  effects.blurH1.set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  effects.blurV1.set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  effects.blurH2.set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  effects.blurV2.set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  effects.composite.set({ scene: targets.scene, sceneDepth: targets.scene.colors[1], bloom: targets.bloomA });
  setHistoryReaders(effects, targets);
}

function sunCamera() {
  const eye: Vec3 = [
    SHADOW.center[0] + LIGHTING.sunDir[0] * SHADOW.distance,
    SHADOW.center[1] + LIGHTING.sunDir[1] * SHADOW.distance,
    SHADOW.center[2] + LIGHTING.sunDir[2] * SHADOW.distance,
  ];
  const e = SHADOW.halfExtent;
  // Keep the light-space depth span tight (scene is within ~2e of the centre)
  // so the NDC bias stays a small fraction of a world unit.
  return multiply(orthographic(-e, e, -e, e, SHADOW.distance - 2 * e, SHADOW.distance + 2 * e), lookAt(eye, SHADOW.center));
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
  effects.fire.set({ params: { time, phase, frame: frameIndex } });
  effects.resolve.set({ resolve: { phase } });
  effects.composite.set({ composite: { time } });
}

async function prewarm(effects: Effects, geometry: Geometry, targets: Targets, output: Output): Promise<void> {
  await Promise.all([
    effects.bakeNoise.compile(targets.noiseAtlas), effects.bakeDetail.compile(targets.detail),
    ...geometry.draws.map((draw) => draw.compile(targets.scene)),
    ...geometry.shadowDraws.map((draw) => draw.compile(targets.shadow)),
    effects.fire.compile(targets.march), effects.resolve.compile(targets.fireHistory.write), effects.brightPass.compile(targets.bloomA),
    effects.blurH1.compile(targets.bloomB), effects.blurV1.compile(targets.bloomA),
    effects.blurH2.compile(targets.bloomB), effects.blurV2.compile(targets.bloomA),
    effects.composite.compile({ colors: [output.format] }),
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
    frame.pass({ target: targets.shadow, clear: [1, 0, 0, 1] }, (pass) => {
      for (const draw of geometry.shadowDraws) pass.draw(draw);
    });
  });
}

function renderChain(frame: Frame, effects: Effects, geometry: Geometry, targets: Targets, output: Output): void {
  // Depth attachment cleared to 0 in colors[1] means "no surface" for the plume.
  frame.pass({ target: targets.scene, clear: [0, 0, 0, 0] }, (pass) => {
    for (const draw of geometry.draws) pass.draw(draw);
  });
  // Plume: march one phase at quarter resolution, interleave it into the
  // half-resolution history, then everything downstream reads the history.
  frame.pass({ target: targets.march, clear: CLEAR }, (pass) => pass.draw(effects.fire));
  frame.pass({ target: targets.fireHistory.write, clear: CLEAR }, (pass) => pass.draw(effects.resolve));
  targets.fireHistory.swap();
  setHistoryReaders(effects, targets);
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.brightPass));
  frame.pass({ target: targets.bloomB, clear: CLEAR }, (pass) => pass.draw(effects.blurH1));
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.blurV1));
  frame.pass({ target: targets.bloomB, clear: CLEAR }, (pass) => pass.draw(effects.blurH2));
  frame.pass({ target: targets.bloomA, clear: CLEAR }, (pass) => pass.draw(effects.blurV2));
  frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(effects.composite));
}

function resizeTargets(gpu: Gpu, targets: Targets, size: readonly [number, number]): void {
  const full = normalizeSize(size);
  targets.scene.resize(full);
  targets.march.resize(marchSize(full));
  // Ping-pong targets do not resize: rebuild the history at the new size.
  destroyHistory(targets.fireHistory);
  targets.fireHistory = createHistory(gpu, full, 'thrusters-live');
  targets.bloomA.resize(bloomSize(full));
  targets.bloomB.resize(bloomSize(full));
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

function bloomSize(size: readonly [number, number]): [number, number] {
  const height = Math.max(1, Math.min(BLOOM_HEIGHT, size[1]));
  return [Math.max(1, Math.round(height * size[0] / size[1])), height];
}

function normalize3(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
