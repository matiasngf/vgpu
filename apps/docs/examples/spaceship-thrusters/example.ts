import type { Draw, Effect, Frame, Gpu, Surface, Target } from 'vgpu';

import bakeDetailWgsl from './bake-detail.wgsl';
import bakeNoiseWgsl from './bake-noise.wgsl';
import blurWgsl from './blur.wgsl';
import brightPassWgsl from './bright-pass.wgsl';
import compositeWgsl from './composite.wgsl';
import debugPreviewWgsl from './debug-preview.wgsl';
import fireWgsl from './fire.wgsl';
import sceneWgsl from './scene.wgsl';
import shadowWgsl from './shadow.wgsl';
import { invert, lookAt, multiply, orthographic, pack, perspective, type Vec3 } from './cad';
import { buildEngine, buildGantry, buildGround, buildStand, DEFAULT_ENGINE, engineToStand } from './engine';

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
const PLUME = { nozzle: [0, AXIS_HEIGHT, 0] as Vec3, r0: 0.93, spread: 0.03, length: 45, sootGain: 0.35, glowGain: 10, exitGain: 5 };
const CAMERA = { position: [-10, 15, 10] as Vec3, target: [0.8, 0.8, -1.2] as Vec3, fovDeg: 40, near: 0.5, far: 400 };
/** Orthographic sun camera covering the stand and the near plume. */
const SHADOW = { size: 2048, halfExtent: 9, center: [-2.5, 1, 0.5] as Vec3, distance: 60 };
const LIGHTING = {
  sunDir: normalize3([-0.35, 0.72, -0.6]),
  sunIntensity: 3.8,
  sunColor: [1.0, 0.96, 0.9],
  ambient: 0.28,
  skyColor: [0.6, 0.72, 0.9],
  groundColor: [0.42, 0.38, 0.33],
  /** In light-space NDC depth; the span is 4 * halfExtent world units, so this is ~0.03 units. */
  shadowBias: 0.0008,
};
/** Segment light that stands in for the plume's glow on the geometry. */
const PLUME_LIGHT = { length: 32, intensity: 14 };

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
  /** HDR fire pass, composited over the scene. */
  fire: Target;
  bloomA: Target;
  bloomB: Target;
}

// Must match the constants in thruster-common.wgsl.
const NOISE_ATLAS_SIZE = (128 + 2) * 8;
const DETAIL_SIZE = 512;
const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const FIRE_SCALE = 1.0; // TODO: optimization pass once the look is locked
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
    resizeTargets(targets, surface.size);
    setBindings(effects, geometry, targets);
  });
  bakeStatic(gpu, effects, geometry, targets);

  const handle = gpu.frame.loop((frame) => {
    // Only the clock changes per frame; every other binding is stable.
    effects.fire.set({ params: { time: gpu.time } });
    effects.composite.set({ composite: { time: gpu.time } });
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

  effects.fire.set({ params: { time } });
  effects.composite.set({ composite: { time } });
  gpu.frame((frame) => renderChain(frame, effects, geometry, targets, target));
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
    ['fire-hdr', targets.fire.color, targets.fire.size, { exposure: 1, mode: 0 }],
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
    fire: gpu.target({ size: fireSize(full), format: HDR_FORMAT, label: `${label}-fire` }),
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
    params: { time: 0, motion: 1 },
    atlas: targets.noiseAtlas,
    detail: targets.detail,
    atlasSamp: effects.clampSampler,
    detailSamp: effects.repeatSampler,
    plume: { ...PLUME, axis: PLUME_AXIS },
  });
  effects.brightPass.set({ samp: effects.clampSampler, bright: { threshold: 0.8, knee: 0.6 } });
  effects.blurH1.set({ samp: effects.clampSampler, blur: { direction: [1, 0], radius: 1 } });
  effects.blurV1.set({ samp: effects.clampSampler, blur: { direction: [0, 1], radius: 1 } });
  effects.blurH2.set({ samp: effects.clampSampler, blur: { direction: [1, 0], radius: 2.6 } });
  effects.blurV2.set({ samp: effects.clampSampler, blur: { direction: [0, 1], radius: 2.6 } });
  effects.composite.set({ samp: effects.clampSampler, composite: { exposure: 1.0, bloomStrength: 0.9, grain: 0.015, time: 0 } });
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
  effects.fire.set({
    params: { resolution: targets.fire.size, sceneScale: [width / targets.fire.size[0], height / targets.fire.size[1]] },
    camera: { invViewProj: invert(viewProj), position: CAMERA.position },
    sceneColor: targets.scene,
    sceneDepth: targets.scene.colors[1],
  });
  effects.brightPass.set({ src: targets.fire });
  effects.blurH1.set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  effects.blurV1.set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  effects.blurH2.set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  effects.blurV2.set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  effects.composite.set({ scene: targets.fire, bloom: targets.bloomA });
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

async function prewarm(effects: Effects, geometry: Geometry, targets: Targets, output: Output): Promise<void> {
  await Promise.all([
    effects.bakeNoise.compile(targets.noiseAtlas), effects.bakeDetail.compile(targets.detail),
    ...geometry.draws.map((draw) => draw.compile(targets.scene)),
    ...geometry.shadowDraws.map((draw) => draw.compile(targets.shadow)),
    effects.fire.compile(targets.fire), effects.brightPass.compile(targets.bloomA),
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
  targets.scene.resize(full);
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

function normalize3(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
