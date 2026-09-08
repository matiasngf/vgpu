import type { Draw, Effect, Frame, Gpu, Surface, Target } from 'vgpu';

import bloomBlurWgsl from './bloom-blur.wgsl';
import bloomDownWgsl from './bloom-down.wgsl';
import bloomUpWgsl from './bloom-up.wgsl';
import brightPassWgsl from './bright-pass.wgsl';
import grassBladesWgsl from './grass-blades.wgsl';
import { createGrassTile, renderGrassTile, TILE_HEIGHT, type GrassTile } from './grass-tile';
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
  /** Samples per pixel: 1 for live rendering, 4 for quick stills, 9 or 16 for final renders. */
  samples?: 1 | 4 | 9 | 16;
  /** Full-quality grass march (96 steps); off drops to 28 steps for a cheaper live frame. */
  grassShadows?: boolean;
  /** Wind amplitude for the blades (0 for stills). */
  wind?: number;
  /**
   * Geometric blades near the camera, as an instance count (0 or omitted keeps the relief
   * only). Stills use several hundred thousand; the relief steps aside inside their zone.
   */
  blades?: number;
  /**
   * Debug views of the sand world behind the door: 1 = free camera at a door-local position,
   * 2 = top-down map, 3 = the main camera inside the sand world (door and opening marked),
   * 4 = the same without overlays, 5 = the geometric blade G-buffer (distance as grey).
   */
  debug?: { mode: 1 | 2 | 3 | 4 | 5; camera?: readonly [number, number, number] };
  /** Camera and door overrides on top of LOOK, for exploring alternative framings. */
  look?: DreamcoreLookOverrides;
}

export interface DreamcoreLookOverrides {
  camera?: Partial<{ height: number; pitch: number; fovY: number }>;
  door?: Partial<{ x: number; z: number; yaw: number; leaf: number }>;
  /** Flat sand of the sand world: where it starts falling away (m behind the sill), fall (tan), hollow depth (m, negative). */
  plain?: Partial<{ tiltFrom: number; tilt: number; far: number }>;
  /** The near dune beyond the hollow: start (m behind the sill), face slope (tan), crest (m), toe line skew (tan). */
  dune?: Partial<{ start: number; slope: number; crest: number; skew: number }>;
  /** Wind ripples on the flat sand: amplitude (m), wavelength (m), distance from the camera where they have faded (m), crest position (0..1 of the period, low = steep side toward the door). */
  sand?: Partial<{ rippleAmp: number; rippleLen: number; rippleFade: number; rippleCrest: number }>;
  /** Photographic finish: exposure multiplier, bloom mix, grain and vignette strength. */
  post?: Partial<{ exposure: number; bloomStrength: number; grain: number; vignette: number }>;
}

/** One level of the bloom mip chain: its own blurred glow and the glow gathered from below. */
interface BloomLevel {
  down: Effect | null;   // null on level 0, which the bright pass fills
  blurH: Effect;
  blurV: Effect;
  up: Effect | null;     // null on the smallest level, which has nothing below it
}

interface Effects {
  scene: Effect;
  grassTile: GrassTile;
  blades: Draw;
  bladesShadow: Draw;   // the same blades, rasterised from the door light
  brightPass: Effect;
  bloom: BloomLevel[];
  post: Effect;
  sampler: GPUSampler;
}

interface BloomTargets {
  glow: Target;   // this level's blurred glow
  temp: Target;   // half of the separable blur
  acc: Target;    // glow gathered from this level down
}

interface Targets {
  scene: Target;
  gbuffer: Target;     // geometric blades: distance + tangent, albedo + position along the blade
  shadowMap: Target;   // the blades from the door light: distance from it
  bloom: BloomTargets[];
}

/**
 * Unreal-style bloom: the bright pass lands at half resolution, then a chain of ever smaller
 * mips is downsampled, blurred one texel per tap, and summed back up through tent filters,
 * so the halo reaches across the frame without any pass skipping pixels.
 */
const BLOOM_LEVELS = 5;
const BLOOM_SIGMA = 2.6;
/** Weight of each level's own glow when summing back up; the small mips carry the wide halo. */
const BLOOM_WEIGHTS = [0.45, 0.3, 0.25, 0.2, 0.2];   // per level, summing to about 1.4 so a large bright area does not pile up its own glow

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const CLEAR: readonly [number, number, number, number] = [0, 0, 0, 1];
const CLEAR_ZERO: readonly [number, number, number, number] = [0, 0, 0, 0];

/**
 * Geometric blades fill the camera's view wedge: full density from `inner` to `knee` metres,
 * then thinning as 1/r^2 out to `outer`, with `nearFraction` of the instances inside the knee.
 */
const BLADE_ZONE = { inner: 2.5, knee: 10, outer: 40, nearFraction: 0.25 } as const;
const BLADE_VERTICES = 36;   // six quads between seven spine samples
/** Paraboloid shadow map of the blades from the door light: size in pixels, light height above the sill, depth bias. */
const BLADE_SHADOW = { size: 3072, height: 1.04, bias: 0.03 } as const;

/** Scene constants measured against the reference photos; see scene.wgsl for the units. */
export const LOOK = {
  /** Standing eye height, level horizon; same vertical FOV for any aspect. */
  camera: { height: 1.5, pitch: 0.02, fovY: 0.733 },
  door: { x: 0, z: 11, yaw: 0, leaf: 2.5 },
  sun: { azimuth: -1.15, elevation: 0.72 },
  texture: 1,
  doorLight: 10,
  /** Blade patch around the door (radius in metres) and tallest blade height. */
  grass: { radius: 60, height: 0.32 },
  /** Rippled flat sand falling into a hollow, a backlit dune running in from the right, dunes growing to the horizon. */
  plain: { tiltFrom: 3.0, tilt: 0.06, far: -3.0 },
  dune: { start: 36, slope: 0.45, crest: 2.0, skew: 1.0 },
  sand: { rippleAmp: 0.04, rippleLen: 0.35, rippleFade: 26, rippleCrest: 0.32 },
  post: { exposure: 1.8, bloomStrength: 0.95, grain: 0.02, vignette: 0.3, nightThreshold: 0.16, dayThreshold: 0.7, knee: 0.1 },
} as const;

/** Night holds, the day sweeps out of the door, holds, then the night flows back in. */
export const CYCLE_SECONDS = 16;

/** Default free-camera position (door-local metres) for the sand-world debug view. */
const DEBUG_CAMERA: readonly [number, number, number] = [-11, 4.5, -7];

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
  gpu.frame((frame) => renderGrassTile(frame, effects.grassTile));

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
    const live: DreamcoreFrameOptions = { phase: phaseAt(gpu.time), time: gpu.time, samples: 1, grassShadows: false, wind: 1 };
    setFrame(effects, live, surface.size);
    setSample(effects, live, [0.5, 0.5], 1, surface.size);
    renderSample(frame, effects, targets, 0, true);
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
  gpu.frame((frame) => renderGrassTile(frame, effects.grassTile));
  const time = opts.time ?? 4.6;
  renderFrame(gpu, effects, targets, target, { phase: phaseAt(time), time, samples: 4, blades: 250000 });
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
  gpu.frame((frame) => renderGrassTile(frame, effects.grassTile));
  renderFrame(gpu, effects, targets, target, frameOpts);
  await gpu.gpu.queue.onSubmittedWorkDone();
  await gpu.settled();
}

function createEffects(gpu: Gpu, label: string): Effects {
  return {
    // The scene adds its sub-pixel samples into the HDR target, one pass each.
    scene: gpu.effect(sceneWgsl, { label: `${label}-scene`, blend: 'additive' }),
    grassTile: createGrassTile(gpu, 2048, `${label}-grass-tile`),
    blades: gpu.draw({ shader: grassBladesWgsl, label: `${label}-blades`, vertices: BLADE_VERTICES }),
    bladesShadow: gpu.draw({ shader: grassBladesWgsl, label: `${label}-blades-shadow`, vertices: BLADE_VERTICES }),
    brightPass: gpu.effect(brightPassWgsl, { label: `${label}-bright-pass` }),
    // Every pass owns its effect, and so its uniform buffer; sharing one would make each
    // pass observe the last values written in the frame.
    bloom: Array.from({ length: BLOOM_LEVELS }, (_, i) => ({
      down: i === 0 ? null : gpu.effect(bloomDownWgsl, { label: `${label}-bloom-down-${i}` }),
      blurH: gpu.effect(bloomBlurWgsl, { label: `${label}-bloom-blur-h-${i}` }),
      blurV: gpu.effect(bloomBlurWgsl, { label: `${label}-bloom-blur-v-${i}` }),
      up: i === BLOOM_LEVELS - 1 ? null : gpu.effect(bloomUpWgsl, { label: `${label}-bloom-up-${i}` }),
    })),
    post: gpu.effect(postWgsl, { label: `${label}-post` }),
    sampler: gpu.sampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' }),
  };
}

function createTargets(gpu: Gpu, size: readonly [number, number], label: string): Targets {
  const full = normalizeSize(size);
  const bloom: BloomTargets[] = [];
  let level = halfSize(full);
  for (let i = 0; i < BLOOM_LEVELS; i++) {
    bloom.push({
      glow: gpu.target({ size: level, format: HDR_FORMAT, label: `${label}-bloom-${i}-glow` }),
      temp: gpu.target({ size: level, format: HDR_FORMAT, label: `${label}-bloom-${i}-temp` }),
      acc: gpu.target({ size: level, format: HDR_FORMAT, label: `${label}-bloom-${i}-acc` }),
    });
    level = halfSize(level);
  }
  return {
    scene: gpu.target({ size: full, format: HDR_FORMAT, label: `${label}-scene` }),
    gbuffer: gpu.target({ size: full, colors: [{ format: 'rgba32float' }, { format: 'rgba16float' }], depth: true, label: `${label}-blades` }),
    shadowMap: gpu.target({ size: [BLADE_SHADOW.size, BLADE_SHADOW.size], colors: [{ format: 'rgba32float' }, { format: 'rgba8unorm' }], depth: true, label: `${label}-blades-shadow` }),
    bloom,
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
      debug: [0, 0, 0, 0],
      plain: [LOOK.plain.tiltFrom, LOOK.plain.tilt, 0, LOOK.plain.far],
      dune: [LOOK.dune.start, LOOK.dune.slope, LOOK.dune.crest, LOOK.dune.skew],
      sand: [LOOK.sand.rippleAmp, LOOK.sand.rippleLen, LOOK.sand.rippleFade, LOOK.sand.rippleCrest],
      blades: [0.5, 0.5, 0, 1],
      shadow: [0, BLADE_SHADOW.height + 0.02, BLADE_SHADOW.bias, 0],
    },
  });
  for (const [draw, light] of [[effects.blades, 0], [effects.bladesShadow, 1]] as const) {
    draw.set({
      blades: {
        camera: [camera.height, camera.pitch, camera.fovY, 1],
        jitter: [0, 0, 0, 0],
        door: [door.x, door.z, grass.radius, grass.height / TILE_HEIGHT],
        zone: [BLADE_ZONE.inner, BLADE_ZONE.knee, BLADE_ZONE.outer, BLADE_ZONE.nearFraction],
        count: [0, 0, 0, 0],
        light: [light, BLADE_SHADOW.height, 0, 0],
      },
    });
  }
  effects.brightPass.set({ samp: effects.sampler, bright: { threshold: post.nightThreshold, knee: post.knee } });
  effects.bloom.forEach((level, i) => {
    level.down?.set({ samp: effects.sampler });
    level.blurH.set({ samp: effects.sampler, blur: { direction: [1, 0], sigma: BLOOM_SIGMA } });
    level.blurV.set({ samp: effects.sampler, blur: { direction: [0, 1], sigma: BLOOM_SIGMA } });
    level.up?.set({ samp: effects.sampler, up: { weight: BLOOM_WEIGHTS[i] ?? 1, _pad: 0 } });
  });
  effects.post.set({
    samp: effects.sampler,
    post: { exposure: post.exposure, bloomStrength: post.bloomStrength, grain: post.grain, vignette: post.vignette, seed: 0.37, _pad: 0 },
  });
}

function setBindings(effects: Effects, targets: Targets): void {
  const tile = effects.grassTile.target;
  effects.scene.set({
    params: { resolution: targets.scene.size },
    tileColor: tile.colors[0],
    tileTangent: tile.colors[1]!,
    tileSamp: effects.sampler,
    bladeDist: targets.gbuffer.colors[0],
    bladeColor: targets.gbuffer.colors[1]!,
    bladeShadow: targets.shadowMap.colors[0],
  });
  effects.brightPass.set({ src: targets.scene });
  effects.bloom.forEach((level, i) => {
    const mine = targets.bloom[i]!;
    const above = targets.bloom[i - 1];
    const below = targets.bloom[i + 1];
    if (level.down && above) level.down.set({ src: above.glow, down: { texelSize: above.glow.texelSize } });
    level.blurH.set({ src: mine.glow, blur: { texelSize: mine.glow.texelSize } });
    level.blurV.set({ src: mine.temp, blur: { texelSize: mine.temp.texelSize } });
    if (level.up && below) level.up.set({ own: mine.glow, smaller: below.acc, up: { texelSize: below.acc.texelSize } });
  });
  const last = targets.bloom[BLOOM_LEVELS - 1]!;
  effects.post.set({ scene: targets.scene, bloom: targets.bloom[0]!.acc, post: { resolution: targets.scene.size } });
  void last;
}

function setFrame(effects: Effects, frame: DreamcoreFrameOptions, size: readonly [number, number]): void {
  const phase = Math.min(1, Math.max(0, frame.phase));
  const { grass } = LOOK;
  const post = { ...LOOK.post, ...frame.look?.post };
  const camera = { ...LOOK.camera, ...frame.look?.camera };
  const bladeCount = bladeCountOf(frame);
  const door = { ...LOOK.door, ...frame.look?.door };
  const plain = { ...LOOK.plain, ...frame.look?.plain };
  const dune = { ...LOOK.dune, ...frame.look?.dune };
  const sand = { ...LOOK.sand, ...frame.look?.sand };
  effects.scene.set({
    params: {
      time: frame.time ?? 0,
      phase,
      camera: [camera.height, camera.pitch, camera.fovY, frame.samples ?? 1],
      door: [door.x, door.z, door.yaw, door.leaf],
      grass: [grass.radius, grass.height, frame.grassShadows === false ? 0 : 1, frame.wind ?? 0],
      debug: frame.debug ? [frame.debug.mode, ...(frame.debug.camera ?? DEBUG_CAMERA)] : [0, 0, 0, 0],
      plain: [plain.tiltFrom, plain.tilt, 0, plain.far],
      dune: [dune.start, dune.slope, dune.crest, dune.skew],
      sand: [sand.rippleAmp, sand.rippleLen, sand.rippleFade, sand.rippleCrest],
      blades: [0.5, 0.5, bladeCount > 0 ? BLADE_ZONE.outer : 0, 1],
      // The sill sits 2 cm under the terrain; the light height is measured from the terrain.
      shadow: [bladeCount > 0 ? BLADE_SHADOW.size : 0, BLADE_SHADOW.height + 0.02, BLADE_SHADOW.bias, 0],
    },
  });
  // Both blade draws must place the blades identically: same camera wedge, same count.
  for (const draw of [effects.blades, effects.bladesShadow]) {
    draw.set({
      blades: {
        camera: [camera.height, camera.pitch, camera.fovY, size[0] / Math.max(1, size[1])],
        jitter: [0, 0, frame.wind ?? 0, frame.time ?? 0],
        door: [door.x, door.z, grass.radius, grass.height / TILE_HEIGHT],
        count: [bladeCount, 0, 0, 0],
      },
    });
  }
  // The door only needs to bloom at night; by day the threshold rises so the field stays crisp.
  effects.brightPass.set({ bright: { threshold: post.nightThreshold + (post.dayThreshold - post.nightThreshold) * phase } });
  // Debug views skip the photographic finish so they stay readable.
  const finish = frame.debug ? { bloomStrength: 0, grain: 0, vignette: 0 } : { bloomStrength: post.bloomStrength, grain: post.grain, vignette: post.vignette };
  effects.post.set({ post: { exposure: post.exposure, seed: 0.37 + (frame.time ?? 0) * 0.01, ...finish } });
}

async function prewarm(effects: Effects, targets: Targets, output: Output): Promise<void> {
  const level0 = targets.bloom[0]!;
  await Promise.all([
    effects.scene.compile(targets.scene), effects.brightPass.compile(level0.glow), effects.grassTile.draw.compile(effects.grassTile.target),
    effects.blades.compile(targets.gbuffer), effects.bladesShadow.compile(targets.shadowMap),
    ...effects.bloom.flatMap((level, i) => {
      const mine = targets.bloom[i]!;
      return [level.down?.compile(mine.glow), level.blurH.compile(mine.temp), level.blurV.compile(mine.glow), level.up?.compile(mine.acc)];
    }),
    effects.post.compile({ colors: [output.format] }),
  ]);
}

/** One sub-pixel sample: the geometric blades into the G-buffer, then the scene added into the HDR target. */
function renderSample(frame: Frame, effects: Effects, targets: Targets, bladeCount: number, first: boolean): void {
  if (bladeCount > 0) {
    frame.pass({ target: targets.gbuffer, clear: CLEAR_ZERO }, (pass) => pass.draw(effects.blades, { instances: bladeCount, vertices: BLADE_VERTICES }));
  }
  frame.pass({ target: targets.scene, clear: first ? CLEAR_ZERO : false }, (pass) => pass.draw(effects.scene));
}

/** Which sub-pixel sample this pass renders and how much of the final pixel it is. */
function setSample(effects: Effects, frame: DreamcoreFrameOptions, offset: readonly [number, number], weight: number, size: readonly [number, number]): void {
  const zone = bladeCountOf(frame) > 0 ? BLADE_ZONE.outer : 0;
  effects.scene.set({ params: { blades: [offset[0], offset[1], zone, weight] } });
  // The rasteriser samples pixel centres; shift its projection so they land on this sample.
  effects.blades.set({ blades: { jitter: [((offset[0] - 0.5) * 2) / size[0], ((offset[1] - 0.5) * 2) / size[1], frame.wind ?? 0, frame.time ?? 0] } });
}

function bladeCountOf(frame: DreamcoreFrameOptions): number {
  return frame.debug && frame.debug.mode !== 5 ? 0 : Math.max(0, Math.floor(frame.blades ?? 0));
}

/** The bloom chain and the photographic finish, from the accumulated HDR scene. */
function renderChain(frame: Frame, effects: Effects, targets: Targets, output: Output): void {
  // Down the chain: bright pass into level 0, then downsample and blur each level.
  frame.pass({ target: targets.bloom[0]!.glow, clear: CLEAR }, (pass) => pass.draw(effects.brightPass));
  effects.bloom.forEach((level, i) => {
    const mine = targets.bloom[i]!;
    if (level.down) frame.pass({ target: mine.glow, clear: CLEAR }, (pass) => pass.draw(level.down!));
    frame.pass({ target: mine.temp, clear: CLEAR }, (pass) => pass.draw(level.blurH));
    frame.pass({ target: mine.glow, clear: CLEAR }, (pass) => pass.draw(level.blurV));
  });
  // Back up: the smallest level is its own accumulation, every other adds the one below.
  const smallest = targets.bloom[BLOOM_LEVELS - 1]!;
  frame.pass({ target: smallest.acc, clear: CLEAR }, (pass) => pass.draw(effects.bloom[BLOOM_LEVELS - 1]!.blurV));
  for (let i = BLOOM_LEVELS - 2; i >= 0; i--) {
    frame.pass({ target: targets.bloom[i]!.acc, clear: CLEAR }, (pass) => pass.draw(effects.bloom[i]!.up!));
  }
  frame.pass({ target: output, clear: CLEAR }, (pass) => pass.draw(effects.post));
}

/**
 * A still: an n x n grid of sub-pixel samples, each its own frame (the blades rasterised
 * and the scene ray marched at that offset, added into the HDR target), then the finish.
 */
function renderFrame(gpu: Gpu, effects: Effects, targets: Targets, output: Target, frameOpts: DreamcoreFrameOptions): void {
  const size = targets.scene.size;
  setFrame(effects, frameOpts, size);
  const bladeCount = bladeCountOf(frameOpts);
  if (bladeCount > 0) {
    // The blades from the door light, once per still: their shadows on each other.
    gpu.frame((frame) => frame.pass({ target: targets.shadowMap, clear: CLEAR_ZERO }, (pass) => pass.draw(effects.bladesShadow, { instances: bladeCount, vertices: BLADE_VERTICES })));
  }
  const n = Math.max(1, Math.round(Math.sqrt(frameOpts.samples ?? 1)));
  const total = n * n;
  for (let i = 0; i < total; i++) {
    const offset: [number, number] = [((i % n) + 0.5) / n, (Math.floor(i / n) + 0.5) / n];
    setSample(effects, frameOpts, offset, 1 / total, size);
    gpu.frame((frame) => renderSample(frame, effects, targets, bladeCount, i === 0));
  }
  gpu.frame((frame) => renderChain(frame, effects, targets, output));
}

function resizeTargets(targets: Targets, size: readonly [number, number]): void {
  const full = normalizeSize(size);
  targets.scene.resize(full);
  targets.gbuffer.resize(full);
  let level = halfSize(full);
  for (const mip of targets.bloom) {
    mip.glow.resize(level);
    mip.temp.resize(level);
    mip.acc.resize(level);
    level = halfSize(level);
  }
}

function normalizeSize(size: readonly [number, number]): [number, number] {
  return [Math.max(1, Math.floor(size[0])), Math.max(1, Math.floor(size[1]))];
}

function halfSize(size: readonly [number, number]): [number, number] {
  return [Math.max(1, Math.ceil(size[0] / 2)), Math.max(1, Math.ceil(size[1] / 2))];
}
