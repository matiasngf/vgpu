import type { Draw, Effect, Frame, Gpu, MeshLike, Surface, Target } from 'vgpu';
import { perspectiveCamera, sphere } from 'vgpu/scene';

import { installControls } from './controls';
import {
  bloomSize,
  cameraBasis,
  EARTH_TUNING,
  normalizeSize,
  orbitPosition,
  sunDegreesAt,
  sunDirection,
  type OrbitState,
} from './planet';

import atmosphereWgsl from './atmosphere.wgsl';
import bakeCloudsWgsl from './bake-clouds.wgsl';
import bakeSurfaceWgsl from './bake-surface.wgsl';
import blurWgsl from './blur.wgsl';
import brightPassWgsl from './bright-pass.wgsl';
import compositeWgsl from './composite.wgsl';
import earthWgsl from './earth.wgsl';
import overlayWgsl from './overlay.wgsl';
import skyWgsl from './sky.wgsl';

type Output = Surface | Target;

interface ThumbOptions {
  warmupFrames?: number;
  dt?: number;
  time?: number;
}

interface Maps {
  /** Equirectangular day albedo in rgb, night-light mask in alpha. */
  readonly surface: Target;
  /** Single-channel cloud coverage. */
  readonly clouds: Target;
}

interface Scene {
  readonly earthMesh: MeshLike;
  readonly atmosphereMesh: MeshLike;
  readonly earth: Draw;
  readonly atmosphere: Draw;
  readonly sky: Effect;
  readonly overlay: Effect;
  readonly bright: Effect;
  readonly blur: readonly [Effect, Effect, Effect, Effect];
  readonly composite: Effect;
  readonly mapSampler: GPUSampler;
  readonly linearSampler: GPUSampler;
}

interface Targets {
  readonly beauty: Target;
  readonly planet: Target;
  readonly bloomA: Target;
  readonly bloomB: Target;
}

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
/** The planet is clamped to 0.7 by its own shader, so 8-bit sRGB is plenty and keeps MSAA available. */
const PLANET_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';
const OPAQUE_BLACK = [0, 0, 0, 1] as const;
const TRANSPARENT = [0, 0, 0, 0] as const;

export async function run(canvas: HTMLCanvasElement): Promise<() => void> {
  const { init } = await import('vgpu');
  const gpu = await init();
  const surface = gpu.surface(canvas, { dpr: [1, 2] });
  const maps = createMaps(gpu, 'earth-live');
  const scene = createScene(gpu, maps, 'earth-live');
  const targets = createTargets(gpu, surface.size, 'earth-live');
  const start: OrbitState = {
    yaw: 0,
    pitch: EARTH_TUNING.poster.pitch,
    radius: EARTH_TUNING.camera.radius,
  };
  const controls = installControls(canvas, start);
  let disposed = false;

  setStaticBindings(scene, maps, targets);
  await Promise.all([bakeMaps(gpu, maps), prewarm(scene, targets, surface)]);

  let sawInitialResize = false;
  const unsubscribeResize = surface.onResize(() => {
    if (!sawInitialResize) {
      sawInitialResize = true;
      return;
    }
    if (disposed) return;
    resizeTargets(targets, surface.size);
    setSizeBindings(scene, targets);
  });

  const handle = gpu.frame.loop((frame) => {
    const deltaTime = Math.min(0.05, gpu.deltaTime);
    setFrameUniforms(scene, surface, controls.step(deltaTime), controls.sunDegrees(deltaTime), gpu.time);
    render(frame, scene, targets, surface);
  });

  return () => {
    if (disposed) return;
    disposed = true;
    handle.stop();
    unsubscribeResize();
    controls.dispose();
    destroyScene(scene);
    destroyTargets(targets);
    destroyMaps(maps);
    surface.dispose();
    gpu.dispose();
  };
}

export async function renderThumb(gpu: Gpu, output: Target, opts: ThumbOptions = {}): Promise<void> {
  const maps = createMaps(gpu, 'earth-thumb');
  const scene = createScene(gpu, maps, 'earth-thumb');
  const targets = createTargets(gpu, output.size, 'earth-thumb');
  setStaticBindings(scene, maps, targets);
  await Promise.all([bakeMaps(gpu, maps), prewarm(scene, targets, output)]);

  // The poster framing is fixed; only `time` moves, and the sun angle is derived
  // from it so a given `thumb.time` always produces the same terminator.
  const { yaw, pitch, radius, sunDegrees } = EARTH_TUNING.poster;
  const dt = opts.dt ?? 1 / 60;
  let time = opts.time ?? 0;
  for (let i = 0; i < Math.max(1, opts.warmupFrames ?? 1); i++) {
    setFrameUniforms(scene, output, { yaw, pitch, radius }, sunDegrees + sunDegreesAt(time), time);
    gpu.frame((frame) => render(frame, scene, targets, output));
    time += dt;
  }

  await gpu.gpu.queue.onSubmittedWorkDone();
  await gpu.settled();
  destroyScene(scene);
  destroyTargets(targets);
  destroyMaps(maps);
}

function createMaps(gpu: Gpu, label: string): Maps {
  const size = EARTH_TUNING.maps.size;
  return {
    // `-srgb` stores the albedo with a transfer curve (precision where the oceans
    // are dark) and hands linear values back to `textureSample`; alpha stays linear.
    surface: gpu.target({ size, format: PLANET_FORMAT, label: `${label}-surface-map` }),
    clouds: gpu.target({ size, format: 'r8unorm', label: `${label}-cloud-map` }),
  };
}

function createScene(gpu: Gpu, maps: Maps, label: string): Scene {
  const earthMesh = gpu.mesh(sphere(EARTH_TUNING.planet));
  const atmosphereMesh = gpu.mesh(sphere(EARTH_TUNING.atmosphere));
  return {
    earthMesh,
    atmosphereMesh,
    earth: gpu.draw({ shader: earthWgsl, mesh: earthMesh, label: `${label}-earth` }),
    // `alpha` blending over the transparent clear is what turns the shell's
    // fresnel alpha into the rim glow, and leaves coverage in the target's alpha.
    atmosphere: gpu.draw({ shader: atmosphereWgsl, mesh: atmosphereMesh, blend: 'alpha', label: `${label}-atmosphere` }),
    sky: gpu.effect(skyWgsl, { label: `${label}-sky` }),
    overlay: gpu.effect(overlayWgsl, { blend: 'premultiplied', label: `${label}-overlay` }),
    bright: gpu.effect(brightPassWgsl, { label: `${label}-bright` }),
    // One effect per blur pass: sharing a single effect would make every encoded
    // pass observe the last direction and radius written to its uniform buffer.
    blur: [
      gpu.effect(blurWgsl, { label: `${label}-blur-h1` }),
      gpu.effect(blurWgsl, { label: `${label}-blur-v1` }),
      gpu.effect(blurWgsl, { label: `${label}-blur-h2` }),
      gpu.effect(blurWgsl, { label: `${label}-blur-v2` }),
    ],
    composite: gpu.effect(compositeWgsl, { label: `${label}-composite` }),
    mapSampler: gpu.sampler({
      minFilter: 'linear',
      magFilter: 'linear',
      // Longitude wraps, latitude does not.
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
    }),
    linearSampler: gpu.sampler({ minFilter: 'linear', magFilter: 'linear' }),
  };
}

function createTargets(gpu: Gpu, size: readonly [number, number], label: string): Targets {
  const full = normalizeSize(size);
  const bloom = bloomSize(full);
  return {
    beauty: gpu.target({ size: full, format: HDR_FORMAT, label: `${label}-beauty` }),
    planet: gpu.target({ size: full, format: PLANET_FORMAT, msaa: true, depth: true, label: `${label}-planet` }),
    bloomA: gpu.target({ size: bloom, format: HDR_FORMAT, label: `${label}-bloom-a` }),
    bloomB: gpu.target({ size: bloom, format: HDR_FORMAT, label: `${label}-bloom-b` }),
  };
}

function setStaticBindings(scene: Scene, maps: Maps, targets: Targets): void {
  const { bloom, grade } = EARTH_TUNING;
  scene.earth.set({ surfaceMap: maps.surface, cloudMap: maps.clouds, mapSampler: scene.mapSampler });
  scene.overlay.set({ planetTexture: targets.planet, samp: scene.linearSampler });
  scene.bright.set({ samp: scene.linearSampler, bright: { threshold: bloom.threshold, knee: bloom.knee } });
  const directions = [
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
  ] as const;
  scene.blur.forEach((pass, index) => {
    pass.set({
      samp: scene.linearSampler,
      blur: { direction: directions[index]!, radius: bloom.radii[index < 2 ? 0 : 1] },
    });
  });
  scene.composite.set({
    samp: scene.linearSampler,
    bloom: targets.bloomA,
    composite: {
      bloomStrength: bloom.strength,
      exposure: grade.exposure,
      vignetteStart: grade.vignetteStart,
      vignetteDarkness: grade.vignetteDarkness,
      grain: grade.grain,
    },
  });
  setSizeBindings(scene, targets);
}

function setSizeBindings(scene: Scene, targets: Targets): void {
  scene.bright.set({ src: targets.beauty });
  scene.blur[0].set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  scene.blur[1].set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  scene.blur[2].set({ src: targets.bloomA, blur: { texelSize: targets.bloomA.texelSize } });
  scene.blur[3].set({ src: targets.bloomB, blur: { texelSize: targets.bloomB.texelSize } });
  scene.composite.set({ beauty: targets.beauty });
}

/**
 * One-shot: the maps never change, so this runs once before the first frame and is
 * never repeated. Both bakes go in a single submit.
 */
async function bakeMaps(gpu: Gpu, maps: Maps): Promise<void> {
  const surface = gpu.effect(bakeSurfaceWgsl, { label: 'earth-bake-surface' });
  const clouds = gpu.effect(bakeCloudsWgsl, { label: 'earth-bake-clouds' });
  await Promise.all([surface.compile(maps.surface), clouds.compile(maps.clouds)]);
  gpu.frame((frame) => {
    frame.pass({ target: maps.surface, clear: TRANSPARENT }, (pass) => pass.draw(surface));
    frame.pass({ target: maps.clouds, clear: TRANSPARENT }, (pass) => pass.draw(clouds));
  });
}

async function prewarm(scene: Scene, targets: Targets, output: Output): Promise<void> {
  await Promise.all([
    scene.sky.compile(targets.beauty),
    scene.earth.compile(targets.planet),
    scene.atmosphere.compile(targets.planet),
    scene.overlay.compile(targets.beauty),
    scene.bright.compile(targets.bloomA),
    scene.blur[0].compile(targets.bloomB),
    scene.blur[1].compile(targets.bloomA),
    scene.blur[2].compile(targets.bloomB),
    scene.blur[3].compile(targets.bloomA),
    scene.composite.compile({ colors: [output.format] }),
  ]);
}

function setFrameUniforms(
  scene: Scene,
  output: Output,
  orbit: OrbitState,
  sunDegrees: number,
  time: number,
): void {
  const { camera, atmosphere, sun, grade } = EARTH_TUNING;
  const size = output.size;
  const aspect = size[0] / Math.max(1, size[1]);
  const position = orbitPosition(orbit);
  const light = sunDirection(sunDegrees);
  const view = perspectiveCamera({
    fov: camera.fov,
    aspect,
    near: camera.near,
    far: camera.far,
    position,
    target: [0, 0, 0],
  });
  const basis = cameraBasis(position, [0, 0, 0], camera.fov);

  scene.earth.set({
    earth: {
      viewProjection: view.viewProjection,
      cameraPosition: position,
      time,
      lightDirection: light,
      nightLights: 1,
    },
  });
  scene.atmosphere.set({
    atmosphere: {
      viewProjection: view.viewProjection,
      cameraPosition: position,
      strength: atmosphere.strength,
      lightDirection: light,
      _pad: 0,
    },
  });
  scene.sky.set({
    sky: {
      right: basis.right,
      tanHalfFov: basis.tanHalfFov,
      up: basis.up,
      aspect,
      forward: basis.forward,
      starBrightness: grade.starBrightness,
      lightDirection: light,
      sunIntensity: sun.intensity,
    },
  });
}

function render(frame: Frame, scene: Scene, targets: Targets, output: Output): void {
  // Sky first, then the planet into its own MSAA target, then lay it over the sky.
  frame.pass({ target: targets.beauty, clear: OPAQUE_BLACK }, (pass) => pass.draw(scene.sky));
  frame.pass({ target: targets.planet, clear: TRANSPARENT }, (pass) => {
    pass.draw(scene.earth);
    pass.draw(scene.atmosphere);
  });
  frame.pass({ target: targets.beauty, clear: false }, (pass) => pass.draw(scene.overlay));

  frame.pass({ target: targets.bloomA, clear: OPAQUE_BLACK }, (pass) => pass.draw(scene.bright));
  frame.pass({ target: targets.bloomB, clear: OPAQUE_BLACK }, (pass) => pass.draw(scene.blur[0]));
  frame.pass({ target: targets.bloomA, clear: OPAQUE_BLACK }, (pass) => pass.draw(scene.blur[1]));
  frame.pass({ target: targets.bloomB, clear: OPAQUE_BLACK }, (pass) => pass.draw(scene.blur[2]));
  frame.pass({ target: targets.bloomA, clear: OPAQUE_BLACK }, (pass) => pass.draw(scene.blur[3]));

  frame.pass({ target: output, clear: OPAQUE_BLACK }, (pass) => pass.draw(scene.composite));
}

function resizeTargets(targets: Targets, size: readonly [number, number]): void {
  const full = normalizeSize(size);
  const bloom = bloomSize(full);
  targets.beauty.resize(full);
  targets.planet.resize(full);
  targets.bloomA.resize(bloom);
  targets.bloomB.resize(bloom);
}

function destroyScene(scene: Scene): void {
  (scene.earthMesh as { destroy?: () => void }).destroy?.();
  (scene.atmosphereMesh as { destroy?: () => void }).destroy?.();
}

function destroyTargets(targets: Targets): void {
  for (const target of [targets.beauty, targets.planet, targets.bloomA, targets.bloomB]) {
    (target as { destroy?: () => void }).destroy?.();
  }
}

function destroyMaps(maps: Maps): void {
  for (const target of [maps.surface, maps.clouds]) {
    (target as { destroy?: () => void }).destroy?.();
  }
}
