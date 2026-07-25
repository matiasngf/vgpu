import type { Draw, Effect, Frame, Gpu, Mesh, Surface, Target } from 'vgpu';
import { box } from 'vgpu/scene';
import { cameraView, spinMatrix, type CameraView } from './camera';
import { installOrbitInput } from './controls';
import skyWgsl from './sky.wgsl';
import glassWgsl from './glass.wgsl';
import presentWgsl from './present.wgsl';

type Output = Surface | Target;
interface ThumbOptions { warmupFrames?: number; dt?: number; time?: number }

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
/** 2:1 is the equirectangular aspect: 360° of yaw by 180° of pitch. */
const ENV_SIZE: readonly [number, number] = [2048, 1024];
const CUBE_SIZE = 1.25;
const EXPOSURE = 0.9;

const SKY = {
  sun_direction: [-0.724, 0.09, -0.684],
  sun_angular_size: 0.05,
  sun_color: [1.0, 0.88, 0.72],
  sun_intensity: 26,
  zenith_color: [0.05, 0.15, 0.44],
  cloud_coverage: 0.56,
  horizon_color: [0.36, 0.48, 0.74],
  cloud_scale: 0.75,
  ground_color: [0.05, 0.05, 0.056],
  ground_scale: 1.7,
} as const;

const GLASS = {
  ior: 1.47,
  dispersion: 0.045,
  absorption: 0.26,
  half_extent: CUBE_SIZE / 2,
  edge_tint: 0.3,
} as const;

interface Scene {
  readonly env: Target;
  readonly hdr: Target;
  readonly mesh: Mesh;
  readonly cube: Draw;
  readonly present: Effect;
}

export async function run(canvas: HTMLCanvasElement): Promise<() => void> {
  const { init } = await import('vgpu');
  const gpu = await init();
  const surface = gpu.surface(canvas, { dpr: [1, 2] });
  const scene = await createScene(gpu, surface);
  const input = installOrbitInput(canvas);
  let disposed = false;
  let sawInitialResize = false;

  const unsubscribeResize = surface.onResize(() => {
    if (!sawInitialResize) { sawInitialResize = true; return; }
    if (disposed) return;
    scene.hdr.resize(surface.size);
    // Resizing recreates the texture, so the composite binding has to be re-pointed.
    scene.present.set({ scene_tex: scene.hdr });
  });

  const loop = gpu.frame.loop((frame) => {
    input.advance(gpu.deltaTime);
    render(frame, scene, surface, cameraView(input.yaw, input.pitch, aspectOf(surface)), gpu.time);
  });

  return () => {
    if (disposed) return;
    disposed = true;
    loop.stop();
    unsubscribeResize();
    input.dispose();
    destroyScene(scene);
    surface.dispose();
    gpu.dispose();
  };
}

export async function renderThumb(gpu: Gpu, output: Target, opts: ThumbOptions = {}): Promise<void> {
  const scene = await createScene(gpu, output);
  const dt = opts.dt ?? 1 / 60;
  let time = opts.time ?? 2.1;

  for (let i = 0; i < Math.max(1, opts.warmupFrames ?? 3); i++) {
    time += dt;
    const view = cameraView(0.62 + time * 0.09, 0.16, aspectOf(output));
    gpu.frame((frame) => render(frame, scene, output, view, time));
  }

  await gpu.gpu.queue.onSubmittedWorkDone();
  await gpu.settled();
  destroyScene(scene);
}

async function createScene(gpu: Gpu, output: Output): Promise<Scene> {
  const env = gpu.target({ size: [...ENV_SIZE], format: HDR_FORMAT, label: 'environment-map-env' });
  const hdr = gpu.target({ size: output.size, format: HDR_FORMAT, depth: true, label: 'environment-map-scene' });
  const envSampler = gpu.sampler({
    minFilter: 'linear',
    magFilter: 'linear',
    // u wraps the horizon; v must clamp so the poles never bleed across.
    addressModeU: 'repeat',
    addressModeV: 'clamp-to-edge',
  });
  const sceneSampler = gpu.sampler({ minFilter: 'linear', magFilter: 'linear' });

  await bakeEnvironment(gpu, env);

  const mesh = gpu.mesh(box({ size: CUBE_SIZE }));
  const cube = gpu.draw({ shader: glassWgsl, mesh, label: 'environment-map-glass' });
  cube.set({ ...GLASS, env_tex: env, env_samp: envSampler });

  const present = gpu.effect(presentWgsl, { label: 'environment-map-present' });
  present.set({ env_tex: env, env_samp: envSampler, scene_tex: hdr, scene_samp: sceneSampler });

  await Promise.all([cube.compile(hdr), present.compile({ colors: [output.format] })]);
  return { env, hdr, mesh, cube, present };
}

/**
 * One-shot pass that fills the equirectangular map; nothing re-runs it per frame.
 *
 * The map is baked procedurally so the example stays self-contained and renders the same
 * frame headlessly. To use a real 360° photo instead, replace this call with an upload and
 * hand the resulting texture to `set({ env_tex })` — nothing downstream changes:
 *
 * ```ts
 * const bitmap = await createImageBitmap(await (await fetch('/hdri.png')).blob());
 * const env = gpu.device.createTexture({
 *   size: [bitmap.width, bitmap.height],
 *   format: 'rgba8unorm',
 *   usage: ['texture_binding', 'copy_dst', 'render_attachment'],
 * });
 * gpu.gpu.queue.copyExternalImageToTexture({ source: bitmap }, { texture: env.gpu }, [bitmap.width, bitmap.height]);
 * ```
 */
async function bakeEnvironment(gpu: Gpu, env: Target): Promise<void> {
  const bake = gpu.effect(skyWgsl, { label: 'environment-map-sky' });
  bake.set({ sky: SKY });
  await bake.compile(env);
  gpu.frame((frame) => frame.pass({ target: env }, (pass) => pass.draw(bake)));
}

function render(frame: Frame, scene: Scene, output: Output, view: CameraView, time: number): void {
  scene.cube.set({
    view_projection: view.camera.viewProjection,
    model: spinMatrix(time),
    camera_position: view.position,
  });
  scene.present.set({
    camera: {
      position: view.position,
      tan_half_fov: view.tanHalfFov,
      forward: view.forward,
      aspect: view.aspect,
      right: view.right,
      exposure: EXPOSURE,
      up: view.up,
      background_intensity: 1,
    },
  });

  // Alpha 0 clear turns the cube pass into a coverage mask the composite reads back.
  frame.pass({ target: scene.hdr, clear: [0, 0, 0, 0] }, (pass) => pass.draw(scene.cube));
  frame.pass({ target: output }, (pass) => pass.draw(scene.present));
}

function aspectOf(output: Output): number {
  return output.size[0] / Math.max(1, output.size[1]);
}

function destroyScene(scene: Scene): void {
  for (const target of [scene.hdr, scene.env]) {
    (target as Target & { destroy?: () => void }).destroy?.();
  }
}
