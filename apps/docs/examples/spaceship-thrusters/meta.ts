export const meta = {
  slug: 'spaceship-thrusters',
  title: 'Spaceship Thrusters',
  description: 'A raymarched rocket exhaust plume with a physically based light model — blackbody soot, optically thin hydrogen glow and a per-channel sensor response — built on tileable 3D noise baked once into a slice atlas so the volume march spends its budget on texture fetches instead of octave loops.',
  thumb: { warmupFrames: 1, time: 6.2 },
  files: ['example.ts', 'fire.wgsl', 'thruster-common.wgsl', 'bake-noise.wgsl', 'bake-detail.wgsl', 'bright-pass.wgsl', 'blur.wgsl', 'composite.wgsl', 'debug-preview.wgsl'],
} as const;
