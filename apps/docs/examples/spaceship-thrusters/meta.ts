export const meta = {
  slug: 'spaceship-thrusters',
  title: 'Spaceship Thrusters',
  description: 'A raymarched rocket exhaust plume — tileable 3D noise is baked once into a slice atlas so the volume march spends its budget on texture fetches instead of octave loops, then an HDR bloom and a teal/orange grade finish the shot.',
  thumb: { warmupFrames: 1, time: 6.2 },
  files: ['example.ts', 'fire.wgsl', 'thruster-common.wgsl', 'bake-noise.wgsl', 'bake-detail.wgsl', 'bright-pass.wgsl', 'blur.wgsl', 'composite.wgsl', 'debug-preview.wgsl'],
} as const;
