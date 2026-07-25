export const meta = {
  slug: 'earth',
  title: 'Earth',
  description:
    'A three.js planet experiment ported to vgpu: albedo, night lights and clouds are baked into equirectangular maps on the GPU, then lit by a remapped terminator with bump-shadowed cloud tops, a sunset ring, an alpha-blended atmosphere shell, and a bloom chain tuned so only the sun glows.',
  thumb: { warmupFrames: 2, dt: 1 / 60, time: 0 },
  files: [
    'example.ts', 'planet.ts', 'controls.ts',
    'planet-common.wgsl', 'bake-surface.wgsl', 'bake-clouds.wgsl',
    'sky.wgsl', 'earth.wgsl', 'atmosphere.wgsl',
    'overlay.wgsl', 'bright-pass.wgsl', 'blur.wgsl', 'composite.wgsl',
  ],
} as const;
