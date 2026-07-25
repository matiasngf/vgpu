export const meta = {
  slug: 'environment-map',
  title: 'Environment Map',
  description: 'One 360° equirectangular map lights the whole scene: it is the background, the reflection, and the dispersive refraction inside a floating glass cube.',
  thumb: { warmupFrames: 3, dt: 1 / 60, time: 2.1 },
  files: ['example.ts', 'camera.ts', 'controls.ts', 'sky.wgsl', 'glass.wgsl', 'present.wgsl', 'env-common.wgsl'],
} as const;
