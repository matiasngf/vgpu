export const meta = {
  slug: 'environment-map',
  title: 'Environment Map',
  description: 'One 360° equirectangular map lights the whole scene: it is the background and every reflection on a mirror-metal cube floating in it.',
  thumb: { warmupFrames: 3, dt: 1 / 60, time: 2.1 },
  files: ['example.ts', 'camera.ts', 'controls.ts', 'sky.wgsl', 'metal.wgsl', 'present.wgsl', 'env-common.wgsl'],
} as const;
