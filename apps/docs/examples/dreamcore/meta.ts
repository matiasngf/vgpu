export const meta = {
  slug: 'dreamcore',
  title: 'Dreamcore Door',
  description: 'A raymarched Bliss-style meadow at night with a door standing in the grass. Through the opening lies a sunlit dune, and the day then pours out of the door and sweeps across the hills. Heightfield hills, grid-traced grass blades lit by a rectangular area light, SDF objects, a portal into a second heightfield and an HDR bloom chain in one effect graph.',
  thumb: { warmupFrames: 1, time: 4.6 },
  files: ['example.ts', 'scene.wgsl', 'bright-pass.wgsl', 'blur.wgsl', 'post.wgsl'],
} as const;
