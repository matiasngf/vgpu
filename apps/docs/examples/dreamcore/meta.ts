export const meta = {
  slug: 'dreamcore',
  title: 'Dreamcore Door',
  description: 'A raymarched Bliss-style field at night with a door standing in the grass. Through the opening the same field is seen in daylight; the day then spills out of the door and sweeps across the hills. Heightfield marching, SDF objects, a portal ray transform and an HDR bloom chain in one effect graph.',
  thumb: { warmupFrames: 1, time: 4.6 },
  files: ['example.ts', 'scene.wgsl', 'bright-pass.wgsl', 'blur.wgsl', 'post.wgsl'],
} as const;
