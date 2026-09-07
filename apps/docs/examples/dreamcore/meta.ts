export const meta = {
  slug: 'dreamcore',
  title: 'Dreamcore Door',
  description: 'A raymarched Bliss-style meadow at night with a door standing in the grass. Through the opening lies rippled sand under a backlit dune, and the day then pours out of the door and sweeps across the hills. Heightfield hills, a relief-mapped tile of thousands of bent blades with Kajiya-Kay fibre shading, a rectangular area light, a portal into a second heightfield and an Unreal-style mip-chain bloom in one effect graph.',
  thumb: { warmupFrames: 1, time: 4.6 },
  files: ['example.ts', 'grass-tile.ts', 'scene.wgsl', 'grass-tile.wgsl', 'bright-pass.wgsl', 'bloom-down.wgsl', 'bloom-blur.wgsl', 'bloom-up.wgsl', 'post.wgsl'],
} as const;
