// Shared helpers for the baked ground materials (concrete pad, gravel apron).
// Everything is periodic over one tile so the textures can repeat.

import { pnoise2 } from "./thruster-common.wgsl";

export const MAT_SIZE: f32 = 1024.0;
export const KIND_CONCRETE: f32 = 0.0;
export const KIND_GRAVEL: f32 = 1.0;

// vgpu targets have no mip chain, so each material ships as a small atlas of
// four box-filtered levels (1024, 256, 64, 16 texels per tile), each with a
// one-texel periodic border so bilinear filtering wraps cleanly. The scene
// shader picks the level from the pixel footprint (manual trilinear).
export const MAT_LEVELS: i32 = 4;
export const MAT_ATLAS_WIDTH: f32 = 1368.0;   // 1026 + 258 + 66 + 18
export const MAT_ATLAS_HEIGHT: f32 = 1026.0;

export fn levelSize(level: i32) -> f32 {
  return MAT_SIZE / f32(1 << u32(2 * level));
}

export fn levelOffset(level: i32) -> f32 {
  var x = 0.0;
  for (var l = 0; l < level; l++) { x += levelSize(l) + 2.0; }
  return x;
}

/// Atlas uv for tile coordinate `uv` (any range, wraps) at `level`.
export fn atlasUv(uv: vec2f, level: i32) -> vec2f {
  let size = levelSize(level);
  let texel = fract(uv) * size + 1.5 + vec2f(levelOffset(level), 0.0);
  return texel / vec2f(MAT_ATLAS_WIDTH, MAT_ATLAS_HEIGHT);
}

export fn hash2(p: vec2f) -> vec2f {
  let q = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
  return fract(sin(q) * 43758.5453);
}

export fn hash1(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

/// Periodic fbm over the tile: `p` in tile units, `period` lattice cells per tile.
export fn fbm(p: vec2f, period: i32, octaves: i32, seed: u32) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var freq = 1;
  for (var o = 0; o < octaves; o++) {
    let per = period * freq;
    sum += amp * pnoise2(p * f32(per), vec2i(per), seed + u32(o) * 7919u);
    amp *= 0.5;
    freq *= 2;
  }
  return sum; // roughly [-0.5, 0.5]
}

/// Periodic ridged noise (crack / vein look).
export fn ridged(p: vec2f, period: i32, octaves: i32, seed: u32) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var freq = 1;
  var weight = 1.0;
  for (var o = 0; o < octaves; o++) {
    let per = period * freq;
    let r = 1.0 - abs(pnoise2(p * f32(per), vec2i(per), seed + u32(o) * 104729u) * 2.0);
    sum += amp * r * r * weight;
    weight = clamp(r * r * 1.5, 0.0, 1.0);
    amp *= 0.5;
    freq *= 2;
  }
  return sum;
}

/// Periodic Voronoi cells (the "Cells" / "Tile Random" generators in Substance):
/// distance to the nearest and second-nearest feature point, plus the nearest
/// cell's id so every stone can pick its own colour and size (flood fill).
export struct Cells {
  f1: f32,
  f2: f32,
  id: vec2f,
  /// Vector from the shaded point to the nearest feature, in cell units.
  toCenter: vec2f,
}

export fn cells(p: vec2f, n: f32, seed: f32, jitter: f32) -> Cells {
  let q = p * n;
  let qi = floor(q);
  let qf = fract(q);
  var out = Cells(8.0, 8.0, vec2f(0.0), vec2f(0.0));
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let off = vec2f(f32(x), f32(y));
      var cell = qi + off;
      cell = cell - n * floor(cell / n);
      let h = hash2(cell + vec2f(seed, seed * 1.7));
      let point = off + 0.5 + (h - 0.5) * jitter;
      let d = length(point - qf);
      if (d < out.f1) {
        out.f2 = out.f1;
        out.f1 = d;
        out.id = cell + vec2f(seed);
        out.toCenter = point - qf;
      } else if (d < out.f2) {
        out.f2 = d;
      }
    }
  }
  return out;
}
