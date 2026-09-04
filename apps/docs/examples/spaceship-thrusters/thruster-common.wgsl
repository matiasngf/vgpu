// Shared constants and helpers for the spaceship-thrusters example.
//
// All expensive noise (multi-octave gradient noise) is baked ONCE into two
// textures at startup. At runtime every noise lookup is a texture fetch, so the
// raymarcher can afford many samples per pixel without recomputing octaves.
//
// 1. `atlas`  — a tileable 3D noise volume stored as a 2D atlas of slices.
//               Each slice is TILE×TILE texels with a 1-texel periodic border so
//               bilinear filtering never bleeds into the neighbouring tile.
//               Two fetches (slice z0 and z1) give trilinear 3D noise.
// 2. `detail` — a tileable 2D texture of high-frequency fbm / ridged noise and
//               a warp vector, sampled with a repeat sampler.

export const TILE: f32 = 128.0;         // texels per slice edge (without border)
export const BORDER: f32 = 1.0;         // periodic border on every side
export const STRIDE: f32 = TILE + 2.0 * BORDER;
export const COLS: i32 = 8;             // slices per atlas row
export const SLICES: i32 = 64;          // total slices (z resolution)
export const ATLAS_SIZE: f32 = STRIDE * f32(COLS); // 1040 × 1040
export const DETAIL_SIZE: f32 = 512.0;

// Lattice period of the baked noise at octave 0 (in cells per tile). Higher
// numbers mean a higher base frequency inside one tile.
export const PERIOD_XY: i32 = 4;
export const PERIOD_Z: i32 = 2;
export const DETAIL_PERIOD: i32 = 8;

export fn atlasTileUv(uv: vec2f, slice: i32) -> vec2f {
  let col = slice % COLS;
  let row = slice / COLS;
  // Texel-centre of index BORDER + fract(uv) * TILE inside this tile.
  let texel = fract(uv) * TILE + BORDER + 0.5 + vec2f(f32(col), f32(row)) * STRIDE;
  return texel / ATLAS_SIZE;
}

// Trilinear tileable 3D noise: rgba = (fbm, billow, ridged, low-frequency).
export fn noise3(atlas: texture_2d<f32>, samp: sampler, p: vec3f) -> vec4f {
  let z = fract(p.z) * f32(SLICES);
  let z0 = i32(floor(z));
  let fz = z - floor(z);
  let z1 = (z0 + 1) % SLICES;
  let a = textureSampleLevel(atlas, samp, atlasTileUv(p.xy, z0), 0.0);
  let b = textureSampleLevel(atlas, samp, atlasTileUv(p.xy, z1), 0.0);
  return mix(a, b, fz);
}

// --- Periodic gradient noise used by the bake passes ------------------------

export fn wrap3(i: vec3i, period: vec3i) -> vec3i {
  return ((i % period) + period) % period;
}

export fn wrap2(i: vec2i, period: vec2i) -> vec2i {
  return ((i % period) + period) % period;
}

export fn hashU(x0: u32) -> u32 {
  var x = x0;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

export fn grad3(i: vec3i, seed: u32) -> vec3f {
  let h = hashU(((u32(i.x) * 0x9E3779B1u) ^ (u32(i.y) * 0x85EBCA77u)) ^ ((u32(i.z) * 0xC2B2AE3Du) ^ seed));
  let h2 = hashU(h ^ 0x27d4eb2fu);
  let a = f32(h & 0xffffu) / 65535.0 * 6.28318530718;
  let cz = f32(h2 & 0xffffu) / 65535.0 * 2.0 - 1.0;
  let sz = sqrt(max(0.0, 1.0 - cz * cz));
  return vec3f(cos(a) * sz, sin(a) * sz, cz);
}

export fn grad2(i: vec2i, seed: u32) -> vec2f {
  let h = hashU(((u32(i.x) * 0x9E3779B1u) ^ (u32(i.y) * 0x85EBCA77u)) ^ seed);
  let a = f32(h & 0xffffu) / 65535.0 * 6.28318530718;
  return vec2f(cos(a), sin(a));
}

export fn pnoise3(p: vec3f, period: vec3i, seed: u32) -> f32 {
  let i = vec3i(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var n = array<f32, 8>();
  for (var c = 0; c < 8; c++) {
    let o = vec3i(c & 1, (c >> 1) & 1, (c >> 2) & 1);
    n[c] = dot(grad3(wrap3(i + o, period), seed), f - vec3f(o));
  }
  let x0 = mix(n[0], n[1], u.x);
  let x1 = mix(n[2], n[3], u.x);
  let x2 = mix(n[4], n[5], u.x);
  let x3 = mix(n[6], n[7], u.x);
  return mix(mix(x0, x1, u.y), mix(x2, x3, u.y), u.z);
}

export fn pnoise2(p: vec2f, period: vec2i, seed: u32) -> f32 {
  let i = vec2i(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let n00 = dot(grad2(wrap2(i, period), seed), f);
  let n10 = dot(grad2(wrap2(i + vec2i(1, 0), period), seed), f - vec2f(1.0, 0.0));
  let n01 = dot(grad2(wrap2(i + vec2i(0, 1), period), seed), f - vec2f(0.0, 1.0));
  let n11 = dot(grad2(wrap2(i + vec2i(1, 1), period), seed), f - vec2f(1.0, 1.0));
  return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}
