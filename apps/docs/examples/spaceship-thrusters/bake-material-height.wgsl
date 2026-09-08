// First half of the ground material bake: height and masks for one tile of
// concrete or gravel, in the spirit of a Substance graph. Layers are stacked
// by height ("height blend"): whichever layer is taller wins, with a soft
// edge, so stones sit in dirt and aggregate pokes through worn concrete.
//
// Output (rgba16float):
//   r  height, tile units (1 = tile width)
//   g  per-feature random (stone tint, aggregate tint)
//   b  layer mask: 1 = stone / exposed aggregate, 0 = dirt / cement paste
//   a  secondary mask: pits + cracks (concrete), dust (gravel)

import { MAT_SIZE, KIND_CONCRETE, hash1, hash2, fbm, ridged, cells } from "./material-common.wgsl";

struct Mat {
  kind: f32,
}

@group(0) @binding(0) var<uniform> mat: Mat;

struct HeightOut {
  height: f32,
  random: f32,
  layer: f32,
  extra: f32,
}

// A pebble: a polygonal cell with bevelled edges and a domed top. Stones whose
// hash says so are missing, which leaves dirt between the others.
fn stone(p: vec2f, n: f32, seed: f32, bevel: f32, keep: f32, sizeMin: f32) -> vec3f {
  let c = cells(p, n, seed, 0.9);
  let h = hash2(c.id * 3.1 + vec2f(seed));
  if (h.x > keep) { return vec3f(0.0, h.y, 0.0); }
  let size = mix(sizeMin, 1.0, h.y);
  // Edge distance from F2 - F1 (the classic Substance "edge detect" on cells),
  // clipped by the cell's own radius so a stone does not fill its whole cell.
  let edge = clamp((c.f2 - c.f1) / bevel, 0.0, 1.0);
  let radial = clamp(1.0 - c.f1 / (0.62 * size), 0.0, 1.0);
  // Crushed stone: a flat-ish top that drops off quickly at the edge, with
  // a couple of facets tilting the top so it reads angular rather than round.
  let top = pow(max(radial, 0.0), 0.35);
  let shape = smoothstep(0.0, 1.0, edge) * top;
  let facets = fbm(p * 2.0 + h, 16, 2, 0x51u + u32(seed * 13.0)) * 0.4 + fbm(p, 96, 2, 0x52u) * 0.05;
  return vec3f(shape * (0.75 + 0.5 * h.x) * (1.0 + facets), h.x, select(0.0, 1.0, shape > 0.0));
}

// Heights are in tile units (tile = 1.5 m): a 4 cm pebble is ~0.027.
fn gravel(p: vec2f) -> HeightOut {
  // Dirt / sand bed: gentle undulation, fine grain, and grit (tiny cells).
  let grit = cells(p, 220.0, 17.0, 1.0);
  let gritDome = clamp(1.0 - grit.f1 / 0.5, 0.0, 1.0);
  let bed = 0.02 + 0.008 * fbm(p, 4, 3, 0x1000u) + 0.003 * fbm(p, 32, 3, 0x1001u)
    + 0.0025 * gritDome * gritDome * step(0.5, hash1(grit.id));
  // Three stone sizes; larger stones are sparser (missing cells become dirt).
  let large = stone(p, 11.0, 3.0, 0.1, 0.6, 0.6);
  let medium = stone(p, 21.0, 7.0, 0.14, 0.72, 0.55);
  let small = stone(p, 40.0, 11.0, 0.2, 0.85, 0.5);
  var out = HeightOut(bed, 0.5 + 0.5 * fbm(p, 48, 3, 0x1003u), 0.0, 0.0);
  let layers = array<vec3f, 3>(large * vec3f(0.06, 1.0, 1.0), medium * vec3f(0.04, 1.0, 1.0), small * vec3f(0.026, 1.0, 1.0));
  for (var i = 0; i < 3; i++) {
    let top = layers[i].x;
    // Height blend: soft max against what is already there.
    let mask = smoothstep(-0.003, 0.003, top - out.height) * layers[i].z;
    out.height = mix(out.height, top, mask);
    out.random = mix(out.random, layers[i].y, mask);
    out.layer = max(out.layer, mask);
  }
  // Dust settles on flat tops and in the low bed, not on stone sides.
  out.extra = clamp(0.5 + 0.5 * fbm(p * 1.3 + vec2f(0.3), 6, 3, 0x2000u), 0.0, 1.0);
  return out;
}

// Heights in tile units (tile = 3 m): pits are a few millimetres deep.
fn concrete(p: vec2f) -> HeightOut {
  // Base slab: broad undulation and faint float/trowel marks (anisotropic noise).
  var h = 0.5 + 0.004 * fbm(p, 2, 3, 0x3000u);
  let trowel = fbm(vec2f(p.x * 14.0 + 0.3 * fbm(p, 4, 2, 0x3001u), p.y * 0.5), 2, 4, 0x3002u);
  h += 0.0012 * trowel;
  // Cement paste micro grain.
  h += 0.0008 * fbm(p, 64, 3, 0x3003u) + 0.0004 * fbm(p, 256, 2, 0x3004u);

  // Worn patches where the paste has eroded and the aggregate shows: a grunge
  // mask picks the areas, small cells give the grains, and the grains only
  // rise where the mask says the paste is gone.
  let wear = smoothstep(0.08, 0.3, fbm(p + vec2f(0.5, 0.2), 3, 4, 0x3100u) + 0.5 * fbm(p, 12, 2, 0x3101u));
  let grains = cells(p, 90.0, 5.0, 0.85);
  let grainShape = clamp(1.0 - grains.f1 / 0.55, 0.0, 1.0);
  let grainHash = hash1(grains.id);
  let aggregate = grainShape * grainShape * step(0.35, grainHash) * wear;
  h += 0.0025 * aggregate - 0.002 * wear;

  // Air pits: sparse small cells, a spherical cap sunk into the paste.
  let pitCells = cells(p + vec2f(0.17), 80.0, 9.0, 1.0);
  let pitHash = hash2(pitCells.id * 1.3);
  let pitRadius = mix(0.08, 0.26, pitHash.y) * step(0.8, pitHash.x);
  let pitDepth = clamp(1.0 - (pitCells.f1 * pitCells.f1) / max(pitRadius * pitRadius, 1e-4), 0.0, 1.0);
  let pit = sqrt(pitDepth) * step(0.001, pitRadius);
  h -= 0.004 * pit;
  // Larger, rarer honeycomb pockets.
  let pocketCells = cells(p + vec2f(0.61, 0.29), 20.0, 21.0, 1.0);
  let pocketHash = hash2(pocketCells.id * 2.7);
  let pocketRadius = mix(0.12, 0.28, pocketHash.y) * step(0.9, pocketHash.x);
  let pocket = clamp(1.0 - pocketCells.f1 / max(pocketRadius, 1e-4), 0.0, 1.0) * step(0.001, pocketRadius);
  h -= 0.006 * pocket * pocket;

  // Hairline cracks: warped Voronoi edges, thinned, only along some segments.
  let warp = vec2f(fbm(p, 6, 3, 0x3200u), fbm(p + vec2f(0.4), 6, 3, 0x3201u)) * 0.06;
  let crackCells = cells(p + warp, 5.0, 33.0, 1.0);
  let crackWidth = 0.012 + 0.01 * fbm(p, 40, 2, 0x3202u);
  let crackLine = 1.0 - smoothstep(0.0, crackWidth, crackCells.f2 - crackCells.f1);
  let crackMask = smoothstep(0.12, 0.3, fbm(p + vec2f(0.8, 0.1), 3, 3, 0x3203u) + 0.3 * ridged(p, 8, 2, 0x3204u) - 0.2);
  let crack = crackLine * crackMask;
  h -= 0.004 * crack;

  return HeightOut(h, grainHash, aggregate, clamp(pit + pocket + crack, 0.0, 1.0));
}

@fragment fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let p = position.xy / MAT_SIZE;
  var out: HeightOut;
  if (mat.kind == KIND_CONCRETE) { out = concrete(p); } else { out = gravel(p); }
  return vec4f(out.height, out.random, out.layer, out.extra);
}
