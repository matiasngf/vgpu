// Second half of the ground material bake: derives the normal map, a cavity
// mask (height below its neighbourhood, what Substance's "Dirt" node reads
// from AO/curvature), and then albedo and roughness from height + masks.
//
// Outputs:
//   color:  rgb albedo (linear, ~1.0 mean, tinted by the scene shader), a roughness
//   normal: rg tangent-space normal * 0.5 + 0.5, b height, a cavity

import { MAT_SIZE, KIND_CONCRETE, hash1, hash2, fbm } from "./material-common.wgsl";

struct Mat {
  kind: f32,
}

@group(0) @binding(0) var<uniform> mat: Mat;
@group(0) @binding(1) var height: texture_2d<f32>;

struct FinishOut {
  @location(0) color: vec4f,
  @location(1) normal: vec4f,
}

fn load(pixel: vec2i) -> vec4f {
  let size = i32(MAT_SIZE);
  return textureLoad(height, ((pixel % size) + size) % size, 0);
}

// Height minus a wide box blur of the height: negative in hollows.
fn concavity(pixel: vec2i, radius: i32, stride: i32) -> f32 {
  var sum = 0.0;
  var count = 0.0;
  for (var y = -radius; y <= radius; y++) {
    for (var x = -radius; x <= radius; x++) {
      sum += load(pixel + vec2i(x, y) * stride).r;
      count += 1.0;
    }
  }
  return load(pixel).r - sum / count;
}

@fragment fn fs_main(@builtin(position) position: vec4f) -> FinishOut {
  let pixel = vec2i(position.xy);
  let p = position.xy / MAT_SIZE;
  let center = load(pixel);
  let h = center.r;
  let texel = 1.0 / MAT_SIZE;
  // Sobel gradient in tile units.
  let l = load(pixel + vec2i(-1, 0)).r; let r = load(pixel + vec2i(1, 0)).r;
  let u = load(pixel + vec2i(0, -1)).r; let d = load(pixel + vec2i(0, 1)).r;
  let ul = load(pixel + vec2i(-1, -1)).r; let ur = load(pixel + vec2i(1, -1)).r;
  let dl = load(pixel + vec2i(-1, 1)).r; let dr = load(pixel + vec2i(1, 1)).r;
  let gx = ((ur + 2.0 * r + dr) - (ul + 2.0 * l + dl)) / (8.0 * texel);
  let gy = ((dl + 2.0 * d + dr) - (ul + 2.0 * u + ur)) / (8.0 * texel);
  let slope = length(vec2f(gx, gy));
  let n = normalize(vec3f(-gx, -gy, 1.0));

  // Curvature scale: concrete relief is millimetres, gravel centimetres.
  let relief = select(45.0, 350.0, mat.kind == KIND_CONCRETE);
  let cavityNear = clamp(-concavity(pixel, 2, 2) * relief, 0.0, 1.0);        // tight hollows: pits, gaps between stones
  let cavityWide = clamp(-concavity(pixel, 3, 6) * relief * 0.4, 0.0, 1.0);  // broader dips
  let cavity = clamp(cavityNear * 0.7 + cavityWide * 0.5, 0.0, 1.0);
  let peak = clamp(concavity(pixel, 3, 6) * relief * 0.5, 0.0, 1.0);         // exposed tops

  let random = center.g;
  let layer = center.b;
  let extra = center.a;
  let speckle = hash1(floor(position.xy)) - 0.5;
  var albedo: vec3f;
  var roughness: f32;

  if (mat.kind == KIND_CONCRETE) {
    // Cement paste: mean 1.0 with soft macroTone clouds; aggregate grains are a
    // touch lighter and greyer with their own tint; hollows collect dirt and
    // go darker and warmer; cracks and pits are darkest.
    // Clouds at three scales (Substance "Clouds 2" stacked), the worn areas
    // where the paste has gone, damp stains and a little efflorescence.
    let clouds = 0.12 * fbm(p, 2, 3, 0x4000u) + 0.07 * fbm(p, 8, 3, 0x4001u) + 0.04 * fbm(p, 32, 2, 0x4002u);
    let wear = smoothstep(0.08, 0.3, fbm(p + vec2f(0.5, 0.2), 3, 4, 0x3100u) + 0.5 * fbm(p, 12, 2, 0x3101u));
    let stain = smoothstep(0.18, 0.4, fbm(p + vec2f(0.2, 0.9), 3, 4, 0x4003u) + 0.3 * fbm(p, 24, 2, 0x4004u));
    let salt = smoothstep(0.24, 0.38, fbm(p + vec2f(0.7, 0.3), 4, 3, 0x4005u));
    var tone = vec3f(1.0 + clouds);
    tone *= 1.0 - 0.12 * wear;                                            // eroded paste is darker and rougher
    let grainTint = mix(vec3f(0.95, 0.96, 1.0), vec3f(1.08, 1.02, 0.94), hash1(vec2f(random, 0.3)));
    tone = mix(tone, grainTint * (0.92 + 0.22 * random), layer * 0.75);
    tone *= 1.0 + 0.05 * speckle;
    tone = mix(tone, tone * vec3f(0.7, 0.66, 0.62), stain * 0.55);       // damp, slightly warm
    tone = mix(tone, vec3f(1.12, 1.11, 1.08), salt * 0.25);              // whitish salt bloom
    tone = mix(tone, tone * vec3f(0.62, 0.58, 0.53), cavity * 0.6);      // dirt in hollows
    tone = mix(tone, tone * vec3f(0.6, 0.58, 0.55), extra * 0.5);         // pits and cracks
    tone *= mix(1.0, 1.05, peak);
    albedo = tone;
    roughness = 0.84 + 0.1 * cavity + 0.1 * extra - 0.08 * peak + 0.04 * speckle + 0.08 * wear - 0.2 * stain;
  } else {
    // Stones: per-stone value and hue from the flood-fill random; dirt is a
    // warmer, darker sand with fine grain; a dust film lightens flat tops.
    // Mostly grey-brown crushed stone, a few cooler and a few rustier ones.
    let stoneHue = hash2(vec2f(random * 7.0, random * 3.0));
    var stoneColor = mix(vec3f(0.9, 0.9, 0.9), vec3f(1.06, 0.98, 0.86), stoneHue.x);  // neutral grey ... warm grey
    stoneColor = mix(stoneColor, vec3f(0.84, 0.86, 0.9), step(0.8, stoneHue.y) * 0.5);  // cooler
    stoneColor = mix(stoneColor, vec3f(1.05, 0.88, 0.74), step(0.92, stoneHue.y) * 0.6); // rusty
    stoneColor *= 0.72 + 0.6 * random;
    // Sand between the stones: warm, with fine grain and slow colour drift.
    let sandGrain = 0.5 + 0.5 * fbm(p, 200, 2, 0x5000u);
    let sandDrift = fbm(p, 3, 3, 0x5001u);
    let dirtColor = vec3f(0.88, 0.8, 0.68) * (0.85 + 0.25 * sandGrain + 0.2 * sandDrift);
    let stoneFace = 1.0 - smoothstep(0.35, 0.9, slope);                  // top faces vs sides
    var tone = mix(dirtColor, stoneColor * (0.85 + 0.15 * stoneFace), layer);
    tone *= 1.0 + 0.12 * speckle * (1.0 - layer) + 0.05 * speckle * layer;
    tone = mix(tone, tone * vec3f(0.55, 0.5, 0.45), cavity * 0.8);
    let flatTop = layer * (1.0 - smoothstep(0.3, 1.2, slope)) * extra;
    tone = mix(tone, vec3f(1.0, 0.97, 0.9) * 0.95, flatTop * 0.35);
    albedo = tone;
    roughness = mix(0.95, 0.72 + 0.15 * random, layer) + 0.1 * cavity;
  }

  var out: FinishOut;
  out.color = vec4f(clamp(albedo, vec3f(0.0), vec3f(2.0)) * 0.5, clamp(roughness, 0.3, 1.0));
  out.normal = vec4f(n.xy * 0.5 + 0.5, clamp(h, 0.0, 1.0), cavity);
  return out;
}
