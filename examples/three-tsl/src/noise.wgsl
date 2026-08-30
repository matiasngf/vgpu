import { hash3 } from "@vgpu/wgsl-std/hash";

// Trilinear value noise over a lattice hashed with @vgpu/wgsl-std's hash3.
export fn valueNoise3(position: vec3f) -> f32 {
  let cell = floor(position);
  let local = fract(position);
  let fade = local * local * (3.0 - 2.0 * local);
  let c000 = hash3(cell + vec3f(0.0, 0.0, 0.0)).x;
  let c100 = hash3(cell + vec3f(1.0, 0.0, 0.0)).x;
  let c010 = hash3(cell + vec3f(0.0, 1.0, 0.0)).x;
  let c110 = hash3(cell + vec3f(1.0, 1.0, 0.0)).x;
  let c001 = hash3(cell + vec3f(0.0, 0.0, 1.0)).x;
  let c101 = hash3(cell + vec3f(1.0, 0.0, 1.0)).x;
  let c011 = hash3(cell + vec3f(0.0, 1.0, 1.0)).x;
  let c111 = hash3(cell + vec3f(1.0, 1.0, 1.0)).x;
  let bottom = mix(mix(c000, c100, fade.x), mix(c010, c110, fade.x), fade.y);
  let top = mix(mix(c001, c101, fade.x), mix(c011, c111, fade.x), fade.y);
  return mix(bottom, top, fade.z);
}

fn gradientAt(cell: vec3f) -> vec3f {
  return hash3(cell) * 2.0 - 1.0;
}

// Quintic-faded gradient (Perlin-style) noise, ~0..1. Much crisper than
// value noise: features are isotropic blobs instead of blurry lattice cells.
export fn perlin3(position: vec3f) -> f32 {
  let cell = floor(position);
  let local = fract(position);
  let fade = local * local * local * (local * (local * 6.0 - 15.0) + 10.0);
  let d000 = dot(gradientAt(cell + vec3f(0.0, 0.0, 0.0)), local - vec3f(0.0, 0.0, 0.0));
  let d100 = dot(gradientAt(cell + vec3f(1.0, 0.0, 0.0)), local - vec3f(1.0, 0.0, 0.0));
  let d010 = dot(gradientAt(cell + vec3f(0.0, 1.0, 0.0)), local - vec3f(0.0, 1.0, 0.0));
  let d110 = dot(gradientAt(cell + vec3f(1.0, 1.0, 0.0)), local - vec3f(1.0, 1.0, 0.0));
  let d001 = dot(gradientAt(cell + vec3f(0.0, 0.0, 1.0)), local - vec3f(0.0, 0.0, 1.0));
  let d101 = dot(gradientAt(cell + vec3f(1.0, 0.0, 1.0)), local - vec3f(1.0, 0.0, 1.0));
  let d011 = dot(gradientAt(cell + vec3f(0.0, 1.0, 1.0)), local - vec3f(0.0, 1.0, 1.0));
  let d111 = dot(gradientAt(cell + vec3f(1.0, 1.0, 1.0)), local - vec3f(1.0, 1.0, 1.0));
  let bottom = mix(mix(d000, d100, fade.x), mix(d010, d110, fade.x), fade.y);
  let top = mix(mix(d001, d101, fade.x), mix(d011, d111, fade.x), fade.y);
  return clamp(mix(bottom, top, fade.z) * 0.72 + 0.5, 0.0, 1.0);
}

// Rotation applied between fbm octaves so no lattice direction survives.
const octaveRotation = mat3x3f(
  vec3f(0.0, 0.8, 0.6),
  vec3f(-0.8, 0.36, -0.48),
  vec3f(0.6, -0.48, 0.64),
);

export fn fbm3(position: vec3f, octaves: u32) -> f32 {
  var total = 0.0;
  var amplitude = 0.5;
  var sample = position;
  var normalization = 0.0;
  for (var i = 0u; i < octaves; i++) {
    total += perlin3(sample) * amplitude;
    normalization += amplitude;
    amplitude *= 0.5;
    sample = octaveRotation * sample * 2.03 + vec3f(11.5, 5.2, 7.8);
  }
  return total / max(normalization, 1e-5);
}

// Turbulence: fbm over |signed noise|. The absolute value folds every zero
// crossing into a crease, so the result reads as fractal rock rather than
// smooth clouds.
export fn turbulence3(position: vec3f, octaves: u32) -> f32 {
  var total = 0.0;
  var amplitude = 0.5;
  var sample = position;
  var normalization = 0.0;
  for (var i = 0u; i < octaves; i++) {
    total += abs(perlin3(sample) * 2.0 - 1.0) * amplitude;
    normalization += amplitude;
    amplitude *= 0.55;
    sample = octaveRotation * sample * 2.13 + vec3f(11.5, 5.2, 7.8);
  }
  return total / max(normalization, 1e-5);
}

// Ridged multifractal: sharp creases where the noise crosses its midline.
// Returns 0..1 with thin bright ridges near 1.
export fn ridged3(position: vec3f, octaves: u32) -> f32 {
  var total = 0.0;
  var amplitude = 0.5;
  var sample = position;
  var normalization = 0.0;
  for (var i = 0u; i < octaves; i++) {
    let ridge = 1.0 - abs(perlin3(sample) * 2.0 - 1.0);
    total += ridge * ridge * amplitude;
    normalization += amplitude;
    amplitude *= 0.5;
    sample = octaveRotation * sample * 2.03 + vec3f(11.5, 5.2, 7.8);
  }
  return total / max(normalization, 1e-5);
}
