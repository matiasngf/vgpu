// Rasterises the blade mesh top-down into a repeating grass tile: colour + height in the
// first target, blade tangent + position-along-blade in the second. Depth keeps the tallest
// blade on top, exactly like looking down at a lawn.

struct Tile {
  size: f32,     // tile edge (m)
  height: f32,   // tallest blade encoded (m)
}

@group(0) @binding(0) var<uniform> tile: Tile;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) tangent: vec3f,
  @location(1) extra: vec2f,     // (fraction along the blade, per-blade seed)
  @location(2) height: f32,
}

@vertex fn vs_main(
  @location(0) position: vec3f,
  @location(1) tangent: vec3f,
  @location(2) extra: vec2f,
) -> VertexOut {
  var out: VertexOut;
  let u = position.x / tile.size;
  let v = position.z / tile.size;
  let hf = clamp(position.y / tile.height, 0.0, 1.0);
  // Higher blades sit nearer the top-down camera, so they win the depth test.
  out.position = vec4f(u * 2.0 - 1.0, v * 2.0 - 1.0, 1.0 - hf, 1.0);
  out.tangent = tangent;
  out.extra = extra;
  out.height = position.y;
  return out;
}

struct FragmentOut {
  @location(0) color: vec4f,
  @location(1) tangent: vec4f,
}

fn srgb2lin(c: vec3f) -> vec3f {
  return pow((c + vec3f(0.055)) / 1.055, vec3f(2.4));
}

fn lin2srgb(c: vec3f) -> vec3f {
  return 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
}

@fragment fn fs_main(in: VertexOut) -> FragmentOut {
  let s = in.extra.x;
  let seed = in.extra.y;
  let base = srgb2lin(vec3f(46.0, 86.0, 27.0) / 255.0);
  let tip = srgb2lin(vec3f(148.0, 176.0, 64.0) / 255.0);
  var albedo = mix(base, tip, smoothstep(0.05, 1.0, s));
  albedo *= 0.72 + 0.56 * seed;
  albedo *= mix(vec3f(1.0), vec3f(1.1, 1.0, 0.82), fract(seed * 7.31) * 0.6);
  // A few dead, straw-coloured blades.
  if (fract(seed * 13.7) < 0.07) {
    albedo = mix(albedo, srgb2lin(vec3f(158.0, 128.0, 66.0) / 255.0), 0.85);
  }
  var out: FragmentOut;
  // Packed for 8-bit targets: sRGB-encoded albedo, height as a fraction of the tallest blade,
  // tangent remapped to [0,1].
  out.color = vec4f(lin2srgb(albedo), clamp(in.height / tile.height, 0.0, 1.0));
  out.tangent = vec4f(normalize(in.tangent) * 0.5 + vec3f(0.5), s);
  return out;
}
