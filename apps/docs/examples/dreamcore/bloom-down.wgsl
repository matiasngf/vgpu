// Downsample one bloom mip into the next (half size) with the 13-tap filter from Jimenez
// (Call of Duty: Advanced Warfare, SIGGRAPH 2014): five overlapping 2x2 boxes weighted
// toward the centre. Every source texel contributes, so nothing is skipped and small bright
// spots do not flicker or turn blocky as they travel down the chain.
struct Down {
  texelSize: vec2f,   // of the source
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> down: Down;

fn tap(uv: vec2f, dx: f32, dy: f32) -> vec3f {
  return textureSampleLevel(src, samp, uv + vec2f(dx, dy) * down.texelSize, 0.0).rgb;
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let a = tap(uv, -2.0, -2.0);
  let b = tap(uv, 0.0, -2.0);
  let c = tap(uv, 2.0, -2.0);
  let d = tap(uv, -1.0, -1.0);
  let e = tap(uv, 1.0, -1.0);
  let f = tap(uv, -2.0, 0.0);
  let g = tap(uv, 0.0, 0.0);
  let h = tap(uv, 2.0, 0.0);
  let i = tap(uv, -1.0, 1.0);
  let j = tap(uv, 1.0, 1.0);
  let k = tap(uv, -2.0, 2.0);
  let l = tap(uv, 0.0, 2.0);
  let m = tap(uv, 2.0, 2.0);
  var sum = (d + e + i + j) * 0.5 * 0.25;
  sum += (a + b + f + g) * 0.125 * 0.25;
  sum += (b + c + g + h) * 0.125 * 0.25;
  sum += (f + g + k + l) * 0.125 * 0.25;
  sum += (g + h + l + m) * 0.125 * 0.25;
  return vec4f(sum, 1.0);
}
