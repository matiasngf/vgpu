// Separable Gaussian blur of one bloom mip, one texel per tap so no pixel is skipped: the
// reach comes from running it on ever smaller mips, not from stretching the kernel.
struct Blur {
  texelSize: vec2f,
  direction: vec2f,   // (1,0) or (0,1)
  sigma: f32,         // in texels
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> blur: Blur;

const TAPS: i32 = 8;   // each side

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let step = blur.texelSize * blur.direction;
  let s2 = 2.0 * blur.sigma * blur.sigma;
  var sum = textureSampleLevel(src, samp, uv, 0.0).rgb;
  var total = 1.0;
  for (var i = 1; i <= TAPS; i++) {
    let w = exp(-f32(i * i) / s2);
    let o = step * f32(i);
    sum += (textureSampleLevel(src, samp, uv + o, 0.0).rgb + textureSampleLevel(src, samp, uv - o, 0.0).rgb) * w;
    total += 2.0 * w;
  }
  return vec4f(sum / total, 1.0);
}
