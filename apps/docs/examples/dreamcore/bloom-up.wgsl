// Walk the bloom chain back up: the smaller, blurrier mip is upsampled with a 3x3 tent
// filter (bilinear taps one texel apart, so it stays smooth instead of blocky) and added
// to this level's own blurred glow. Each level keeps a weight, so the wide halo from the
// small mips reaches far while the tight core stays bright.
struct Up {
  texelSize: vec2f,   // of the smaller mip
  weight: f32,        // of this level's own glow
  _pad: f32,
}

@group(0) @binding(0) var own: texture_2d<f32>;
@group(0) @binding(1) var smaller: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> up: Up;

fn tap(uv: vec2f, dx: f32, dy: f32) -> vec3f {
  return textureSampleLevel(smaller, samp, uv + vec2f(dx, dy) * up.texelSize, 0.0).rgb;
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  var tent = tap(uv, 0.0, 0.0) * 4.0;
  tent += (tap(uv, -1.0, 0.0) + tap(uv, 1.0, 0.0) + tap(uv, 0.0, -1.0) + tap(uv, 0.0, 1.0)) * 2.0;
  tent += tap(uv, -1.0, -1.0) + tap(uv, 1.0, -1.0) + tap(uv, -1.0, 1.0) + tap(uv, 1.0, 1.0);
  tent /= 16.0;
  let mine = textureSampleLevel(own, samp, uv, 0.0).rgb;
  return vec4f(mine * up.weight + tent, 1.0);
}
