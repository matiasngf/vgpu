// Headless debugging aid: `target.read()` only supports 8-bit formats, so HDR
// intermediates are tonemapped into an rgba8 target before readback.
struct Preview {
  exposure: f32,
  mode: f32, // 0 = ACES tonemap, 1 = raw linear clamp
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> preview: Preview;

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + vec3f(0.03))) / (x * (2.43 * x + vec3f(0.59)) + vec3f(0.14)), vec3f(0.0), vec3f(1.0));
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let hdr = textureSampleLevel(src, samp, uv, 0.0).rgb * preview.exposure;
  let mapped = select(aces(hdr), clamp(hdr, vec3f(0.0), vec3f(1.0)), preview.mode > 0.5);
  return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
}
