// Headless debugging aid: `target.read()` only supports 8-bit formats, so HDR
// intermediates are tonemapped into an rgba8 target before readback.
struct Preview {
  exposure: f32,
  mode: f32, // 0 = ACES tonemap, 1 = raw linear clamp, 2 = scalar / exposure as grey
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> preview: Preview;

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + vec3f(0.03))) / (x * (2.43 * x + vec3f(0.59)) + vec3f(0.14)), vec3f(0.0), vec3f(1.0));
}

@fragment fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  // textureLoad so unfilterable formats (r32float depth) preview too.
  let texel = textureLoad(src, vec2i(position.xy), 0);
  if (preview.mode > 1.5) {
    let g = clamp(texel.r / preview.exposure, 0.0, 1.0);
    return vec4f(vec3f(pow(g, 1.0 / 2.2)), 1.0);
  }
  let hdr = texel.rgb * preview.exposure;
  let mapped = select(aces(hdr), clamp(hdr, vec3f(0.0), vec3f(1.0)), preview.mode > 0.5);
  return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
}
