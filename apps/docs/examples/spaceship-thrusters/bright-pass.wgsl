struct Bright {
  threshold: f32,
  knee: f32,
}

@group(0) @binding(0) var fire: texture_2d<f32>;   // premultiplied plume
@group(0) @binding(1) var scene: texture_2d<f32>;  // lit geometry
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> bright: Bright;

// Exponent all ones = NaN or inf. One such texel would be smeared into a
// black rectangle by the separable blur, so it is dropped here instead.
fn finite(v: vec3f) -> vec3f {
  let bits = bitcast<vec3u>(v) & vec3u(0x7f800000u);
  return select(v, vec3f(0.0), bits == vec3u(0x7f800000u));
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let plume = textureSampleLevel(fire, samp, uv, 0.0);
  let color = finite(plume.rgb + plume.a * textureSampleLevel(scene, samp, uv, 0.0).rgb);
  let luminance = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  let knee = max(bright.knee, 0.0001);
  let soft = clamp((luminance - bright.threshold + knee) / (2.0 * knee), 0.0, 1.0);
  let contribution = max(soft * soft * knee, luminance - bright.threshold);
  return vec4f(color * max(contribution / max(luminance, 0.0001), 0.0), 1.0);
}
