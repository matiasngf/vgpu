struct Composite {
  exposure: f32,
  bloomStrength: f32,
  grain: f32,
  time: f32,
}

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> composite: Composite;

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + vec3f(0.03))) / (x * (2.43 * x + vec3f(0.59)) + vec3f(0.14)), vec3f(0.0), vec3f(1.0));
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += vec2f(dot(q, q + vec2f(45.32)));
  return fract(q.x * q.y);
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  // The fire pass rendered at half resolution; bilinear upsampling is enough
  // because the plume has no hard edges.
  let hdr = textureSampleLevel(scene, samp, uv, 0.0).rgb;
  let glow = textureSampleLevel(bloom, samp, uv, 0.0).rgb;
  var color = (hdr + glow * composite.bloomStrength) * composite.exposure;

  // Teal/orange grade measured from the reference: shadows lean cyan, the
  // highlights keep a warm pink cast.
  let luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  let shadowTint = vec3f(0.96, 1.02, 1.04);
  let highlightTint = vec3f(1.04, 0.99, 0.97);
  color *= mix(shadowTint, highlightTint, smoothstep(0.05, 0.9, luma));

  color = aces(color);

  let centered = uv - vec2f(0.5);
  let vignette = 1.0 - smoothstep(0.5, 1.2, length(centered) * 1.55);
  color *= mix(0.78, 1.0, vignette);

  color = pow(color, vec3f(1.0 / 2.2));
  color += (hash21(uv * 1024.0 + fract(composite.time) * 17.0) - 0.5) * composite.grain;
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
