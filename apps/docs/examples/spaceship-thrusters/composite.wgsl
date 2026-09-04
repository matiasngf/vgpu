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

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += vec2f(dot(q, q + vec2f(45.32)));
  return fract(q.x * q.y);
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  // The fire pass is scene-linear radiance. Halation from the bloom chain is
  // added with the red bias film shows, then the sensor response soft-clips
  // each channel independently: an orange core saturates R first, then G,
  // then B, which is what turns the hottest part of a flame white on camera.
  let radiance = textureSampleLevel(scene, samp, uv, 0.0).rgb;
  let halation = textureSampleLevel(bloom, samp, uv, 0.0).rgb * vec3f(1.0, 0.85, 0.78);
  let exposed = (radiance + halation * composite.bloomStrength) * composite.exposure;
  var color = vec3f(1.0) - exp(-exposed);

  // Gentle film-style contrast: lift mids slightly, keep the toe.
  color = mix(color, color * color * (3.0 - 2.0 * color), 0.35);

  let centered = uv - vec2f(0.5);
  let vignette = 1.0 - smoothstep(0.5, 1.2, length(centered) * 1.55);
  color *= mix(0.8, 1.0, vignette);

  color = pow(color, vec3f(1.0 / 2.2));
  color += (hash21(uv * 1024.0 + fract(composite.time) * 17.0) - 0.5) * composite.grain;
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
