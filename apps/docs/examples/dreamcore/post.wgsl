struct Post {
  resolution: vec2f,
  exposure: f32,
  bloomStrength: f32,
  grain: f32,
  vignette: f32,
  seed: f32,
  _pad: f32,
}

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> post: Post;

fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + vec3f(33.33));
  return fract((p3.x + p3.y) * p3.z);
}

// Gentle highlight roll-off: linear below the knee, soft-clipped above so the door glow
// keeps its colour instead of blowing out.
fn softClip(x: vec3f) -> vec3f {
  let knee = 0.78;
  let above = max(x - vec3f(knee), vec3f(0.0));
  let below = min(x, vec3f(knee));
  return below + (1.0 - knee) * tanh(above / (1.0 - knee));
}

fn lin2srgb(c: vec3f) -> vec3f {
  return 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let centered = uv - vec2f(0.5);
  // Tiny chromatic aberration toward the corners.
  let ca = centered * 0.0008 * length(centered);
  let r = textureSampleLevel(scene, samp, uv + ca, 0.0).r;
  let g = textureSampleLevel(scene, samp, uv, 0.0).g;
  let b = textureSampleLevel(scene, samp, uv - ca, 0.0).b;
  let hdr = vec3f(r, g, b);
  let glow = textureSampleLevel(bloom, samp, uv, 0.0).rgb;
  // All-pass bloom that conserves energy: every pixel trades a share of itself for its own
  // multi-scale blur, so the whole frame gets a soft veil at the same brightness and the
  // bright sand through the door spreads its light in proportion.
  var color = mix(hdr, glow, post.bloomStrength) * post.exposure;
  color = softClip(color);

  let aspect = post.resolution.x / max(post.resolution.y, 1.0);
  let v = length(centered * vec2f(aspect, 1.0));
  color *= 1.0 - post.vignette * smoothstep(0.35, 1.05, v);

  var out = lin2srgb(color);
  // Fine luminance grain, like a slightly compressed photo.
  let px = floor(uv * post.resolution);
  let n = hash12(px + vec2f(post.seed * 17.0, post.seed * 31.0)) - 0.5;
  out += vec3f(n * post.grain);
  return vec4f(clamp(out, vec3f(0.0), vec3f(1.0)), 1.0);
}
