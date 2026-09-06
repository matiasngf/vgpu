// Final look pass of the social pipeline, after the composite has already
// exposed and gamma-encoded the frame: a lens-style softness that grows
// toward the frame edges, a light dark vignette, and a fine photographic
// grain on top. Everything is tuned in the display domain on purpose: this
// is the camera, not the scene.

struct Post {
  time: f32,
  grain: f32,      // grain amplitude in display units
  vignette: f32,   // darkening at the corners (0 = none)
  edgeBlur: f32,   // blur radius at the corners as a fraction of frame height
  edgeStart: f32,  // normalised distance from centre where the softness begins
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> post: Post;

const DISC = 12;

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += vec2f(dot(q, q + vec2f(45.32)));
  return fract(q.x * q.y);
}

fn ign(pixel: vec2f) -> f32 {
  return fract(52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

// Poisson-ish disc: golden-angle spiral, rotated per pixel so the taps do
// not resolve into a visible pattern (the grain hides the residual noise).
fn softened(uv: vec2f, radius: vec2f, rotation: f32) -> vec3f {
  var sum = textureSampleLevel(src, samp, uv, 0.0).rgb;
  var weightSum = 1.0;
  for (var i = 0; i < DISC; i++) {
    let t = (f32(i) + 0.5) / f32(DISC);
    let angle = f32(i) * 2.39996 + rotation;
    let offset = vec2f(cos(angle), sin(angle)) * sqrt(t) * radius;
    let w = 1.0 - 0.35 * t;
    sum += textureSampleLevel(src, samp, uv + offset, 0.0).rgb * w;
    weightSum += w;
  }
  return sum / weightSum;
}

@fragment fn fs_main(@builtin(position) position: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(src));
  let aspect = size.x / size.y;
  // Distance from the centre, 1 at the corners, aspect-corrected so the
  // softness and vignette are round rather than following the frame.
  let centered = (uv - 0.5) * vec2f(aspect, 1.0);
  let d = length(centered) / length(vec2f(0.5 * aspect, 0.5));

  // Edge softness: sharp inside edgeStart, easing out to edgeBlur at the corners.
  let soft = smoothstep(post.edgeStart, 1.0, d);
  let radiusPx = soft * soft * post.edgeBlur * size.y;
  var color = textureSampleLevel(src, samp, uv, 0.0).rgb;
  if (radiusPx > 0.5) {
    color = softened(uv, vec2f(radiusPx) / size, ign(position.xy) * 6.2831853);
  }

  // Light vignette, kept smooth so it reads as lens falloff, not a frame.
  color *= 1.0 - post.vignette * smoothstep(0.3, 1.15, d);

  // Photographic grain: two hashes averaged for a softer distribution,
  // stronger in the mids and shadows than in the highlights (silver density),
  // with a little decorrelated colour so it does not look like monochrome
  // dither. Changes every frame.
  let seed = position.xy + vec2f(fract(post.time * 0.731) * 977.0, fract(post.time * 0.317) * 613.0);
  let luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  let mono = (hash21(seed) + hash21(seed + vec2f(17.3, 31.7))) * 0.5 - 0.5;
  let chroma = vec3f(hash21(seed + vec2f(3.1, 5.7)), hash21(seed + vec2f(9.2, 1.3)), hash21(seed + vec2f(21.4, 8.8))) - 0.5;
  let amount = post.grain * mix(1.0, 0.35, smoothstep(0.55, 1.0, luma)) * (0.6 + 0.4 * sqrt(max(luma, 0.0)));
  color += (vec3f(mono) + chroma * 0.35) * amount;

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
