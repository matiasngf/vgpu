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

// Integer hash (lowbias32 over a 3D lattice): no visible structure across
// the frame, unlike the fract(sin()) family, and cheap to reseed per frame.
fn hashU(x0: u32) -> u32 {
  var x = x0;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn rand3(p: vec2i, frame: u32, salt: u32) -> f32 {
  let h = hashU((u32(p.x) * 0x9E3779B1u) ^ (u32(p.y) * 0x85EBCA77u) ^ (frame * 0xC2B2AE3Du) ^ salt);
  return f32(h & 0xffffffu) / 16777216.0;
}

// Approximately Gaussian: mean of four uniforms.
fn gauss(p: vec2i, frame: u32, salt: u32) -> f32 {
  return (rand3(p, frame, salt) + rand3(p, frame, salt + 1u) + rand3(p, frame, salt + 2u) + rand3(p, frame, salt + 3u)) * 0.5 - 1.0;
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
  color *= 1.0 - post.vignette * smoothstep(0.6, 1.2, d);

  // Photographic grain, reseeded every frame: a Gaussian luminance grain
  // that is strongest in the mids and shadows (silver density), plus a
  // coarser, independent chroma grain in YCbCr that nudges the hue of each
  // clump the way colour negative's dye clouds do. Applied in display space.
  let frame = u32(floor(post.time * 60.0));
  let pixel = vec2i(position.xy);
  let luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  let amount = post.grain * mix(1.0, 0.35, smoothstep(0.55, 1.0, luma)) * (0.6 + 0.4 * sqrt(max(luma, 0.0)));
  let mono = gauss(pixel, frame, 11u);
  // Chroma clumps are ~2 px so the colour shifts read as tint, not confetti.
  let clump = vec2i(floor(position.xy / 2.0));
  let cb = gauss(clump, frame, 101u);
  let cr = gauss(clump, frame, 211u);
  let chroma = vec3f(1.402 * cr, -0.344136 * cb - 0.714136 * cr, 1.772 * cb) * 0.45;
  color += (vec3f(mono) + chroma) * amount;

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
