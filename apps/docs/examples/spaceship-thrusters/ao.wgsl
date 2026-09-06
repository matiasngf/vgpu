// Screen-space ambient occlusion for the social pipeline (Alchemy / SAO style
// estimator, McGuire et al. 2011/2012): for every pixel, taps on a spiral
// whose screen radius corresponds to a fixed world radius reconstruct the
// world position of each neighbour from the camera distance the scene pass
// wrote, and every neighbour that lies above the tangent plane counts as an
// occluder with a falloff on its distance. Works in world units, so the
// darkening is the same size on the pad and on the engine, and needs no
// depth linearisation: the scene pass stores the Euclidean camera distance.

struct Camera {
  invViewProj: mat4x4f,
  position: vec3f,
}

struct Ao {
  radius: f32,      // world units
  intensity: f32,
  bias: f32,        // cosine bias against self-occlusion on flat surfaces
  projScale: f32,   // pixels per world unit at distance 1
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> ao: Ao;
@group(0) @binding(2) var sceneDepth: texture_2d<f32>;  // camera distance, 0 = sky
@group(0) @binding(3) var sceneAux: texture_2d<f32>;    // world normal * 0.5 + 0.5

const TAPS = 16;
const TURNS = 7.0;
const PI = 3.14159265;

// The far plane is an affine image of NDC, so the three unprojections happen
// once per fragment and every tap reconstructs its ray with two mads.
struct FarPlane {
  origin: vec3f,  // far point at ndc (0, 0)
  dx: vec3f,      // change per unit ndc.x
  dy: vec3f,      // change per unit ndc.y
}

fn farPoint(ndc: vec2f) -> vec3f {
  let clip = camera.invViewProj * vec4f(ndc, 1.0, 1.0);
  return clip.xyz / clip.w;
}

fn farPlane() -> FarPlane {
  let origin = farPoint(vec2f(0.0));
  return FarPlane(origin, farPoint(vec2f(1.0, 0.0)) - origin, farPoint(vec2f(0.0, 1.0)) - origin);
}

fn worldAt(plane: FarPlane, pixel: vec2i, size: vec2f, dist: f32) -> vec3f {
  let uv = (vec2f(pixel) + 0.5) / size;
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let dir = normalize(plane.origin + plane.dx * ndc.x + plane.dy * ndc.y - camera.position);
  return camera.position + dir * dist;
}

// Interleaved gradient noise (Jimenez 2014): a per-pixel rotation that the
// bilateral blur averages out cleanly.
fn ign(pixel: vec2f) -> f32 {
  return fract(52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

@fragment fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(sceneDepth));
  let pixel = vec2i(position.xy);
  let dist = textureLoad(sceneDepth, pixel, 0).r;
  if (dist <= 0.0) { return vec4f(1.0); }
  let plane = farPlane();
  let p = worldAt(plane, pixel, size, dist);
  let n = normalize(textureLoad(sceneAux, pixel, 0).xyz * 2.0 - 1.0);

  let radiusPx = clamp(ao.radius * ao.projScale / dist, 3.0, 96.0);
  let base = ign(position.xy) * 2.0 * PI;
  var occlusion = 0.0;
  for (var i = 0; i < TAPS; i++) {
    let alpha = (f32(i) + 0.5) / f32(TAPS);
    let angle = alpha * TURNS * 2.0 * PI + base;
    let offset = vec2f(cos(angle), sin(angle)) * (radiusPx * alpha);
    let tap = clamp(pixel + vec2i(round(offset)), vec2i(0), vec2i(size) - 1);
    let tapDist = textureLoad(sceneDepth, tap, 0).r;
    if (tapDist <= 0.0) { continue; }
    let v = worldAt(plane, tap, size, tapDist) - p;
    let vv = dot(v, v);
    if (vv < 1e-6) { continue; }
    let invLen = inverseSqrt(vv);
    let falloff = max(1.0 - 1.0 / (invLen * ao.radius), 0.0);
    occlusion += falloff * max(dot(v, n) * invLen - ao.bias, 0.0);
  }
  // Only the taps on the occluder's side of a crease can contribute, so the
  // estimate is normalised to half the taps: a 90-degree corner lands near
  // the 0.5 visibility it should have with intensity 1.
  let visibility = clamp(1.0 - ao.intensity * occlusion / (0.5 * f32(TAPS)), 0.0, 1.0);
  return vec4f(visibility);
}
