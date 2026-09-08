import { Plume, BOUND_SCALE, plumeFrame, blackbody, evaluatePlume } from "./plume-volume.wgsl";

// Direct variant of fire.wgsl: the same interleaved march, but it evaluates
// the volume model (plume-volume.wgsl) at every step instead of reading the
// per-frame grid. Cheaper below ~1080p, since the grid must evaluate the
// whole bounding cone; the grid wins once pixels x steps outgrow its voxels.
//
// Reference analysis (crops of a rocket-stage photo, sRGB 8-bit means):
//   sky            (82, 135, 149)  teal, darker toward the top
//   exit gas       translucent silver-blue cylinder with longitudinal streaks
//   afterburn body (203, 110, 79)  orange, feathered edge eroded by wisps
//   plume core     (248, 212, 205) pink-white, widens to ~60% of the plume
//   edge halo      (169, 113, 80)  fades into the sky through grey-pink

struct Params {
  resolution: vec2f, // size of the half-resolution history this march feeds
  time: f32,
  motion: f32,
  sceneScale: vec2f, // scene texels per history texel
  phase: u32,        // which pixel of the 2x2 pattern this frame marches
  frame: f32,        // frame counter for the golden-ratio jitter sequence
}

// Camera: rays are unprojected from NDC with the inverse view-projection.
struct Camera {
  invViewProj: mat4x4f,
  position: vec3f,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> plume: Plume;
// Camera distance of the lit geometry, from scene.wgsl (0 = nothing drawn).
@group(0) @binding(3) var sceneDepth: texture_2d<f32>;
@group(0) @binding(4) var atlas: texture_2d<f32>;
@group(0) @binding(5) var atlasSamp: sampler;
@group(0) @binding(6) var detail: texture_2d<f32>;
@group(0) @binding(7) var detailSamp: sampler;

// The pass outputs the plume alone, premultiplied: rgb = radiance reaching the
// camera, a = transmittance left for whatever is behind. The composite adds
// the full-resolution scene behind it. `aux` carries the heat-haze offset
// (xy, in scene pixels), the surface distance this ray was clipped to (z, for
// depth-aware upsampling) and the exit-gas shimmer strength (w).
struct FireOut {
  @location(0) fire: vec4f,
  @location(1) aux: vec4f,
}

const STEPS: i32 = 48;

fn cameraRay(ndc: vec2f) -> vec3f {
  let nearPoint = camera.invViewProj * vec4f(ndc, 0.0, 1.0);
  let farPoint = camera.invViewProj * vec4f(ndc, 1.0, 1.0);
  return normalize(farPoint.xyz / farPoint.w - nearPoint.xyz / nearPoint.w);
}

// Ray interval inside the (widened) bounding cone, clipped to 0 <= s <= LENGTH.
fn coneInterval(o: vec3f, d: vec3f) -> vec2f {
  let AXIS = plume.axis;
  let NOZZLE = plume.nozzle;
  let LENGTH = plume.length;
  let k = plume.spread * BOUND_SCALE;
  let r0 = plume.r0 * BOUND_SCALE;
  let apex = NOZZLE - AXIS * (r0 / k);
  let cos2 = 1.0 / (1.0 + k * k);
  let w = o - apex;
  let dd = dot(d, AXIS);
  let wD = dot(w, AXIS);
  let a = dd * dd - cos2;
  let b = 2.0 * (dd * wD - dot(d, w) * cos2);
  let c = wD * wD - dot(w, w) * cos2;
  var t0 = 0.0;
  var t1 = 0.0;
  if (abs(a) < 1e-5) {
    // Ray parallel to the cone surface: one crossing, inside on one side.
    if (abs(b) < 1e-6) { return vec2f(1.0, 0.0); }
    let t = -c / b;
    if (b > 0.0) { t0 = -1e9; t1 = t; } else { t0 = t; t1 = 1e9; }
  } else {
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) { return vec2f(1.0, 0.0); }
    let sq = sqrt(disc);
    t0 = min((-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a));
    t1 = max((-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a));
    if (a > 0.0) {
      // Ray steeper than the cone (looking nearly along the axis): the ray is
      // inside the double cone outside [t0, t1]; keep the forward-nappe half.
      let forward = dot(o + d * t1 - apex, AXIS) > 0.0;
      if (forward) { t0 = t1; t1 = 1e9; } else { t1 = t0; t0 = -1e9; }
    }
  }
  // Reject the mirror nappe.
  let probe = o + d * clamp(0.5 * (t0 + t1), max(t0, 0.0), max(t1, 0.0));
  if (dot(probe - apex, AXIS) < 0.0) { return vec2f(1.0, 0.0); }
  // Slab 0 <= s <= LENGTH along the axis (s measured from the nozzle).
  let sA = dot(o - NOZZLE, AXIS);
  if (abs(dd) > 1e-4) {
    var ts0 = (0.0 - sA) / dd;
    var ts1 = (LENGTH - sA) / dd;
    if (ts0 > ts1) { let tmp = ts0; ts0 = ts1; ts1 = tmp; }
    t0 = max(t0, ts0);
    t1 = min(t1, ts1);
  } else if (sA < 0.0 || sA > LENGTH) {
    return vec2f(1.0, 0.0);
  }
  t0 = max(t0, 0.0);
  return vec2f(t0, t1);
}

fn ign(p: vec2f) -> f32 {
  // Interleaved gradient noise: stable per-pixel jitter for the march start.
  return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

@fragment fn fs_main(@builtin(position) position: vec4f) -> FireOut {
  // This pass renders a quarter-size target: each texel stands for one
  // history pixel of the current 2x2 phase.
  let res = params.resolution;
  let historyPixel = floor(position.xy) * 2.0 + vec2f(f32(params.phase & 1u), f32(params.phase >> 1u)) + 0.5;
  let ndc = vec2f((historyPixel.x / res.x) * 2.0 - 1.0, 1.0 - (historyPixel.y / res.y) * 2.0);
  let dir = cameraRay(ndc);
  let origin = camera.position;
  let AXIS = plume.axis;
  let NOZZLE = plume.nozzle;
  let LENGTH = plume.length;
  let time = params.time * params.motion;

  // Geometry behind this pixel: distance from the camera, 0 where nothing
  // was drawn (the scene target is cleared to 0). The march never continues
  // behind a surface, and the surface shows through the remaining
  // transmittance instead of the sky.
  let scenePixel = vec2i(historyPixel * params.sceneScale);
  let surfaceDistance = textureLoad(sceneDepth, scenePixel, 0).r;
  let hasSurface = surfaceDistance > 0.0;
  var interval = coneInterval(origin, dir);
  if (hasSurface) { interval.y = min(interval.y, surfaceDistance); }

  // Heat haze: hot air around the plume refracts whatever is behind it. The
  // path length through the bounding cone says how close the ray passes to
  // the axis; a scrolling noise field jitters the background lookup.
  let pathThroughCone = max(interval.y - interval.x, 0.0);
  let heatHaze = smoothstep(0.0, plume.r0 * 3.0, pathThroughCone) * 0.6;
  let wobble = (textureSampleLevel(detail, detailSamp, historyPixel / 256.0 + vec2f(time * 0.35, -time * 1.6), 0.0).ba - 0.5) * 14.0 * heatHaze * params.sceneScale;
  var out: FireOut;
  out.aux = vec4f(wobble, surfaceDistance, 0.0);
  if (interval.y <= interval.x) {
    out.fire = vec4f(0.0, 0.0, 0.0, 1.0);
    return out;
  }

  let frame = plumeFrame(plume.axis);
  let coreWhite = vec3f(1.0, 0.86, 1.0) * 1.4; // lavender-white core, not thermal orange
  let sootCold = blackbody(1900.0);
  let sootHot = blackbody(2600.0);
  let dtWorld = (interval.y - interval.x) / f32(STEPS);
  // Per-pixel interleaved-gradient noise advanced by the golden ratio each
  // frame: successive frames land on different sub-steps, so the temporal
  // history averages them out and the step count stays low.
  var t = interval.x + dtWorld * fract(ign(historyPixel) + 0.6180339887 * params.frame);
  var color = vec3f(0.0);
  var transmittance = 1.0;

  for (var i = 0; i < STEPS; i++) {
    let sample = evaluatePlume(atlas, atlasSamp, detail, detailSamp, plume, frame, origin + dir * t, time, coreWhite, sootCold, sootHot);
    if (sample.a > 0.0 || dot(sample.rgb, sample.rgb) > 0.0) {
      let alpha = 1.0 - exp(-sample.a * dtWorld);
      color += transmittance * sample.rgb * dtWorld;
      transmittance *= 1.0 - alpha;
      if (transmittance < 0.012) { break; }
    }
    t += dtWorld;
  }

  out.fire = vec4f(color, transmittance);
  out.aux.w = heatHaze * 0.5;
  return out;
}
