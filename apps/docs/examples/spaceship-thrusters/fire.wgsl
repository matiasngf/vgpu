import { noise3 } from "./thruster-common.wgsl";

// Raymarched exhaust plume. Runs at half resolution into an HDR target.
//
// Reference analysis (crops of a rocket-stage photo, sRGB 8-bit means):
//   sky            (82, 135, 149)  teal, darker toward the top
//   exit gas       translucent silver-blue cylinder with longitudinal streaks
//   afterburn body (203, 110, 79)  orange, feathered edge eroded by wisps
//   plume core     (248, 212, 205) pink-white, widens to ~60% of the plume
//   edge halo      (169, 113, 80)  fades into the sky through grey-pink
// The plume brightens DOWNSTREAM (fuel-rich exhaust meets air), so heat ramps
// up with distance from the nozzle instead of decaying.
//
// The plume is a cone volume (nozzle radius r0, linear spread). Every march
// step does: 2 atlas fetches (3D noise), 2 atlas fetches (domain warp) and one
// detail fetch — no runtime octave loops. Ray/cone intersection bounds the
// march to the volume, and the loop exits early once transmittance is spent.

struct Params {
  resolution: vec2f,
  time: f32,
  motion: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var detail: texture_2d<f32>;
@group(0) @binding(3) var atlasSamp: sampler;
@group(0) @binding(4) var detailSamp: sampler;

const PI: f32 = 3.14159265359;
const STEPS: i32 = 40;

// Camera (eye at origin looking down -Z).
const FOV_TAN: f32 = 0.4452; // tan(24°)

// Plume geometry in world units.
const NOZZLE: vec3f = vec3f(-1.05, 1.5, -3.9);
const AXIS: vec3f = vec3f(0.3707, -0.9268, 0.0556); // normalized
const R0: f32 = 0.30;        // exit radius
const SPREAD: f32 = 0.095;   // radius growth per unit length
const LENGTH: f32 = 8.0;
const BOUND_SCALE: f32 = 1.35; // march bounds are wider than the nominal cone

fn plumeFrame() -> mat3x3f {
  // Orthonormal basis (U, V, AXIS) for cylindrical coordinates.
  let u = normalize(cross(AXIS, vec3f(0.0, 0.0, 1.0)));
  let v = cross(AXIS, u);
  return mat3x3f(u, v, AXIS);
}

fn sky(dir: vec3f) -> vec3f {
  // Teal sky measured from the reference: darker toward the top.
  let t = smoothstep(-0.55, 0.45, -dir.y);
  let top = vec3f(0.045, 0.135, 0.175);
  let bottom = vec3f(0.092, 0.235, 0.290);
  return mix(top, bottom, t);
}

// Emission per unit opacity as a function of energy `e`.
// Reference colors (sRGB → linear): orange body (0.60,0.16,0.08),
// salmon core (0.94,0.66,0.61), highlights close to white with a pink cast.
fn fireRamp(e: f32) -> vec3f {
  let c0 = vec3f(0.42, 0.045, 0.008);   // deep red fringe
  let c1 = vec3f(1.15, 0.28, 0.055);    // orange
  let c2 = vec3f(1.75, 0.92, 0.62);     // salmon
  let c3 = vec3f(2.9, 2.5, 2.4);        // pink-white core (HDR)
  var c = mix(c0, c1, smoothstep(0.0, 0.3, e));
  c = mix(c, c2, smoothstep(0.3, 0.7, e));
  c = mix(c, c3, smoothstep(0.68, 1.15, e));
  return c;
}

// Ray interval inside the (widened) bounding cone, clipped to 0 <= s <= LENGTH.
fn coneInterval(o: vec3f, d: vec3f) -> vec2f {
  let k = SPREAD * BOUND_SCALE;
  let r0 = R0 * BOUND_SCALE;
  let apex = NOZZLE - AXIS * (r0 / k);
  let cos2 = 1.0 / (1.0 + k * k);
  let w = o - apex;
  let dd = dot(d, AXIS);
  let wD = dot(w, AXIS);
  let a = dd * dd - cos2;
  let b = 2.0 * (dd * wD - dot(d, w) * cos2);
  let c = wD * wD - dot(w, w) * cos2;
  let disc = b * b - 4.0 * a * c;
  if (disc < 0.0 || a >= 0.0) { return vec2f(1.0, 0.0); }
  let sq = sqrt(disc);
  var t0 = (-b + sq) / (2.0 * a);
  var t1 = (-b - sq) / (2.0 * a);
  if (t0 > t1) { let tmp = t0; t0 = t1; t1 = tmp; }
  // Reject the mirror nappe.
  let mid = o + d * (0.5 * (t0 + t1));
  if (dot(mid - apex, AXIS) < 0.0) { return vec2f(1.0, 0.0); }
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

@fragment fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let res = params.resolution;
  let ndc = (position.xy / res) * 2.0 - 1.0;
  let aspect = res.x / res.y;
  let dir = normalize(vec3f(ndc.x * aspect * FOV_TAN, -ndc.y * FOV_TAN, -1.0));
  let origin = vec3f(0.0);
  let time = params.time * params.motion;

  let background = sky(dir);
  let interval = coneInterval(origin, dir);
  if (interval.y <= interval.x) {
    return vec4f(background, 1.0);
  }

  let frame = plumeFrame();
  let dt = (interval.y - interval.x) / f32(STEPS);
  var t = interval.x + dt * ign(position.xy);
  var color = vec3f(0.0);
  var transmittance = 1.0;
  var haze = 0.0; // accumulated near-nozzle gas, used for a heat-shimmer tint

  for (var i = 0; i < STEPS; i++) {
    let p = origin + dir * t;
    let rel = p - NOZZLE;
    let s = dot(rel, AXIS);
    let q = rel - AXIS * s;
    let qx = dot(q, frame[0]);
    let qy = dot(q, frame[1]);
    let radius = R0 + SPREAD * s;
    let rad = length(q) / radius;

    // Downstream regimes: `shock` is the translucent exit region, `burn` the
    // afterburning mixing region further down.
    let shock = 1.0 - smoothstep(0.0, 2.4, s);
    let burn = smoothstep(0.4, 3.2, s);
    let heat = smoothstep(1.0, 3.8, s);

    // Domain warp from the low-frequency channel, then two 3D lookups at
    // different frequencies. Coordinates scroll along the axis so the
    // turbulence flows downstream; the second lookup scrolls faster so the
    // pattern evolves instead of sliding rigidly.
    let flow = vec3f(0.0, 0.0, -time * 1.3);
    let local = vec3f(qx, qy, s * 0.7);
    let warpN = noise3(atlas, atlasSamp, local * 0.4 + flow * 0.55 + vec3f(0.31, 0.77, 0.0));
    let warp = (vec2f(warpN.a, warpN.r) - 0.5) * (0.5 + 1.0 * burn) * radius;
    let warped = vec3f((qx + warp.x) * 1.4, (qy + warp.y) * 1.4, s * 0.75);
    let n = noise3(atlas, atlasSamp, warped + flow);
    let n2 = noise3(atlas, atlasSamp, warped * 3.1 + flow * 1.9 + vec3f(0.5, 0.2, 0.37));

    // Fine grain: cylindrical mapping stretched along the flow gives streaks.
    let theta = atan2(qy, qx) / (2.0 * PI);
    let detScaleS = mix(0.3, 0.12, shock);
    let det = textureSampleLevel(detail, detailSamp, vec2f(theta * 3.0 + warp.x * 0.4, s * detScaleS - time * 1.1), 0.0);
    let streak = smoothstep(0.3, 0.8, det.r);

    let turb = (n.r - 0.5) * 2.0 + (n2.r - 0.5) * 1.0 + (n.b - 0.5) * 0.6 * smoothstep(0.5, 1.1, rad);
    let erosion = mix(0.25, 0.9, burn);
    let shell = 1.0 - rad + turb * erosion + (det.r - 0.5) * (0.1 + 0.4 * burn);
    var density = smoothstep(0.0, 0.1, shell);
    density *= 0.35 + 1.0 * n.g * (0.6 + 0.8 * n2.g);
    density *= smoothstep(0.0, 0.2, s) * (1.0 - 0.4 * smoothstep(5.5, LENGTH, s));

    // Energy: heat builds up in the afterburn region, hotter toward the core,
    // and the lumps carry the heat so the structure shows in the color too.
    let core = 1.0 - rad * rad * 0.45;
    var energy = burn * core * (0.55 + 0.6 * n.g) * (0.8 + 0.4 * n2.g) * (0.9 + 0.2 * det.r);
    energy *= 0.7 + 0.55 * heat;
    energy *= 1.0 - 0.2 * smoothstep(5.0, LENGTH, s);

    // Shock region: faint silver-blue gas with longitudinal streaks and a
    // couple of soft diamond bands on the axis.
    let bands = pow(max(0.0, sin(s * 5.5 + 0.7)), 5.0) * smoothstep(0.55, 0.05, rad) * 0.6;
    let hazeColor = (vec3f(0.26, 0.38, 0.52) * (0.55 + 0.5 * streak) + vec3f(1.2, 1.1, 1.0) * bands) * shock;
    // Crisp cylinder silhouette: barely any erosion, so the limb brightens
    // where the ray path through the gas is longest.
    let hazeDensity = smoothstep(0.0, 0.08, 1.0 - rad + (det.r - 0.5) * 0.08 + (n.r - 0.5) * 0.06) * (0.5 + 0.5 * streak) * 0.65 * shock;

    let sigma = density * (0.4 + 4.5 * burn) + hazeDensity * 0.9;
    let alpha = 1.0 - exp(-sigma * dt);
    let perOpacity = (fireRamp(energy) * density * (0.4 + 4.5 * burn) * burn + hazeColor * hazeDensity * 0.9) / max(sigma, 1e-4);
    color += transmittance * perOpacity * alpha;
    haze += transmittance * hazeDensity * dt;
    transmittance *= 1.0 - alpha;
    if (transmittance < 0.012) { break; }
    t += dt;
  }

  // The near-nozzle gas is mostly transparent: let a slightly cooled, brighter
  // sky show through it so the exit reads as hot glass rather than smoke.
  let shimmer = clamp(haze * 1.6, 0.0, 1.0);
  let seenSky = mix(background, background * vec3f(1.15, 1.2, 1.25) + vec3f(0.03, 0.05, 0.06), shimmer);
  let result = color + transmittance * seenSky;
  return vec4f(result, 1.0);
}
