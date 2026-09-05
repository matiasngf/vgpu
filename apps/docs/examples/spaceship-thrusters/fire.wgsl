import { noise3 } from "./thruster-common.wgsl";

// Raymarched exhaust plume rendered as scene-linear radiance into an HDR target.
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
// Noise style (from further launch photos): the dominant texture is long,
// translucent fibres stretched ~10:1 along the flow at several scales, not
// cauliflower billows; the fringe fans outward like a herringbone; the exit
// shows discrete parallel engine jets; and periodic shock diamonds sit on the
// axis. Billow noise only shapes the soft underlying body.
//
// The plume is a cone volume (nozzle radius r0, linear spread). Every march
// step does: 6 atlas fetches (warp, body, fibres — 2 slices each) and five
// detail fetches — no runtime octave loops. Quality mode: full-resolution
// pass and 64 steps; the optimization pass comes once the look is locked. Ray/cone intersection bounds the
// march to the volume, and the loop exits early once transmittance is spent.

struct Params {
  resolution: vec2f,
  time: f32,
  motion: f32,
  sceneScale: vec2f, // scene texels per fire texel
}

// Camera: rays are unprojected from NDC with the inverse view-projection.
struct Camera {
  invViewProj: mat4x4f,
  position: vec3f,
}

// Plume placement in world space (nozzle exit point, unit axis, exit radius,
// radius growth per unit length, marched length).
struct Plume {
  nozzle: vec3f,
  r0: f32,
  axis: vec3f,
  spread: f32,
  length: f32,
  // Light gains (scene-linear): soot blackbody, hydrogen glow, exit jets.
  sootGain: f32,
  glowGain: f32,
  exitGain: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(5) var<uniform> camera: Camera;
@group(0) @binding(6) var<uniform> plume: Plume;
// Lit geometry (scene-linear radiance) and its camera distance, from scene.wgsl.
@group(0) @binding(7) var sceneColor: texture_2d<f32>;
@group(0) @binding(8) var sceneDepth: texture_2d<f32>;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var detail: texture_2d<f32>;
@group(0) @binding(3) var atlasSamp: sampler;
@group(0) @binding(4) var detailSamp: sampler;

const PI: f32 = 3.14159265359;
const STEPS: i32 = 64;
const BOUND_SCALE: f32 = 1.5;  // march bounds are wider than the nominal cone

fn plumeFrame() -> mat3x3f {
  // Orthonormal basis (U, V, axis) for cylindrical coordinates.
  let helper = select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), abs(plume.axis.z) > 0.9);
  let u = normalize(cross(plume.axis, helper));
  let v = cross(plume.axis, u);
  return mat3x3f(u, v, plume.axis);
}

fn cameraRay(ndc: vec2f) -> vec3f {
  let nearPoint = camera.invViewProj * vec4f(ndc, 0.0, 1.0);
  let farPoint = camera.invViewProj * vec4f(ndc, 1.0, 1.0);
  return normalize(farPoint.xyz / farPoint.w - nearPoint.xyz / nearPoint.w);
}

fn sky(dir: vec3f) -> vec3f {
  // Teal sky measured from the reference: darker toward the top.
  let t = smoothstep(-0.55, 0.45, -dir.y);
  let top = vec3f(0.070, 0.205, 0.265);
  let bottom = vec3f(0.140, 0.355, 0.440);
  return mix(top, bottom, t);
}

// --- Light model -------------------------------------------------------------
// Radiance is in scene-linear units where the sky sits around 0.1-0.3 and the
// plume core reaches well above 1, so the camera response in composite.wgsl
// clips it per channel the way a sensor does (R saturates first, then G, B).
//
// 1. Soot in the initial mixing zone radiates as a blackbody (1550-2800 K):
//    deep red fringe, orange body, yellow-white where it is hottest.
// 2. The exhaust gas itself glows through hydrogen Balmer / OH emission —
//    optically thin, magenta-pink. It dominates the core (clipping to
//    pink-white) and the thin fringe, which mixes additively with the teal
//    sky into the lavender edge seen in the reference.
// 3. Near the exit, fuel-rich gas emits blue-violet Swan bands along the
//    engine jets, with warm-white shock diamonds on the axis.
const GAS_GLOW: vec3f = vec3f(1.0, 0.27, 0.28);
const EXIT_GLOW: vec3f = vec3f(0.42, 0.52, 1.0);
const DIAMOND_GLOW: vec3f = vec3f(1.0, 0.92, 1.0);

// Planckian-locus chromaticity (Kang et al. 2002 fit, valid 1667-4000 K)
// converted to linear sRGB with Y = 1, scaled by a T^4 luminance term.
fn blackbody(temperature: f32) -> vec3f {
  let T = clamp(temperature, 1667.0, 4000.0);
  let x = -0.2661239e9 / (T * T * T) - 0.2343589e6 / (T * T) + 0.8776956e3 / T + 0.179910;
  let y = select(
    -0.9549476 * x * x * x - 1.37418593 * x * x + 2.09137015 * x - 0.16748867,
    -1.1063814 * x * x * x - 1.34811020 * x * x + 2.18555832 * x - 0.20219683,
    T < 2222.0,
  );
  let X = x / y;
  let Z = (1.0 - x - y) / y;
  let rgb = vec3f(
    3.2406 * X - 1.5372 - 0.4986 * Z,
    -0.9689 * X + 1.8758 + 0.0415 * Z,
    0.0557 * X - 0.2040 + 1.0570 * Z,
  );
  let luminance = pow(temperature / 2600.0, 4.0);
  return max(rgb, vec3f(0.0)) * luminance;
}

// Two nested bodies, both in exit radii as a function of distance in exit
// radii (measured on a single-engine sea-level test and matching the
// over-expanded-jet + afterburning physics):
//
// 1. envelopeProfile — the hot exhaust GAS. Leaves at the lip width, is
//    squeezed gently by ambient pressure toward the first Mach disk, then
//    drifts wider as it mixes with air. Translucent, faint violet-pink, the
//    whole way.
// 2. coreProfile — the bright FIRE inside it: the exhaust capsule right at the
//    exit, then nothing through the induction gap, then the afterburner that
//    ignites narrow inside the envelope and grows until it overflows it.
fn envelopeProfile(sR: f32) -> f32 {
  return mix(1.0, 0.8, smoothstep(0.3, 2.5, sR)) + 0.035 * max(sR - 3.0, 0.0);
}

fn coreProfile(sR: f32) -> f32 {
  let capsule = 0.85 * (1.0 - smoothstep(1.8, 3.4, sR));
  let burner = smoothstep(4.0, 6.0, sR) * mix(0.22, 1.0, smoothstep(5.0, 11.0, sR)) * mix(1.0, 1.7, smoothstep(10.0, 18.0, sR));
  return max(capsule, burner);
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

@fragment fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let res = params.resolution;
  let ndc = vec2f((position.x / res.x) * 2.0 - 1.0, 1.0 - (position.y / res.y) * 2.0);
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
  let scenePixel = vec2i(position.xy * params.sceneScale);
  let surfaceDistance = textureLoad(sceneDepth, scenePixel, 0).r;
  let hasSurface = surfaceDistance > 0.0;
  var interval = coneInterval(origin, dir);
  if (hasSurface) { interval.y = min(interval.y, surfaceDistance); }

  // Heat haze: hot air around the plume refracts whatever is behind it. The
  // path length through the bounding cone says how close the ray passes to
  // the axis; a scrolling noise field jitters the background lookup.
  let pathThroughCone = max(interval.y - interval.x, 0.0);
  let heatHaze = smoothstep(0.0, plume.r0 * 3.0, pathThroughCone) * 0.6;
  let wobble = (textureSampleLevel(detail, detailSamp, position.xy / 256.0 + vec2f(time * 0.35, -time * 1.6), 0.0).ba - 0.5) * 14.0 * heatHaze;
  let hazePixel = clamp(scenePixel + vec2i(wobble * params.sceneScale), vec2i(0), vec2i(textureDimensions(sceneDepth)) - 1);
  let hazeSurface = textureLoad(sceneDepth, hazePixel, 0).r > 0.0;
  let background = select(sky(dir), textureLoad(sceneColor, select(scenePixel, hazePixel, hazeSurface), 0).rgb, hasSurface);
  if (interval.y <= interval.x) {
    return vec4f(background, 1.0);
  }

  let frame = plumeFrame();
  let coreWhite = blackbody(2900.0);
  let dtWorld = (interval.y - interval.x) / f32(STEPS);
  var t = interval.x + dtWorld * ign(position.xy);
  var color = vec3f(0.0);
  var transmittance = 1.0;
  var haze = 0.0; // accumulated near-nozzle gas, used for a heat-shimmer tint

  for (var i = 0; i < STEPS; i++) {
    let p = origin + dir * t;
    let rel = p - NOZZLE;
    let sWorld = dot(rel, AXIS);
    let q = rel - AXIS * sWorld;
    let sR = sWorld / plume.r0; // distance in exit radii
    let envelopeWorld = plume.r0 * envelopeProfile(sR) + plume.spread * max(sWorld - 10.0 * plume.r0, 0.0);
    let coreWorld = plume.r0 * max(coreProfile(sR), 0.02) + plume.spread * max(sWorld - 12.0 * plume.r0, 0.0);
    let radEnv = length(q) / envelopeWorld;
    let radCore = length(q) / coreWorld;
    // Everything below is expressed in "plume units" (the look was tuned for
    // an exit radius of 0.3), so the same shader fits any engine size.
    let unit = 0.3 / plume.r0;
    let s = sWorld * unit;
    let qx = dot(q, frame[0]) * unit;
    let qy = dot(q, frame[1]) * unit;
    let radius = envelopeWorld * unit;
    let dt = dtWorld * unit;

    // Flow regimes along the jet, in exit radii:
    //   exitCore  — hot exhaust leaving the nozzle, bright white, full width
    //   machDisk  — the normal shock at the neck re-heats the envelope gas
    //   burn/heat — afterburning of CO/H2 with entrained air, the intense
    //               pink-white fire that ignites inside the envelope
    //   shock     — the exit-gas regime (blue-violet engine jets), fading out
    let exitCore = 1.0 - smoothstep(1.5, 2.6, sR);
    let machDisk = exp(-pow((sR - 3.0) / 0.5, 2.0)) * smoothstep(1.0, 0.4, radEnv);
    let shock = 1.0 - smoothstep(0.0, 5.0, sR);
    let burn = smoothstep(4.2, 6.5, sR);
    let heat = smoothstep(5.0, 8.5, sR);

    // Soft body: domain-warped 3D fbm/billow, mildly stretched along the flow.
    // This only sets the low-frequency silhouette and opacity.
    let flow = vec3f(0.0, 0.0, -time * 1.3);
    let warpN = noise3(atlas, atlasSamp, vec3f(qx, qy, s * 0.7) * 0.4 + flow * 0.55 + vec3f(0.31, 0.77, 0.0));
    let warp = (vec2f(warpN.a, warpN.r) - 0.5) * (0.25 + 0.35 * burn) * radius;
    let warped = vec3f((qx + warp.x) * 1.4, (qy + warp.y) * 1.4, s * 0.75);
    let n = noise3(atlas, atlasSamp, warped + flow);

    // Fibre field (the reference's dominant texture): ridged noise stretched
    // ~10x along the flow. One volumetric lookup (atlas .b) so fibres have
    // depth, plus a cylindrical 2D lookup (detail .g) for the fine hairs,
    // sheared outward with radius so the fringe fans out like a herringbone.
    let fib3 = noise3(atlas, atlasSamp, vec3f((qx + warp.x * 0.5) * 1.7, (qy + warp.y * 0.5) * 1.7, s * 0.1) + flow * 0.45 + vec3f(0.5, 0.2, 0.37)).b;
    let theta = atan2(qy, qx) / (2.0 * PI);
    let fibreUv = vec2f(theta * 7.0 + warp.x * 0.35, (s - radEnv * radius * 0.6) * 0.085 - time * 0.75);
    let fib2 = textureSampleLevel(detail, detailSamp, fibreUv, 0.0);
    let fib2b = textureSampleLevel(detail, detailSamp, fibreUv * vec2f(2.7, 2.1) + vec2f(0.37, 0.11), 0.0);
    let fib2c = textureSampleLevel(detail, detailSamp, fibreUv * vec2f(6.1, 4.3) + vec2f(0.71, 0.53), 0.0);
    // Knots: a nearly isotropic lookup along the flow breaks the streaks into
    // segments of varying brightness instead of uniform brush strokes.
    let knots = textureSampleLevel(detail, detailSamp, vec2f(theta * 7.0 + 0.13, s * 0.55 - time * 0.75 + fib2.b * 0.2), 0.0).r;
    let filament = clamp((fib3 * 0.42 + fib2.g * 0.32 + fib2b.g * 0.26 + fib2c.g * 0.16) * (0.65 + 0.7 * knots), 0.0, 1.0);
    // Thin, high-contrast hairs: only the ridge tops light up.
    let hairs = smoothstep(0.55, 0.95, filament);

    // Further shock diamonds repeat past the first Mach disk and fade as the
    // shear layer mixes the jet with air.
    let phase = fract((sR - 4.5) / 1.5);
    let diamond = smoothstep(0.5, 0.05, abs(phase - 0.5) * 1.6 + radEnv * 0.9) * smoothstep(4.2, 5.0, sR) * (1.0 - smoothstep(6.0, 14.0, sR));

    let turb = (n.r - 0.5) * 2.0;
    let erosion = mix(0.12, 0.3, burn);
    // Attach cleanly to the lip: turbulence only starts a little past the exit.
    let ramp = smoothstep(0.1, 0.8, s);
    let fadeEnd = smoothstep(0.0, 0.15, s) * pow(1.0 - smoothstep(6.0, LENGTH * unit, s), 1.5);

    // Envelope: smooth, translucent hot gas with only a little edge breakup.
    let shellEnv = 1.0 - radEnv + (turb * 0.12 + (fib2.r - 0.5) * 0.1) * ramp;
    let densityEnv = smoothstep(0.0, 0.15, shellEnv) * (0.7 + 0.3 * n.g) * fadeEnd;

    // Exit capsule: smooth, dense, opaque white — the exhaust is still one
    // solid supersonic jet here, no fibres yet.
    let capsuleDensity = smoothstep(0.0, 0.2, 1.0 - radCore + turb * 0.05 * ramp) * exitCore * fadeEnd;

    // Core: the fibrous fire. Fibres both erode the shell and poke past it.
    let shellCore = 1.0 - radCore + (turb * erosion + (filament - 0.45) * (0.12 + 0.22 * burn) + (fib2.r - 0.5) * 0.12) * ramp;
    var density = smoothstep(0.0, 0.1, shellCore);
    density *= 0.3 + 0.5 * n.g + 1.3 * hairs;
    density *= fadeEnd;

    // Soot burns in the shear layer of the afterburner where the fuel-rich
    // gas meets air. Absorbing and emitting.
    let core = 1.0 - radCore * radCore * 0.45;
    let sootFrac = smoothstep(5.0, 6.5, sR) * (1.0 - smoothstep(9.0, 16.0, sR)) * (0.55 + 0.45 * n.g) * smoothstep(0.35, 0.85, radCore);
    let sootT = 1900.0 + 700.0 * clamp((0.4 + 1.0 * hairs) * (0.7 + 0.5 * heat), 0.0, 1.0);
    let sootRadiance = blackbody(sootT) * plume.sootGain;

    // Gas glow: optically thin, so it adds along the ray instead of riding on
    // opacity. Fibres and the hot core carry most of it.
    let ridge = hairs * sqrt(hairs);
    // Radiance climbs steeply toward the axis: the shell just clips red on the
    // sensor, the core saturates every channel.
    let axial = core * core * core;
    let glow = GAS_GLOW * (density * burn * (0.22 + 2.0 * axial + 2.8 * ridge) * plume.glowGain)
      // The densest, hottest core also radiates thermally (warm white).
      + coreWhite * (density * heat * axial * (0.35 + 1.5 * ridge) * 7.0)
      // Exhaust capsule leaving the nozzle: saturated white-blue, full width.
      + mix(vec3f(1.0, 0.6, 0.8), vec3f(1.0, 0.98, 1.0), clamp(core, 0.0, 1.0)) * (capsuleDensity * (0.6 + 0.8 * core) * plume.exitGain * 12.0)
      // The hot-gas envelope glows faintly violet near the exit, pink further
      // out, brighter around the capsule and along the fibres.
      + mix(EXIT_GLOW, GAS_GLOW, smoothstep(2.0, 8.0, sR)) * (densityEnv * (0.5 + 0.5 * hairs) * plume.exitGain * 0.28)
      // Mach disk: a thin bright re-heated slab in the envelope at the neck.
      + DIAMOND_GLOW * (densityEnv * machDisk * plume.exitGain * 0.5);

    // Exit region: discrete engine jets read as sharp parallel streaks of
    // blue-violet gas, with the first diamonds glowing warm white.
    let jets = smoothstep(0.5, 0.9, fib2.g * 0.6 + fib3 * 0.5);
    let hazeDensity = smoothstep(0.0, 0.08, 1.0 - radEnv + (fib2.r - 0.5) * 0.08) * (0.2 + 1.1 * jets + 0.3 * diamond) * 0.7 * shock;
    let exitGlow = (EXIT_GLOW * (0.25 + 1.2 * jets) + DIAMOND_GLOW * diamond * 1.5) * hazeDensity * plume.exitGain;
    let diamondGlow = DIAMOND_GLOW * diamond * density * burn * 2.5;

    // High absorption in the fire keeps its visible layer thin, so fibres at
    // different depths do not average into mush; the envelope stays thin.
    let sigma = density * (0.4 + 8.0 * burn) * mix(1.0, 1.4, sootFrac) + capsuleDensity * 6.0 + densityEnv * 0.7 + hazeDensity * 0.35;
    let alpha = 1.0 - exp(-sigma * dt);
    color += transmittance * (sootRadiance * sootFrac * alpha + (glow + exitGlow + diamondGlow) * dt);
    haze += transmittance * hazeDensity * dt;
    transmittance *= 1.0 - alpha;
    if (transmittance < 0.012) { break; }
    t += dtWorld;
  }

  // The near-nozzle gas is mostly transparent: let a slightly cooled, brighter
  // sky show through it so the exit reads as hot glass rather than smoke.
  let shimmer = clamp(haze * 1.6, 0.0, 1.0) * select(1.0, 0.0, hasSurface);
  let seenSky = mix(background, background * vec3f(1.15, 1.2, 1.25) + vec3f(0.03, 0.05, 0.06), shimmer);
  let result = color + transmittance * seenSky;
  return vec4f(result, 1.0);
}
