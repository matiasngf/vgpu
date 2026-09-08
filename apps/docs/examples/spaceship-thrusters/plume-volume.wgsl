import { noise3 } from "./thruster-common.wgsl";

// Shared plume volume model: everything that turns a world-space point into
// emission and extinction. Used by grid.wgsl (which evaluates it once per
// frame into a cone-fitted slice atlas) and, through that atlas, by the
// raymarch in fire.wgsl.

export const PI: f32 = 3.14159265359;
// The march/grid bounds are wider than the nominal cone so eroded fibres fit.
export const BOUND_SCALE: f32 = 1.5;

// Plume placement in world space (nozzle exit point, unit axis, exit radius,
// radius growth per unit length, marched length) and light gains.
export struct Plume {
  nozzle: vec3f,
  r0: f32,
  axis: vec3f,
  spread: f32,
  length: f32,
  sootGain: f32,
  glowGain: f32,
  exitGain: f32,
}

// Cone-fitted grid: 256 slices along the axis, each a 64x64 cross-section of
// the bounding cone at that distance (plus a 1-texel clamp border so bilinear
// filtering never reads the neighbouring slice), packed 16x16 in one atlas.
export const GRID_TILE: f32 = 64.0;
export const GRID_BORDER: f32 = 1.0;
export const GRID_STRIDE: f32 = GRID_TILE + 2.0 * GRID_BORDER;
export const GRID_COLS: i32 = 16;
export const GRID_SLICES: i32 = 256;
export const GRID_ATLAS: f32 = GRID_STRIDE * f32(GRID_COLS); // 1056

export fn plumeFrame(axis: vec3f) -> mat3x3f {
  // Orthonormal basis (U, V, axis) for cylindrical coordinates.
  let helper = select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), abs(axis.z) > 0.9);
  let u = normalize(cross(axis, helper));
  let v = cross(axis, u);
  return mat3x3f(u, v, axis);
}

/** Radius of the bounding cone at axial distance s (world units). */
export fn boundRadius(plume: Plume, s: f32) -> f32 {
  return BOUND_SCALE * (plume.r0 + plume.spread * s);
}

/** Atlas uv of grid coordinate g in [-1, 1]^2 within `slice`. */
export fn gridTileUv(g: vec2f, slice: i32) -> vec2f {
  let col = slice % GRID_COLS;
  let row = slice / GRID_COLS;
  let texel = (g * 0.5 + 0.5) * GRID_TILE + GRID_BORDER + 0.5 + vec2f(f32(col), f32(row)) * GRID_STRIDE;
  return texel / GRID_ATLAS;
}

// --- Light model -------------------------------------------------------------
// Radiance is in scene-linear units where the sky sits around 0.1-0.3 and the
// plume core reaches well above 1, so the camera response in composite.wgsl
// clips it per channel the way a sensor does (R saturates first, then G, B).
//
// 1. Soot in the initial mixing zone radiates as a blackbody (1550-2800 K):
//    deep red fringe, orange body, yellow-white where it is hottest.
// 2. The exhaust gas itself glows through hydrogen Balmer / OH emission —
//    optically thin, magenta-violet. It dominates the core (clipping to
//    lavender-white) and the thin fringe. Chromaticities were sampled from a
//    night test-firing photo: exit jet (0.36, 0.48, 1.0), mid plume
//    (0.86, 0.70, 1.0), far plume (0.88, 0.52, 1.0) in linear ratios.
// 3. Near the exit, fuel-rich gas emits blue-violet Swan bands along the
//    engine jets, with white shock diamonds on the axis.
const GAS_GLOW: vec3f = vec3f(0.9, 0.48, 1.0);
const EXIT_GLOW: vec3f = vec3f(0.36, 0.5, 1.0);
const DIAMOND_GLOW: vec3f = vec3f(0.95, 0.92, 1.0);
// Jet speed in plume units per second (the plume is ~14.5 units long): the
// fine fibres stream at this speed, the large eddies that shape the
// silhouette lag behind like the shear layer does. ~7% of the plume length
// per frame at 30 fps, fast enough to read as exhaust yet still trackable.
const FLOW_SPEED: f32 = 30.0;

// Planckian-locus chromaticity (Kang et al. 2002 fit, valid 1667-4000 K)
// converted to linear sRGB with Y = 1, scaled by a T^4 luminance term.
export fn blackbody(temperature: f32) -> vec3f {
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
// radii (matched to a single-engine sea-level test firing):
//
// 1. fireProfile — the FIRE itself, one body from lip to tail: it leaves at
//    the lip width soft and translucent, is squeezed a little by ambient
//    pressure, then opens up and intensifies into the fibrous pink
//    afterburning plume.
// 2. glowProfile — the white GLOW inside it at the exit: a teardrop that
//    shrinks while it fades out.
export fn fireProfile(sR: f32) -> f32 {
  let squeeze = mix(1.0, 0.85, smoothstep(0.2, 2.0, sR));
  return mix(squeeze, 1.3, smoothstep(3.5, 14.0, sR));
}

export fn glowProfile(sR: f32) -> f32 {
  // Starts a little downstream of the lip (a visible gap of thin gas first),
  // ~0.6 of the exit wide, then narrows to ~0.3 and dissolves into the fire
  // there: it never closes to a point.
  return mix(0.6, 0.3, smoothstep(2.6, 5.6, sR));
}


/**
 * Emission (rgb, per world unit of ray) and extinction (a, per world unit) at
 * world point p. `coreWhite`, `sootCold`, `sootHot` are blackbody colours the
 * caller evaluates once.
 */
export fn evaluatePlume(
  atlas: texture_2d<f32>, atlasSamp: sampler, detail: texture_2d<f32>, detailSamp: sampler,
  plume: Plume, frame: mat3x3f, p: vec3f, time: f32,
  coreWhite: vec3f, sootCold: vec3f, sootHot: vec3f,
) -> vec4f {
  let NOZZLE = plume.nozzle;
  let AXIS = plume.axis;
  let LENGTH = plume.length;
  let rel = p - NOZZLE;
  let sWorld = dot(rel, AXIS);
  let q = rel - AXIS * sWorld;
  let sR = sWorld / plume.r0; // distance in exit radii
  let fireWorld = plume.r0 * fireProfile(sR) + plume.spread * max(sWorld - 12.0 * plume.r0, 0.0);
  let glowWorld = plume.r0 * glowProfile(sR);
  let radEnv = length(q) / fireWorld;   // fire body
  let radCore = length(q) / glowWorld;  // white exit glow
  // Cheap test first (Nubis-style): the noise can push the shell out by at
  // most ~0.5 radii, so beyond that no fetch can produce density.
  if (sWorld <= 0.0 || sWorld >= LENGTH || (radEnv > 1.5 && radCore > 1.6)) { return vec4f(0.0); }
  // Everything below is expressed in "plume units" (the look was tuned for
  // an exit radius of 0.3), so the same shader fits any engine size.
  let unit = 0.3 / plume.r0;
  // Everything below is expressed in "plume units" (the look was tuned for
  // an exit radius of 0.3), so the same shader fits any engine size.
  let s = sWorld * unit;
  let qx = dot(q, frame[0]) * unit;
  let qy = dot(q, frame[1]) * unit;
  let radius = fireWorld * unit;

  // Flow regimes along the jet, in exit radii:
  //   exitCore  — hot exhaust leaving the nozzle, bright white, full width
  //   machDisk  — the normal shock at the neck re-heats the envelope gas
  //   burn/heat — afterburning of CO/H2 with entrained air, the intense
  //               pink-white fire that ignites inside the envelope
  //   shock     — the exit-gas regime (blue-violet engine jets), fading out
  let exitCore = smoothstep(1.0, 1.8, sR) * (1.0 - smoothstep(3.4, 6.0, sR));
  let machDisk = exp(-pow((sR - 2.2) / 0.5, 2.0)) * smoothstep(1.0, 0.4, radEnv);
  let shock = 1.0 - smoothstep(0.0, 5.0, sR);
  // The fire starts soft and translucent and gains body as it opens up.
  let fireStrength = mix(0.06, 1.0, smoothstep(1.5, 8.5, sR));
  let burn = smoothstep(2.0, 8.0, sR);
  let heat = smoothstep(4.0, 11.0, sR);
  // Downstream the plume opens up and mixes out: it gets thinner and its
  // edge breaks into wisps, so the wider body does not get brighter.
  let disperse = smoothstep(3.5, 13.0, sR);

  // Soft body: domain-warped 3D fbm/billow, mildly stretched along the flow.
  // This only sets the low-frequency silhouette and opacity.
  // Advection: how far the flow has travelled along the axis, in plume
  // units. Each lookup subtracts it from its own s coordinate (scaled by
  // that lookup's frequency) so every layer moves at a physical speed.
  let travel = time * FLOW_SPEED;
  let warpN = noise3(atlas, atlasSamp, vec3f(qx, qy, (s - travel * 0.5) * 0.7) * 0.4 + vec3f(0.31, 0.77, 0.0));
  let warp = (vec2f(warpN.a, warpN.r) - 0.5) * (0.25 + 0.35 * burn) * radius;
  let warped = vec3f((qx + warp.x) * 1.4, (qy + warp.y) * 1.4, (s - travel * 0.6) * 0.75);
  let n = noise3(atlas, atlasSamp, warped);

  // Fibre field (the reference's dominant texture): ridged noise stretched
  // ~10x along the flow. One volumetric lookup (atlas .b) so fibres have
  // depth, plus a cylindrical 2D lookup (detail .g) for the fine hairs,
  // sheared outward with radius so the fringe fans out like a herringbone.
  let fib3 = noise3(atlas, atlasSamp, vec3f((qx + warp.x * 0.5) * 1.7, (qy + warp.y * 0.5) * 1.7, (s - travel) * 0.1) + vec3f(0.5, 0.2, 0.37)).b;
  let theta = atan2(qy, qx) / (2.0 * PI);
  let fibreUv = vec2f(theta * 7.0 + warp.x * 0.35, (s - radEnv * radius * 0.6 - travel) * 0.085);
  let fib2 = textureSampleLevel(detail, detailSamp, fibreUv, 0.0);
  // Knots: a nearly isotropic lookup along the flow breaks the streaks into
  // segments of varying brightness instead of uniform brush strokes. They
  // ride slightly slower than the fibres so the segments shimmer along them.
  let knots = textureSampleLevel(detail, detailSamp, vec2f(theta * 7.0 + 0.13, (s - travel * 0.85) * 0.55 + fib2.b * 0.2), 0.0).r;
  // fib2.a is the three-octave ridged stack baked into the detail texture.
  let filament = clamp((fib3 * 0.42 + fib2.a * 0.74) * (0.65 + 0.7 * knots), 0.0, 1.0);
  // Thin, high-contrast hairs: only the ridge tops light up.
  let hairs = smoothstep(0.55, 0.95, filament);

  // Further shock diamonds repeat past the first Mach disk and fade as the
  // shear layer mixes the jet with air.
  let phase = fract((sR - 4.5) / 1.5);
  let diamond = smoothstep(0.5, 0.05, abs(phase - 0.5) * 1.6 + radEnv * 0.9) * smoothstep(4.2, 5.0, sR) * (1.0 - smoothstep(6.0, 14.0, sR));

  let turb = (n.r - 0.5) * 2.0;
  let erosion = mix(0.12, 0.3, burn) + 0.22 * disperse;
  // Attach cleanly to the lip: turbulence only starts a little past the exit.
  let ramp = smoothstep(0.1, 0.8, s);
  let fadeEnd = smoothstep(0.0, 0.15, s) * pow(1.0 - smoothstep(6.0, LENGTH * unit, s), 1.5);

  // White exit glow: smooth, dense, opaque — the exhaust is still one solid
  // supersonic jet here, no fibres yet. Shrinks and fades along the tail.
  // Its edge is crisp where it appears and blurs out as it dissolves.
  // The edge is crisp where the glow appears and is fully blurred into the
  // surrounding fire well before it fades out.
  let glowSoftness = mix(0.2, 1.4, smoothstep(1.8, 5.0, sR));
  let capsuleDensity = smoothstep(0.0, glowSoftness, 1.0 - radCore + turb * 0.04 * ramp) * exitCore * fadeEnd;

  // The fire: one fibrous body from the lip onward. Fibres both erode the
  // shell and poke past it; near the exit it is thin and see-through.
  let shellFire = 1.0 - radEnv + (turb * erosion + (filament - 0.45) * (0.12 + 0.22 * burn + 0.12 * disperse) + (fib2.r - 0.5) * (0.12 + 0.08 * disperse)) * ramp;
  var density = smoothstep(0.0, 0.1 + 0.15 * disperse, shellFire);
  density *= 0.3 + 0.5 * n.g + 1.3 * hairs;
  // Downstream this body becomes the fringe: the tips that fan out and thin.
  density *= fadeEnd * fireStrength * mix(1.0, 0.22, disperse);

  // Downstream the fire splits in two (reference: a single-engine night
  // firing): the COLUMN keeps roughly the nozzle width and its opacity and
  // only warms from lavender-white to orange as soot heats up, while the
  // fringe above expands and dilutes. The column takes over from the shared
  // body as dispersion sets in.
  let columnWorld = plume.r0 * mix(0.9, 1.05, smoothstep(3.5, 16.0, sR));
  let radCol = length(q) / columnWorld;
  // Fibres erode the column's edge and carry most of its density, so the
  // long hairs and knots of the original fire stay visible inside it.
  let colShell = 1.0 - radCol + (turb * 0.25 + (filament - 0.45) * 0.38 + (fib2.r - 0.5) * 0.16) * ramp;
  let colDensity = smoothstep(0.0, 0.1, colShell) * (0.25 + 0.3 * n.g + 1.8 * hairs) * fadeEnd * fireStrength * disperse;
  let colCore = clamp(1.0 - radCol * radCol * 0.6, 0.0, 1.0);
  let colAxial = colCore * colCore;

  // Soot burns in the shear layer of the afterburner where the fuel-rich
  // gas meets air. Absorbing and emitting.
  let core = 1.0 - radEnv * radEnv * 0.45;
  let glowCore = 1.0 - radCore * radCore * 0.45;
  let sootFrac = smoothstep(4.0, 6.5, sR) * (1.0 - smoothstep(9.0, 16.0, sR)) * (0.55 + 0.45 * n.g) * smoothstep(0.35, 0.85, radEnv);
  let sootHeat = clamp((0.4 + 1.0 * hairs) * (0.7 + 0.5 * heat), 0.0, 1.0);
  let sootRadiance = mix(sootCold, sootHot, sootHeat) * plume.sootGain;

  // Gas glow: optically thin, so it adds along the ray instead of riding on
  // opacity. Fibres and the hot core carry most of it.
  let ridge = hairs * sqrt(hairs);
  // Radiance climbs steeply toward the axis: the shell just clips red on the
  // sensor, the core saturates every channel.
  let axial = core * core * core;
  // Emission per unit falls off hard as the gas mixes out: the far plume in
  // the reference is a mid-tone, only the exit region clips the sensor.
  let spent = mix(1.0, 0.2, disperse);
  // The afterburning gas starts out blue-violet just past the neck (the
  // reference's electric-blue jets) and turns magenta as it burns through.
  let warm = smoothstep(5.0, 14.0, sR);
  let gasTint = mix(mix(vec3f(0.45, 0.58, 1.0), GAS_GLOW, smoothstep(3.0, 8.5, sR)), vec3f(1.0, 0.62, 0.55), warm * 0.6);
  let glow = gasTint * (density * burn * (0.22 + 2.0 * axial + 2.8 * ridge) * plume.glowGain * spent)
    // The densest, hottest core also radiates thermally (lavender white).
    + coreWhite * (density * heat * axial * (0.35 + 1.5 * ridge) * 3.5 * spent)
    // Exhaust capsule leaving the nozzle: blue-white, full width.
    + mix(vec3f(0.42, 0.56, 1.0), vec3f(0.8, 0.86, 1.0), clamp(glowCore, 0.0, 1.0)) * (capsuleDensity * (0.6 + 0.8 * glowCore) * plume.exitGain * 12.0)
    // Near the exit the thin fire glows blue-violet (Swan bands).
    + mix(EXIT_GLOW, GAS_GLOW, 0.3) * (density * (1.0 - burn) * (0.6 + 0.4 * hairs) * plume.exitGain * 0.8)
    // Mach disk: a thin bright re-heated slab at the squeeze.
    + DIAMOND_GLOW * (density * machDisk * plume.exitGain * 0.25);
  // The column: bright on the axis (clips white on the sensor), orange at its
  // edge once the soot has warmed it, magenta-white before that.
  // Orange only at the rim; the axis is hot enough to clip every channel, so
  // the column reads as incandescent rather than as a painted orange body.
  let rimTint = mix(vec3f(0.9, 0.6, 1.0), vec3f(1.0, 0.4, 0.14), warm);
  let coreTint = mix(vec3f(1.0, 0.85, 1.0), vec3f(1.0, 0.8, 0.55), warm);
  let columnTint = mix(rimTint, coreTint, colAxial);
  // Only the hair ridges clip; between them the column sits in the mid-tones,
  // so the fibre structure reads instead of a flat clipped band.
  let columnGlow = columnTint * (colDensity * (0.15 + 1.2 * colAxial + 5.5 * ridge) * plume.glowGain * 0.9);

  // Exit region: discrete engine jets read as sharp parallel streaks of
  // blue-violet gas, with the first diamonds glowing warm white.
  let jets = smoothstep(0.5, 0.9, fib2.g * 0.6 + fib3 * 0.5);
  let hazeDensity = smoothstep(0.0, 0.08, 1.0 - radEnv + (fib2.r - 0.5) * 0.08) * (0.2 + 1.1 * jets + 0.3 * diamond) * 0.18 * shock;
  let exitGlow = (EXIT_GLOW * (0.25 + 1.2 * jets) + DIAMOND_GLOW * diamond * 1.5) * hazeDensity * plume.exitGain * 1.1;
  let diamondGlow = DIAMOND_GLOW * diamond * density * burn * 2.5 * spent;

  // High absorption in the fire keeps its visible layer thin, so fibres at
  // different depths do not average into mush; the envelope stays thin.
  // Mixed-out gas downstream is far less opaque, so the widened plume stays
  // see-through instead of a solid bright body.
  let sigma = density * (0.4 + 8.0 * burn) * mix(1.0, 1.4, sootFrac) * mix(1.0, 0.25, disperse) + colDensity * (0.4 + 8.0 * burn) + capsuleDensity * 6.0 + hazeDensity * 0.35;
  // Soot rides on opacity in the original integrator; per unit length that is
  // its radiance times the extinction. Convert plume units to world units.
  let emission = sootRadiance * sootFrac * sigma * spent + glow + columnGlow + exitGlow + diamondGlow;
  return vec4f(emission * unit, sigma * unit);
}
