// Dreamcore: a Bliss-like meadow at night with a door standing in the grass. Through the
// opening there is a sunlit sand dune; as `phase` rises the day pours out of the door and
// sweeps across the hills.
//
// Fullscreen raymarcher: heightfield hills, a fur-style grass volume (a combed strand
// density field ray marched with front-to-back compositing, Kajiya-Kay fibre shading, depth
// occlusion and light transmittance), an SDF door, a rectangular area light for the door
// spill, single scattering in the night air, and a second heightfield world (the dune)
// behind the portal.
//
// Grass techniques follow Kajiya & Kay (fur as a lit 3D texture, anisotropic fibre shading)
// and Boulanger et al. (lit grass volume with occlusion and shadows through the layer).

struct Params {
  resolution: vec2f,
  time: f32,
  phase: f32,        // 0 = night, 1 = day (the day expands out of the door)
  camera: vec4f,     // height, pitch (rad, + looks up), vertical fov (rad), aa samples
  door: vec4f,       // x, z, yaw (rad), leaf angle (rad)
  look: vec4f,       // sun azimuth (rad), sun elevation (rad), texture strength, door light
  grass: vec4f,      // blade radius (m), blade height (m), blade shadow rays (0/1), wind
}

@group(0) @binding(0) var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const DOOR_W: f32 = 0.92;   // opening width
const DOOR_H: f32 = 2.08;   // opening height
const FRAME_T: f32 = 0.085; // frame member width
const FRAME_D: f32 = 0.07;  // frame half depth
const LEAF_D: f32 = 0.022;  // leaf half thickness
const TMAX: f32 = 170.0;


// ---------------------------------------------------------------- utils

fn srgb2lin(c: vec3f) -> vec3f {
  return pow((c + vec3f(0.055)) / 1.055, vec3f(2.4));
}

fn rgb8(r: f32, g: f32, b: f32) -> vec3f {
  return srgb2lin(vec3f(r, g, b) / 255.0);
}

fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + vec3f(33.33));
  return fract((p3.x + p3.y) * p3.z);
}

fn hash13(p: vec3f) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 += dot(p3, p3.zyx + vec3f(31.32));
  return fract((p3.x + p3.y) * p3.z);
}

fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  return mix(mix(hash12(i), hash12(i + vec2f(1.0, 0.0)), u.x),
             mix(hash12(i + vec2f(0.0, 1.0)), hash12(i + vec2f(1.0, 1.0)), u.x), u.y);
}

fn fbm(p0: vec2f) -> f32 {
  var p = p0;
  var v = 0.0;
  var a = 0.5;
  for (var i = 0; i < 4; i++) {
    v += a * vnoise(p);
    p = mat2x2f(1.6, 1.2, -1.2, 1.6) * p;
    a *= 0.5;
  }
  return v;
}

fn fbm3(p0: vec2f) -> f32 {
  var p = p0;
  var v = 0.0;
  var a = 0.5;
  for (var i = 0; i < 3; i++) {
    v += a * vnoise(p);
    p = mat2x2f(1.6, 1.2, -1.2, 1.6) * p;
    a *= 0.5;
  }
  return v;
}

// ---------------------------------------------------------------- terrain

fn bump(p: vec2f, c: vec2f, r: vec2f) -> f32 {
  let d = (p - c) / r;
  return exp(-dot(d, d));
}

fn terrainHeight(p: vec2f) -> f32 {
  let r = length(p);
  // The meadow drops away far out so the sky dips between the hills.
  var h = -0.28 * max(r - 78.0, 0.0);
  // Skyline fitted numerically against the reference photo.
  h += 3.64 * bump(p, vec2f(-9.3, 40.0), vec2f(4.2, 11.4));
  h += 2.5 * bump(p, vec2f(-4.0, 47.0), vec2f(4.9, 5.2));
  h += 1.98 * bump(p, vec2f(-15.8, 43.7), vec2f(6.7, 8.7));
  h += 2.6 * bump(p, vec2f(-29.8, 52.0), vec2f(10.9, 8.2));
  h += 4.99 * bump(p, vec2f(10.2, 54.0), vec2f(9.3, 6.9));
  h += 2.19 * bump(p, vec2f(11.0, 40.0), vec2f(8.2, 4.0));
  h += 12.18 * bump(p, vec2f(-2.0, 125.9), vec2f(15.5, 10.7));
  h += 4.5 * bump(p, vec2f(-34.0, 60.0), vec2f(15.0, 13.0));
  h += 4.8 * bump(p, vec2f(34.0, 64.0), vec2f(15.0, 14.0));
  h += 3.0 * bump(p, vec2f(-52.0, 45.0), vec2f(14.0, 12.0));
  h += 3.4 * bump(p, vec2f(52.0, 48.0), vec2f(14.0, 12.0));
  // Gentle rolling of the plain, flat where the blade grid lives.
  h += 0.3 * fbm(p * 0.045 + vec2f(3.1, 7.7)) * smoothstep(params.grass.x, params.grass.x + 14.0, length(p - params.door.xy));
  return h;
}

fn terrainNormal(p: vec3f, eps: f32) -> vec3f {
  let e = vec2f(eps, 0.0);
  let hx = terrainHeight(p.xz + e.xy) - terrainHeight(p.xz - e.xy);
  let hz = terrainHeight(p.xz + e.yx) - terrainHeight(p.xz - e.yx);
  return normalize(vec3f(-hx, 2.0 * eps, -hz));
}

fn marchTerrain(ro: vec3f, rd: vec3f, tStart: f32, tmax: f32) -> f32 {
  var t = max(tStart, 0.02);
  var lastT = 0.0;
  var hit = -1.0;
  for (var i = 0; i < 220; i++) {
    let p = ro + rd * t;
    let d = p.y - terrainHeight(p.xz);
    if (d < 0.0) {
      hit = t;
      break;
    }
    if (t > tmax || (p.y > 15.0 && rd.y > 0.0)) { break; }
    lastT = t;
    t += clamp(d * 0.45, 0.05 + 0.012 * t, 8.0);
  }
  if (hit < 0.0) { return -1.0; }
  var a = lastT;
  var b = hit;
  for (var i = 0; i < 7; i++) {
    let m = 0.5 * (a + b);
    let p = ro + rd * m;
    if (p.y - terrainHeight(p.xz) < 0.0) { b = m; } else { a = m; }
  }
  return 0.5 * (a + b);
}

fn terrainShadow(p: vec3f, l: vec3f) -> f32 {
  var s = 1.0;
  var t = 0.6;
  for (var i = 0; i < 24; i++) {
    let q = p + l * t;
    let d = q.y - terrainHeight(q.xz);
    s = min(s, clamp(3.0 * d / t, 0.0, 1.0));
    if (s < 0.01 || q.y > 16.0) { break; }
    t += clamp(d, 0.3, 4.0) * 1.2;
  }
  return s;
}

// ---------------------------------------------------------------- door

fn sdBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdBox2(p: vec2f, b: vec2f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

struct DoorFrame {
  right: vec3f,
  fwd: vec3f,
  origin: vec3f,
}

fn doorFrame() -> DoorFrame {
  let yaw = params.door.z;
  var f: DoorFrame;
  f.right = vec3f(cos(yaw), 0.0, sin(yaw));
  f.fwd = vec3f(-sin(yaw), 0.0, cos(yaw));
  let x = params.door.x;
  let z = params.door.y;
  f.origin = vec3f(x, terrainHeight(vec2f(x, z)) - 0.02, z);
  return f;
}

fn toDoor(p: vec3f, f: DoorFrame) -> vec3f {
  let d = p - f.origin;
  return vec3f(dot(d, f.right), d.y, dot(d, f.fwd));
}

// Recessed panels on both faces of the leaf. q is leaf-local: x from the hinge (0) to the
// free edge (DOOR_W), y up, z across the thickness. Returns a solid to subtract.
fn leafPanels(q: vec3f) -> f32 {
  let cx = DOOR_W * 0.5;
  let up = vec2f(q.x - cx, q.y - 1.34);
  var panel = sdBox2(up, vec2f(0.10, 0.34)) - 0.17;
  let lx = abs(q.x - cx) - 0.16;
  let low = vec2f(lx, q.y - 0.40);
  let panel2 = sdBox2(low, vec2f(0.055, 0.19)) - 0.03;
  panel = min(panel, panel2);
  let depth = (LEAF_D - 0.005) - abs(q.z);
  return max(panel, depth);
}

fn sdDoor(p: vec3f, f: DoorFrame) -> f32 {
  let q = toDoor(p, f);
  let hw = DOOR_W * 0.5;
  let outer = sdBox(q - vec3f(0.0, (DOOR_H + FRAME_T) * 0.5 - 0.05, 0.0),
                    vec3f(hw + FRAME_T, (DOOR_H + FRAME_T) * 0.5 + 0.05, FRAME_D)) - 0.004;
  let opening = sdBox(q - vec3f(0.0, DOOR_H * 0.5 - 0.1, 0.0), vec3f(hw, DOOR_H * 0.5 + 0.1, 1.0));
  var d = max(outer, -opening);
  // Leaf hinged on the right jamb, swung toward the viewer by params.door.w.
  let a = params.door.w;
  let u = vec3f(-cos(a), 0.0, -sin(a));
  let n = vec3f(-u.z, 0.0, u.x);
  let hinge = vec3f(hw, 0.0, 0.0);
  let r = q - hinge;
  let l = vec3f(dot(r, u), r.y, dot(r, n));
  let leafBox = vec3f(l.x - hw, l.y - DOOR_H * 0.5 - 0.005, l.z);
  var leaf = sdBox(leafBox, vec3f(hw - 0.004, DOOR_H * 0.5 - 0.012, LEAF_D)) - 0.003;
  leaf = max(leaf, -leafPanels(l));
  let knob = length(l - vec3f(DOOR_W - 0.09, 1.02, LEAF_D + 0.03)) - 0.032;
  let knob2 = length(l - vec3f(DOOR_W - 0.09, 1.02, -LEAF_D - 0.03)) - 0.032;
  leaf = min(leaf, min(knob, knob2));
  d = min(d, leaf);
  return d;
}

fn raySphere(ro: vec3f, rd: vec3f, c: vec3f, r: f32) -> vec2f {
  let oc = ro - c;
  let b = dot(oc, rd);
  let cc = dot(oc, oc) - r * r;
  let h = b * b - cc;
  if (h < 0.0) { return vec2f(-1.0, -1.0); }
  let s = sqrt(h);
  return vec2f(-b - s, -b + s);
}

fn marchDoor(ro: vec3f, rd: vec3f, tmax: f32, f: DoorFrame) -> f32 {
  let sd = raySphere(ro, rd, f.origin + vec3f(0.0, 1.1, 0.0), 1.9);
  if (sd.y < 0.0 || sd.x > tmax) { return -1.0; }
  var t = max(sd.x, 0.0);
  for (var i = 0; i < 96; i++) {
    let d = sdDoor(ro + rd * t, f);
    if (d < 0.0005 * max(t, 1.0)) { return t; }
    t += d * 0.85;
    if (t > min(sd.y, tmax)) { break; }
  }
  return -1.0;
}

fn doorNormal(p: vec3f, f: DoorFrame) -> vec3f {
  let e = 0.001;
  let k = vec2f(1.0, -1.0);
  return normalize(
    k.xyy * sdDoor(p + k.xyy * e, f) +
    k.yyx * sdDoor(p + k.yyx * e, f) +
    k.yxy * sdDoor(p + k.yxy * e, f) +
    k.xxx * sdDoor(p + k.xxx * e, f));
}

// Soft shadow from the door. Lower k = wider penumbra.
fn doorShadow(p: vec3f, l: vec3f, f: DoorFrame, maxT: f32, k: f32) -> f32 {
  let sd = raySphere(p, l, f.origin + vec3f(0.0, 1.1, 0.0), 1.9);
  if (sd.y < 0.0) { return 1.0; }
  var s = 1.0;
  var t = max(sd.x, 0.02);
  for (var i = 0; i < 40; i++) {
    let d = sdDoor(p + l * t, f);
    s = min(s, clamp(k * d / t, 0.0, 1.0));
    if (s < 0.005) { break; }
    t += clamp(d, 0.01, 0.4);
    if (t > min(sd.y, maxT)) { break; }
  }
  return s;
}

// ---------------------------------------------------------------- grass fur volume

// The grass is a fur-like volume (Kajiya-Kay style 3D texture): a dense field of thin strands
// described by a density function inside a slab above the ground, ray marched with
// front-to-back compositing. Combing shifts the strand pattern with height so strands lie
// along the comb direction instead of standing up.
const FUR_COMB: f32 = 1.7;      // horizontal strand drift per metre of height (lying down)
const FUR_SIGMA: f32 = 150.0;   // extinction inside a strand-dense region (1/m)

// Local fur coverage: clumpy, fading toward the edge of the fur patch around the door.
fn grassCoverage(xz: vec2f) -> f32 {
  let clump = clamp(0.45 + 1.1 * (fbm3(xz * 0.45 + vec2f(9.0, 4.0)) - 0.3), 0.3, 1.25);
  let radius = params.grass.x;
  return clump * (1.0 - smoothstep(radius * 0.55, radius * 0.98, length(xz - params.door.xy)));
}

// Comb direction of the flattened grass: a prevailing direction with gentle waves.
fn combDir(xz: vec2f) -> vec2f {
  let a = PI * 0.78 + 0.7 * (fbm3(xz * 0.07 + vec2f(3.0, 1.0)) - 0.5) + 0.2 * sin(xz.x * 0.9 + xz.y * 0.4);
  return vec2f(sin(a), cos(a));
}

struct FurSample {
  density: f32,
  strand: f32,   // strand pattern value, for per-strand colour variation
}

fn furDensity(p: vec3f, dir: vec2f, cover: f32, fineMix: f32) -> FurSample {
  var s: FurSample;
  let h = max(params.grass.y, 0.01);
  let hf = clamp(p.y / h, 0.0, 1.0);
  // Combing: the pattern drifts along the comb direction as we go up, so a strand column
  // becomes a strand lying along `dir`.
  let q = p.xz - dir * (p.y * FUR_COMB);
  // Strand pattern stretched along the comb direction: brushed streaks, not speckle. The fine
  // octave fades out where a pixel is wider than a strand, which keeps far rows from aliasing.
  let perp = vec2f(-dir.y, dir.x);
  let u = vec2f(dot(q, dir) * 0.45, dot(q, perp));
  let coarse = vnoise(u * 90.0);
  let fine = vnoise(u * 190.0 + vec2f(7.3, 2.1));
  let strand = mix(coarse, coarse * 0.65 + fine * 0.35, fineMix);
  let lenVar = 0.55 + 0.45 * vnoise(q * 14.0 + vec2f(3.0, 5.0));
  let d = strand * lenVar * cover - hf * 0.85;
  s.density = clamp(d * 3.5, 0.0, 1.0);
  s.strand = strand;
  return s;
}

// Light transmittance down into the fur toward a light: strands are densest near the ground.
fn furTransmittance(hf: f32, l: vec3f, cover: f32) -> f32 {
  let depth = pow(1.0 - hf, 1.5);
  return exp(-1.4 * cover * depth / max(l.y, 0.12));
}

struct FurResult {
  color: vec3f,
  alpha: f32,
}

fn furAlbedo(hf: f32, strand: f32) -> vec3f {
  let base = rgb8(46.0, 82.0, 26.0);
  let tip = rgb8(150.0, 176.0, 66.0);
  var albedo = mix(base, tip, smoothstep(0.0, 1.0, hf));
  albedo *= 0.8 + 0.4 * strand;
  return albedo;
}

// March the fur slab between tEnter and tExit along the ray and composite front to back.
fn furMarch(ro: vec3f, rd: vec3f, tEnter: f32, tExit: f32, f: DoorFrame, dayMix: f32, rim: f32, pixelAngle: f32) -> FurResult {
  var res: FurResult;
  res.color = vec3f(0.0);
  res.alpha = 0.0;
  if (tExit <= tEnter) { return res; }
  let steps = select(40, 88, params.grass.z > 0.5);
  let dt = (tExit - tEnter) / f32(steps);
  // Per-ray start jitter turns step banding into fine noise that supersampling averages out.
  let jitter = hash13(rd * 977.0) * dt;
  let footprint = pixelAngle * tExit;
  let fineMix = 1.0 - smoothstep(0.003, 0.009, footprint);
  let h = max(params.grass.y, 0.01);
  // Per-ray constants: the comb field and coverage vary slowly, so sample them once at the
  // ground point; shadows from the door are evaluated once as well.
  let pg = ro + rd * tExit;
  let cover = grassCoverage(pg.xz);
  if (cover < 0.01) { return res; }
  let dir = combDir(pg.xz);
  let tangent = normalize(vec3f(dir.x * FUR_COMB, 1.0, dir.y * FUR_COMB));
  let sun = sunDir();
  let sunCol = sunColor();
  let v = -rd;
  let sunSh = doorShadow(pg + vec3f(0.0, 0.05, 0.0), sun, f, 8.0, 5.0) * terrainShadow(pg, sun);
  let doorCenter = f.origin + vec3f(0.0, DOOR_H * 0.5, 0.0);
  let toDoorC = doorCenter - pg;
  let doorSh = doorShadow(pg + vec3f(0.0, 0.04, 0.0), normalize(toDoorC), f, length(toDoorC) - 0.3, 12.0);
  // Kajiya-Kay terms depend only on directions, so they are per ray.
  let tlSun = dot(tangent, sun);
  let kkSun = sqrt(max(1.0 - tlSun * tlSun, 0.0));
  let hSun = normalize(sun + v);
  let thSun = dot(tangent, hSun);
  let specSun = pow(sqrt(max(1.0 - thSun * thSun, 0.0)), 18.0) * 0.12;
  let moon = moonDir();
  let tlMoon = dot(tangent, moon);
  let kkMoon = sqrt(max(1.0 - tlMoon * tlMoon, 0.0));

  var T = 1.0;
  var col = vec3f(0.0);
  for (var i = 0; i < steps; i++) {
    let t = tEnter + f32(i) * dt + jitter;
    if (t > tExit) { break; }
    let p = ro + rd * t;
    let fs = furDensity(p, dir, cover, fineMix);
    if (fs.density < 0.002) { continue; }
    let hf = clamp(p.y / h, 0.0, 1.0);
    let albedo = furAlbedo(hf, fs.strand);
    let ao = 0.25 + 0.75 * hf;
    // Day: sun through the fur plus sky ambient, occluded with depth.
    var day = albedo * (sunCol * kkSun * furTransmittance(hf, sun, cover) * sunSh + AMBIENT * ao);
    day += sunCol * specSun * furTransmittance(hf, sun, cover) * sunSh * albedo * 2.0;
    // Night: moon fill and the door light.
    let lumA = dot(albedo, vec3f(0.2126, 0.7152, 0.0722));
    let nightAlbedo = mix(albedo, lumA * vec3f(0.65, 0.85, 1.0), 0.4);
    var night = nightAlbedo * (vec3f(0.28, 0.4, 0.68) * 0.16 * kkMoon * furTransmittance(hf, moon, cover) + vec3f(0.006, 0.01, 0.026) * ao);
    if (dayMix < 0.999) {
      var e = 0.0;
      var spec = 0.0;
      for (var j = 0; j < 3; j++) {
        let o = DOOR_SAMPLES[j];
        let s = f.origin + f.right * o.x + vec3f(0.0, o.y, 0.0);
        let toL = s - p;
        let d = max(length(toL), 0.05);
        let l = toL / d;
        let facing = max(dot(f.fwd, l), 0.0);
        let geom = facing / (d * d + 0.6);
        let tl = dot(tangent, l);
        let kk = sqrt(max(1.0 - tl * tl, 0.0));
        let tr = furTransmittance(hf, l, cover);
        e += geom * (0.35 + 0.65 * kk) * tr;
        let hl = normalize(l + v);
        let th = dot(tangent, hl);
        spec += geom * pow(sqrt(max(1.0 - th * th, 0.0)), 14.0) * tr;
      }
      let front = dot(p - f.origin, f.fwd);
      let inFront = select(0.0, 1.0, front < -0.01);
      night += albedo * DOOR_COLOR * params.look.w * (e * 0.333 + spec * 0.08) * doorSh * inFront;
    }
    var sampleCol = mix(night, day, dayMix) + albedo * rimGlow(vec3f(0.0, 1.0, 0.0), rim);
    let a = 1.0 - exp(-fs.density * FUR_SIGMA * dt);
    col += T * a * sampleCol;
    T *= 1.0 - a;
    if (T < 0.02) { break; }
  }
  res.color = col;
  res.alpha = 1.0 - T;
  return res;
}

// Ray/slab interval for the fur layer (y in [0, height]) restricted to the fur patch radius.
fn furInterval(ro: vec3f, rd: vec3f, tMax: f32) -> vec2f {
  let hMax = params.grass.y;
  let radius = params.grass.x;
  var t0 = 0.0;
  var t1 = tMax;
  if (abs(rd.y) > 1e-6) {
    let ta = (0.0 - ro.y) / rd.y;
    let tb = (hMax - ro.y) / rd.y;
    t0 = max(t0, min(ta, tb));
    t1 = min(t1, max(ta, tb));
  } else if (ro.y < 0.0 || ro.y > hMax) {
    return vec2f(1.0, 0.0);
  }
  let a2 = dot(rd.xz, rd.xz);
  if (a2 > 1e-8) {
    let oc = ro.xz - params.door.xy;
    let b2 = dot(oc, rd.xz);
    let c2 = dot(oc, oc) - radius * radius;
    let disc = b2 * b2 - a2 * c2;
    if (disc < 0.0) { return vec2f(1.0, 0.0); }
    let sq = sqrt(disc);
    t0 = max(t0, (-b2 - sq) / a2);
    t1 = min(t1, (-b2 + sq) / a2);
  }
  return vec2f(t0, t1);
}

// ---------------------------------------------------------------- portal + sand world

fn portalHit(ro: vec3f, rd: vec3f, f: DoorFrame) -> f32 {
  let denom = dot(rd, f.fwd);
  if (abs(denom) < 1e-5) { return -1.0; }
  let t = dot(f.origin - ro, f.fwd) / denom;
  if (t < 0.0) { return -1.0; }
  let q = toDoor(ro + rd * t, f);
  if (abs(q.x) < DOOR_W * 0.5 && q.y > 0.0 && q.y < DOOR_H) { return t; }
  return -1.0;
}

// Dune heightfield in door-local space (z grows away from the door). A slip face rises
// right behind the threshold so the opening is nothing but sand.
fn duneHeight(p: vec2f) -> f32 {
  let crestWobble = 0.9 * sin(p.x * 0.55 + 1.0) + 1.4 * (fbm3(p * 0.09 + vec2f(2.0, 5.0)) - 0.5);
  let start = 1.2 + 0.06 * sin(p.x * 1.3) + crestWobble * 0.3 + 0.15 * (fbm3(p * 0.8 + vec2f(5.0, 5.0)) - 0.5);
  let rise = p.y - start;
  // Flat sand at the threshold, a soft toe, then a 31 degree slip face.
  let toe = 1.2;
  let face = max(rise, 0.0) + toe * log(1.0 + exp(-abs(rise) / toe));
  let face0 = toe * log(1.0 + exp(-start / toe));
  var h = 0.6 * (face - face0);
  let swell = 0.3 + 0.7 * smoothstep(0.0, 4.0, rise);
  h += 1.8 * (fbm3(p * 0.16 + vec2f(7.0, 1.0)) - 0.5) * swell;
  h += 0.4 * (fbm3(p * 0.42 + vec2f(1.0, 8.0)) - 0.5) * swell;
  h += 0.18 * (fbm3(p * 0.9 + vec2f(3.0, 9.0)) - 0.5) * smoothstep(0.0, 2.0, rise);
  return h;
}

// Wind ripples: asymmetric waves with wandering crests, about 10 cm apart.
fn ripple(p: vec2f) -> f32 {
  let dir = vec2f(0.22, 0.975);
  let warp = 2.6 * (fbm3(p * 0.7 + vec2f(1.0, 4.0)) - 0.5) + 0.9 * (fbm3(p * 2.3 + vec2f(6.0, 2.0)) - 0.5) + 0.3 * sin(p.x * 2.6);
  let u = dot(p, dir) * (2.0 * PI / 0.1) + warp * 3.5;
  let amp = 0.0045 * (0.55 + 0.45 * fbm3(p * 0.5 + vec2f(3.0, 7.0)));
  return amp * (sin(u) - 0.32 * sin(2.0 * u) + 0.12 * sin(3.0 * u));
}

fn duneNormal(p: vec2f) -> vec3f {
  let e = 0.05;
  let hx = duneHeight(p + vec2f(e, 0.0)) - duneHeight(p - vec2f(e, 0.0));
  let hz = duneHeight(p + vec2f(0.0, e)) - duneHeight(p - vec2f(0.0, e));
  let re = 0.004;
  let r0 = ripple(p);
  let rx = ripple(p + vec2f(re, 0.0)) - r0;
  let rz = ripple(p + vec2f(0.0, re)) - r0;
  return normalize(vec3f(-hx / (2.0 * e) - rx / re, 1.0, -hz / (2.0 * e) - rz / re));
}

fn marchDune(ro: vec3f, rd: vec3f) -> f32 {
  var t = 0.02;
  var lastT = 0.0;
  var hit = -1.0;
  for (var i = 0; i < 160; i++) {
    let p = ro + rd * t;
    let d = p.y - duneHeight(p.xz);
    if (d < 0.0) { hit = t; break; }
    if (t > 60.0) { break; }
    lastT = t;
    t += clamp(d * 0.5, 0.01 + 0.01 * t, 2.0);
  }
  if (hit < 0.0) { return -1.0; }
  var a = lastT;
  var b = hit;
  for (var i = 0; i < 6; i++) {
    let m = 0.5 * (a + b);
    let p = ro + rd * m;
    if (p.y - duneHeight(p.xz) < 0.0) { b = m; } else { a = m; }
  }
  return 0.5 * (a + b);
}

fn shadeSand(p: vec3f, rd: vec3f, footprint: f32) -> vec3f {
  // Golden desert sand under a low sun that rakes across the ripples.
  var albedo = rgb8(214.0, 142.0, 52.0);
  albedo *= 0.9 + 0.2 * fbm3(p.xz * 0.7 + vec2f(4.0, 2.0));
  let grainFade = 1.0 - smoothstep(0.002, 0.012, footprint);
  albedo *= 1.0 + 0.16 * (hash13(floor(p * 700.0)) - 0.5) * grainFade;
  let n0 = duneNormal(p.xz);
  let g = vec3f(hash13(p * 431.0), hash13(p * 517.0 + vec3f(3.0)), hash13(p * 619.0 + vec3f(7.0))) - 0.5;
  let n = normalize(n0 + g * 0.35 * grainFade);
  let sun = normalize(vec3f(-0.62, 0.22, -0.55));
  let sunColor = vec3f(1.3, 0.96, 0.56) * 1.2;
  let skyAmb = vec3f(0.34, 0.16, 0.07) * 0.7;
  let ndl = max(dot(n0, sun), 0.0);
  let r = ripple(p.xz) / 0.0045;
  let occl = 0.5 + 0.5 * smoothstep(-0.9, 0.9, r);
  var col = albedo * (sunColor * ndl * occl + skyAmb * (0.5 + 0.5 * n0.y) * occl);
  let v = -rd;
  let h = normalize(sun + v);
  col += albedo * pow(max(dot(n, h), 0.0), 10.0) * 0.12 * sunColor;
  let glint = pow(max(dot(n, h), 0.0), 140.0) * step(0.93, hash13(floor(p * 900.0))) * grainFade;
  col += sunColor * glint * 0.8;
  return col;
}

fn renderSand(ro: vec3f, rd: vec3f, footprintScale: f32) -> vec3f {
  let t = marchDune(ro, rd);
  if (t < 0.0) {
    return rgb8(236.0, 190.0, 120.0) * 0.8;
  }
  let p = ro + rd * t;
  var col = shadeSand(p, rd, footprintScale * t);
  col = mix(col, rgb8(236.0, 190.0, 120.0) * 0.8, 1.0 - exp(-t * 0.012));
  return col;
}

// ---------------------------------------------------------------- lighting

const AMBIENT: vec3f = vec3f(0.035, 0.09, 0.16);
const DOOR_COLOR: vec3f = vec3f(1.0, 0.685, 0.335);   // sRGB (255,214,156) in linear

fn sunDir() -> vec3f {
  let az = params.look.x;
  let el = params.look.y;
  return normalize(vec3f(sin(az) * cos(el), sin(el), cos(az) * cos(el)));
}

// Sun colour such that flat ground receives exactly (1,1,1) - ambient.
fn sunColor() -> vec3f {
  let el = params.look.y;
  return (vec3f(1.0) - AMBIENT) / max(sin(el), 0.2);
}

fn moonDir() -> vec3f {
  return normalize(vec3f(0.55, 0.62, -0.35));
}

// Sky measured from the reference: (0,86,160) at the horizon to (0,60,148) 22 degrees up.
fn skyDay(rd: vec3f) -> vec3f {
  let horizon = rgb8(0.0, 87.0, 161.0);
  let top = rgb8(0.0, 59.0, 147.0);
  let zenith = rgb8(0.0, 42.0, 128.0);
  let y = max(rd.y, 0.0);
  var c = mix(horizon, top, clamp(y / 0.37, 0.0, 1.0));
  c = mix(c, zenith, smoothstep(0.37, 0.95, y));
  return c;
}

// Night: an almost flat, deep blue.
fn skyNight(rd: vec3f) -> vec3f {
  let horizon = rgb8(13.0, 23.0, 56.0);
  let zenith = rgb8(9.0, 16.0, 44.0);
  return mix(horizon, zenith, smoothstep(0.0, 0.5, max(rd.y, 0.0)));
}

// Expanding day front, centred on the door. Returns day mix in [0,1] and the rim glow.
fn dayFront(p: vec3f, f: DoorFrame) -> vec2f {
  let ph = clamp(params.phase, 0.0, 1.0);
  let e = ph * ph * (3.0 - 2.0 * ph);
  let radius = e * 210.0 - 0.6;
  let d = length(p - (f.origin + vec3f(0.0, 1.0, 0.0)));
  let edge = 1.5 + radius * 0.08;
  let day = 1.0 - smoothstep(radius - edge, radius + edge, d);
  let rim = exp(-pow((d - radius) / (edge * 1.6), 2.0)) * (1.0 - smoothstep(0.85, 1.0, ph)) * smoothstep(0.0, 0.05, ph);
  return vec2f(day, rim);
}

const DOOR_SAMPLES: array<vec2f, 3> = array<vec2f, 3>(vec2f(-0.22, 0.42), vec2f(0.21, 1.12), vec2f(-0.07, 1.78));

// Light from the sunlit dune pouring through the opening: a warm rectangular area light
// sampled at three points, each with its own shadow ray, so penumbras stay soft.
// `tangent` enables Kajiya-Kay fibre shading for blades (zero vector for surfaces).
fn doorLight(p: vec3f, n: vec3f, tangent: vec3f, transl: f32, f: DoorFrame, shadows: bool) -> vec3f {
  let front = dot(p - f.origin, f.fwd);
  if (front > -0.01) { return vec3f(0.0); }
  let fibre = dot(tangent, tangent) > 0.5;
  var sum = 0.0;
  for (var i = 0; i < 3; i++) {
    let o = DOOR_SAMPLES[i];
    let s = f.origin + f.right * o.x + vec3f(0.0, o.y, 0.0);
    let toL = s - p;
    let d = max(length(toL), 0.05);
    let l = toL / d;
    let facing = max(dot(f.fwd, l), 0.0);
    let geom = facing / (d * d + 0.6);
    let ndl = dot(n, l);
    var diffuse = max(ndl, 0.0) + transl * max(-ndl, 0.0) + 0.12 * (1.0 - abs(ndl));
    if (fibre) {
      // Kajiya-Kay: a thin fibre scatters according to the angle to its axis.
      let tl = dot(tangent, l);
      diffuse = mix(diffuse, sqrt(max(1.0 - tl * tl, 0.0)), 0.5);
    }
    var vis = 1.0;
    if (shadows) {
      vis = doorShadow(p + n * 0.002, l, f, d - 0.3, 12.0);
    }
    sum += geom * diffuse * vis;
  }
  return DOOR_COLOR * params.look.w * sum / 3.0;
}

// Single scattering of the door light in the night air along the primary ray: the haze
// that sits around the opening in the reference photos.
fn doorScatter(ro: vec3f, rd: vec3f, tEnd: f32, f: DoorFrame) -> vec3f {
  let steps = 10;
  let tFar = min(tEnd, 40.0);
  let dt = tFar / f32(steps);
  var acc = 0.0;
  for (var i = 0; i < steps; i++) {
    let tt = (f32(i) + 0.5) * dt;
    let q = ro + rd * tt;
    if (dot(q - f.origin, f.fwd) > -0.01) { continue; }
    var e = 0.0;
    for (var j = 0; j < 3; j++) {
      let o = DOOR_SAMPLES[j];
      let s = f.origin + f.right * o.x + vec3f(0.0, o.y, 0.0);
      let toL = s - q;
      let d = max(length(toL), 0.3);
      let l = toL / d;
      let facing = max(dot(f.fwd, l), 0.0);
      // Henyey-Greenstein forward lobe: brightest when looking toward the door.
      let g = 0.45;
      let c = dot(l, rd);
      let phase = (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * c, 1.5));
      e += facing / (d * d + 0.6) * phase;
    }
    acc += e / 3.0 * dt;
  }
  return DOOR_COLOR * params.look.w * acc * 0.005;
}

struct Grass {
  albedoMod: f32,
  warm: f32,
}

// Far-field grass texture (beyond the blade grid). footprint = world size of one pixel.
fn grassTexture(p: vec3f, footprint: f32) -> Grass {
  let strength = params.look.z;
  var g: Grass;
  let sq = vec2f(1.0, 0.45);
  let fBlade = 70.0;
  let bladeFade = 1.0 - smoothstep(0.3, 1.3, footprint * fBlade);
  var blades = 0.0;
  if (bladeFade > 0.001) {
    let n1 = vnoise(vec2f(p.x * fBlade, p.z * fBlade * 0.12));
    let n2 = vnoise(vec2f(p.x * fBlade * 2.1 + 13.1, p.z * fBlade * 0.3 + 7.3));
    blades = ((n1 - 0.5) * 1.0 + (n2 - 0.5) * 0.6) * bladeFade;
  }
  let fClump = 6.0;
  let clumpFade = 1.0 - smoothstep(0.3, 1.2, footprint * fClump);
  var clumps = 0.0;
  if (clumpFade > 0.001) {
    clumps = (fbm3(p.xz * sq * fClump) - 0.5) * clumpFade;
  }
  let fTuft = 4.5;
  let tuftFade = 1.0 - smoothstep(0.35, 1.3, footprint * fTuft);
  var tufts = 0.0;
  if (tuftFade > 0.001) {
    tufts = (fbm3(p.xz * fTuft + vec2f(5.0, 2.0)) - 0.5) * tuftFade;
  }
  let patches = fbm3(p.xz * 0.35 + vec2f(11.0, 3.0)) - 0.5;
  // Brushed streaks along the prevailing comb direction, gently waved by a warp so they read
  // as flattened grass rather than wood grain.
  let a0 = PI * 0.78;
  let dir = vec2f(sin(a0), cos(a0));
  let warp = 0.25 * (fbm3(p.xz * 0.22 + vec2f(2.0, 6.0)) - 0.5);
  let along = dot(p.xz, dir);
  let across = dot(p.xz, vec2f(-dir.y, dir.x)) + warp;
  let fStreak = 24.0;
  let streakFade = 1.0 - smoothstep(0.3, 1.3, footprint * fStreak);
  var streaks = 0.0;
  if (streakFade > 0.001) {
    streaks = (vnoise(vec2f(along * fStreak * 0.28, across * fStreak)) - 0.5) * 0.9
            + (vnoise(vec2f(along * fStreak * 0.5 + 7.0, across * fStreak * 2.1)) - 0.5) * 0.5;
    streaks *= streakFade;
  }
  g.albedoMod = 1.0 + strength * (blades * 0.5 + streaks * 0.4 + clumps * 0.3 + tufts * 0.24 + patches * 0.18);
  g.warm = max(blades, 0.0) * 0.7 + smoothstep(0.6, 0.9, vnoise(p.xz * 45.0)) * bladeFade * 0.4;
  return g;
}

fn nightBase(albedo: vec3f, n: vec3f) -> vec3f {
  // Scotopic look: the eye loses colour at night, so drift toward a cool grey-green.
  let lumA = dot(albedo, vec3f(0.2126, 0.7152, 0.0722));
  let nightAlbedo = mix(albedo, lumA * vec3f(0.65, 0.85, 1.0), 0.4);
  let moon = moonDir();
  let moonColor = vec3f(0.28, 0.4, 0.68) * 0.16;
  let nightAmbient = vec3f(0.006, 0.01, 0.026);
  let hemi = 0.5 + 0.5 * n.y;
  return nightAlbedo * (moonColor * max(dot(n, moon), 0.0) + nightAmbient * hemi);
}

fn rimGlow(n: vec3f, rim: f32) -> vec3f {
  return vec3f(1.0, 0.7, 0.45) * rim * 4.0 * (0.4 + 0.6 * max(dot(n, normalize(vec3f(0.0, 0.6, 1.0))), 0.0));
}

fn shadeGround(p: vec3f, n0: vec3f, rd: vec3f, t: f32, footprint: f32, f: DoorFrame, dayMix: f32, rim: f32) -> vec3f {
  let tex = grassTexture(p, footprint);
  let cover = grassCoverage(p.xz);
  // Reference lawn colour; under the blade layer the ground sits at the bottom of the
  // grass volume, so it is darker and sees less sky.
  var albedo = rgb8(90.0, 132.0, 47.0) * tex.albedoMod;
  albedo = mix(albedo, albedo * vec3f(1.25, 1.08, 0.7), clamp(tex.warm, 0.0, 1.0) * 0.5);
  let ao = mix(1.0, 0.7, clamp(cover, 0.0, 1.0));
  albedo *= mix(1.0, 0.92, clamp(cover, 0.0, 1.0));
  let slope = 1.0 - n0.y;
  albedo *= mix(vec3f(1.0), vec3f(0.62, 0.74, 0.7), smoothstep(0.02, 0.22, slope));
  let n = n0;
  let sun = sunDir();
  let v = -rd;

  // --- Day: sun + blue sky ambient tuned so flat ground = albedo.
  let sunCol = sunColor();
  let ndl = max(dot(n, sun), 0.0);
  var sh = terrainShadow(p, sun) * doorShadow(p, sun, f, 8.0, 5.0);
  sh *= furTransmittance(0.0, sun, cover);
  let hemi = 0.5 + 0.5 * n.y;
  var day = albedo * (sunCol * ndl * sh + AMBIENT * hemi * ao);
  let h = normalize(sun + v);
  day += albedo * pow(max(dot(n, h), 0.0), 5.0) * 0.10 * sh * sunCol;

  // --- Night: moon fill + the door spill.
  var night = nightBase(albedo, n) * ao;
  if (dayMix < 0.999) {
    night += mix(albedo, vec3f(dot(albedo, vec3f(0.33))), 0.2) * doorLight(p, n, vec3f(0.0), 0.0, f, true) * furTransmittance(0.0, normalize(f.origin + vec3f(0.0, 1.0, 0.0) - p), cover);
  }
  return mix(night, day, dayMix) + albedo * rimGlow(n, rim);
}

fn shadeDoor(p: vec3f, n: vec3f, rd: vec3f, f: DoorFrame, dayMix: f32, rim: f32) -> vec3f {
  let albedo = rgb8(86.0, 94.0, 120.0);        // slate blue paint
  let sun = sunDir();
  let v = -rd;
  let sunCol = sunColor() * 0.784;
  let ndl = max(dot(n, sun), 0.0);
  let sh = doorShadow(p + n * 0.003, sun, f, 6.0, 5.0) * terrainShadow(p, sun);
  let hemi = 0.5 + 0.5 * n.y;
  var day = albedo * (sunCol * ndl * sh * 0.85 + AMBIENT * hemi * 1.4);
  let h = normalize(sun + v);
  day += pow(max(dot(n, h), 0.0), 40.0) * 0.25 * sh * sunCol;

  var night = nightBase(albedo, n) * 1.2;
  if (dayMix < 0.999) {
    let spill = doorLight(p, n, vec3f(0.0), 0.0, f, true) * 0.55;
    night += albedo * spill;
    let doorCenter = f.origin + vec3f(0.0, DOOR_H * 0.5, 0.0);
    let l = normalize(doorCenter - p);
    let hn = normalize(l + v);
    night += pow(max(dot(n, hn), 0.0), 30.0) * 0.05 * spill;
  }
  return mix(night, day, dayMix) + albedo * rimGlow(n, rim);
}

// ---------------------------------------------------------------- render

fn render(ro: vec3f, rd: vec3f, pixelAngle: f32) -> vec3f {
  let f = doorFrame();
  let tPortal = portalHit(ro, rd, f);
  let tTerrain = marchTerrain(ro, rd, 0.0, TMAX);
  var tLimit = select(tTerrain, TMAX, tTerrain < 0.0);
  let tDoor = marchDoor(ro, rd, tLimit, f);

  var t = tTerrain;
  var kind = 0;   // 0 sky, 1 terrain, 2 door
  if (t > 0.0) { kind = 1; }
  if (tDoor > 0.0 && (t < 0.0 || tDoor < t)) { t = tDoor; kind = 2; }
  let throughDoor = tPortal > 0.0 && (t < 0.0 || tPortal < t);

  var color = vec3f(0.0);
  if (throughDoor) {
    // The opening is a window onto the dune: continue the ray in door-local space.
    let q = toDoor(ro + rd * tPortal, f);
    let ld = vec3f(dot(rd, f.right), rd.y, dot(rd, f.fwd));
    return renderSand(vec3f(q.x, q.y, 0.0), normalize(ld), pixelAngle);
  }
  var dayMix = 1.0;
  var rim = 0.0;
  if (kind == 0) {
    let far = ro + rd * 120.0;
    let front = dayFront(far, f);
    dayMix = front.x;
    color = mix(skyNight(rd), skyDay(rd), dayMix);
    color += vec3f(1.0, 0.6, 0.35) * front.y * 0.2;
    t = 60.0;
  } else {
    let p = ro + rd * t;
    let front = dayFront(p, f);
    dayMix = front.x;
    rim = front.y;
    let footprint = pixelAngle * t;
    if (kind == 2) {
      let n = doorNormal(p, f);
      color = shadeDoor(p, n, rd, f, dayMix, rim);
    } else {
      let n = terrainNormal(p, max(0.08, footprint * 0.5));
      color = shadeGround(p, n, rd, t, footprint, f, dayMix, rim);
    }
    // The grass fur volume sits on the ground in front of whatever was hit.
    let iv = furInterval(ro, rd, t);
    if (iv.y > iv.x) {
      let fur = furMarch(ro, rd, iv.x, iv.y, f, dayMix, rim, pixelAngle);
      color = fur.color + color * (1.0 - fur.alpha);
    }
    // Aerial perspective: night haze is heavier than the crisp day.
    let fogNight = skyNight(vec3f(rd.x, 0.02, rd.z)) * 0.9;
    let fogDay = skyDay(vec3f(rd.x, 0.03, rd.z));
    let fog = mix(fogNight, fogDay, dayMix);
    let fogAmt = 1.0 - exp(-t * 0.0028);
    color = mix(color, fog, fogAmt * mix(0.6, 0.05, dayMix));
  }
  if (dayMix < 0.999) {
    color += doorScatter(ro, rd, t, f) * (1.0 - dayMix);
  }
  return color;
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let res = params.resolution;
  let aspect = res.x / max(res.y, 1.0);
  let fovY = params.camera.z;
  let focal = 1.0 / tan(fovY * 0.5);
  let pitch = params.camera.y;
  let ro = vec3f(0.0, params.camera.x, 0.0);
  let cp = cos(pitch);
  let sp = sin(pitch);
  let forward = vec3f(0.0, sp, cp);
  let up = vec3f(0.0, cp, -sp);
  let right = vec3f(1.0, 0.0, 0.0);
  let pixelAngle = (2.0 / focal) / res.y;

  let samples = u32(max(params.camera.w, 1.0));
  var acc = vec3f(0.0);
  let n = select(1u, 2u, samples >= 4u);
  let count = n * n;
  for (var i = 0u; i < count; i++) {
    var offset = vec2f(0.5);
    if (n == 2u) {
      offset = vec2f(0.25 + 0.5 * f32(i % 2u), 0.25 + 0.5 * f32(i / 2u));
    }
    let px = uv * res + offset - vec2f(0.5);
    let ndc = (px / res) * 2.0 - 1.0;
    let sx = ndc.x * aspect;
    let sy = -ndc.y;
    let rd = normalize(forward * focal + right * sx + up * sy);
    acc += render(ro, rd, pixelAngle);
  }
  return vec4f(acc / f32(count), 1.0);
}
