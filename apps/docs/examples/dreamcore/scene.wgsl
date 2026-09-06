// Dreamcore: a Bliss-like meadow at night with a door standing in the grass. Through the
// opening there is a sunlit sand dune; as `phase` rises the day pours out of the door and
// sweeps across the hills. Fullscreen raymarcher: heightfield hills, grid-traced grass
// blades, SDF door and chair, a rectangular area light for the door spill, and a second
// heightfield world (the dune) behind the portal. Linear HDR out; post does the rest.

struct Params {
  resolution: vec2f,
  time: f32,
  phase: f32,        // 0 = night, 1 = day (the day expands out of the door)
  camera: vec4f,     // height, pitch (rad, + looks up), vertical fov (rad), aa samples
  door: vec4f,       // x, z, yaw (rad), leaf angle (rad)
  chair: vec4f,      // x, z, yaw (rad), scale
  look: vec4f,       // sun azimuth (rad), sun elevation (rad), texture strength, door light
  grass: vec4f,      // blade radius (m), max blade height (m), shadow quality (0/1), unused
}

@group(0) @binding(0) var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const DOOR_W: f32 = 0.92;   // opening width
const DOOR_H: f32 = 2.08;   // opening height
const FRAME_T: f32 = 0.085; // frame member width
const FRAME_D: f32 = 0.07;  // frame half depth
const LEAF_D: f32 = 0.022;  // leaf half thickness
const TMAX: f32 = 170.0;
const CELL: f32 = 0.1;      // grass grid cell (m); every cell grows three blades

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

fn rot2(a: f32) -> mat2x2f {
  let c = cos(a);
  let s = sin(a);
  return mat2x2f(c, s, -s, c);
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
  // Skyline fitted against the reference photo.
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
  // Gentle rolling of the plain, flat around the camera where the blade grid lives.
  h += 0.3 * fbm(p * 0.045 + vec2f(3.1, 7.7)) * smoothstep(12.0, 26.0, r);
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

// ---------------------------------------------------------------- objects

fn sdBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdBox2(p: vec2f, b: vec2f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
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

// White monobloc garden chair, local space: origin at ground centre, facing -z.
fn sdChair(p0: vec3f) -> f32 {
  let s = params.chair.w;
  let p = p0 / s;
  var d = sdBox(p - vec3f(0.0, 0.43, 0.0), vec3f(0.235, 0.018, 0.22)) - 0.012;
  let pivot = vec3f(0.0, 0.44, 0.2);
  var b = p - pivot;
  let ca = cos(0.23);
  let sa = sin(0.23);
  b = vec3f(b.x, ca * b.y - sa * b.z, sa * b.y + ca * b.z);
  let shell = sdBox2(vec2f(b.x, b.y - 0.24), vec2f(0.19, 0.19)) - 0.05;
  var back = max(shell, abs(b.z) - 0.012) - 0.006;
  for (var i = 0; i < 5; i++) {
    let sx = (f32(i) - 2.0) * 0.078;
    let slot = sdBox2(vec2f(b.x - sx, b.y - 0.25), vec2f(0.016, 0.15));
    back = max(back, -slot);
  }
  d = min(d, back);
  let ax = abs(p.x);
  let arm = sdBox(vec3f(ax - 0.24, p.y - 0.63, p.z + 0.02), vec3f(0.028, 0.012, 0.19)) - 0.008;
  let armFront = sdBox(vec3f(ax - 0.24, p.y - 0.53, p.z + 0.2), vec3f(0.018, 0.1, 0.014)) - 0.006;
  d = min(d, min(arm, armFront));
  let sx = select(-1.0, 1.0, p.x > 0.0);
  let legF = sdCapsule(p, vec3f(sx * 0.2, 0.0, -0.17), vec3f(sx * 0.225, 0.42, -0.2), 0.02);
  let legB = sdCapsule(p, vec3f(sx * 0.27, 0.0, 0.26), vec3f(sx * 0.225, 0.42, 0.19), 0.02);
  d = min(d, min(legF, legB));
  return d * s;
}

fn chairOrigin() -> vec3f {
  let x = params.chair.x;
  let z = params.chair.y;
  return vec3f(x, terrainHeight(vec2f(x, z)) - 0.01, z);
}

fn toChair(p: vec3f) -> vec3f {
  let d = p - chairOrigin();
  let xz = rot2(-params.chair.z) * d.xz;
  return vec3f(xz.x, d.y, xz.y);
}

fn sdObjects(p: vec3f, f: DoorFrame) -> vec2f {
  let dd = sdDoor(p, f);
  let dc = sdChair(toChair(p));
  if (dc < dd) { return vec2f(dc, 2.0); }
  return vec2f(dd, 1.0);
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

struct ObjHit { t: f32, mat: f32 }

fn marchObjects(ro: vec3f, rd: vec3f, tmax: f32, f: DoorFrame) -> ObjHit {
  var res: ObjHit;
  res.t = -1.0;
  res.mat = 0.0;
  let sd = raySphere(ro, rd, f.origin + vec3f(0.0, 1.1, 0.0), 1.9);
  let sc = raySphere(ro, rd, chairOrigin() + vec3f(0.0, 0.45, 0.0), 0.85 * params.chair.w);
  var tStart = 1e9;
  var tEnd = -1e9;
  if (sd.y > 0.0) { tStart = min(tStart, max(sd.x, 0.0)); tEnd = max(tEnd, sd.y); }
  if (sc.y > 0.0) { tStart = min(tStart, max(sc.x, 0.0)); tEnd = max(tEnd, sc.y); }
  if (tEnd < 0.0 || tStart > tmax) { return res; }
  var t = tStart;
  for (var i = 0; i < 96; i++) {
    let p = ro + rd * t;
    let dm = sdObjects(p, f);
    if (dm.x < 0.0005 * max(t, 1.0)) {
      res.t = t;
      res.mat = dm.y;
      return res;
    }
    t += dm.x * 0.85;
    if (t > min(tEnd, tmax)) { break; }
  }
  return res;
}

fn objectNormal(p: vec3f, f: DoorFrame) -> vec3f {
  let e = 0.001;
  let k = vec2f(1.0, -1.0);
  return normalize(
    k.xyy * sdObjects(p + k.xyy * e, f).x +
    k.yyx * sdObjects(p + k.yyx * e, f).x +
    k.yxy * sdObjects(p + k.yxy * e, f).x +
    k.xxx * sdObjects(p + k.xxx * e, f).x);
}

// Soft shadow from the door and chair. Lower k = wider penumbra.
fn objectShadow(p: vec3f, l: vec3f, f: DoorFrame, maxT: f32, k: f32) -> f32 {
  let sd = raySphere(p, l, f.origin + vec3f(0.0, 1.1, 0.0), 1.9);
  let sc = raySphere(p, l, chairOrigin() + vec3f(0.0, 0.45, 0.0), 0.85 * params.chair.w);
  var tStart = 1e9;
  var tEnd = -1e9;
  if (sd.y > 0.0) { tStart = min(tStart, max(sd.x, 0.02)); tEnd = max(tEnd, sd.y); }
  if (sc.y > 0.0) { tStart = min(tStart, max(sc.x, 0.02)); tEnd = max(tEnd, sc.y); }
  if (tEnd < 0.0) { return 1.0; }
  var s = 1.0;
  var t = tStart;
  for (var i = 0; i < 40; i++) {
    let d = sdObjects(p + l * t, f).x;
    s = min(s, clamp(k * d / t, 0.0, 1.0));
    if (s < 0.005) { break; }
    t += clamp(d, 0.01, 0.4);
    if (t > min(tEnd, maxT)) { break; }
  }
  return s;
}

// ---------------------------------------------------------------- grass blades

struct Blade {
  a: vec3f,   // base
  m: vec3f,   // knee
  b: vec3f,   // tip
  r0: f32,    // lower radius
  r1: f32,    // upper radius
  h: f32,
  seed: f32,
}

// Each grid cell grows three blades, kept inside their own cell so the grid walk below
// stays exact. Heights fade to zero toward the blade radius, where the textured heightfield
// takes over.
fn bladeAt(cell: vec2f, k: f32) -> Blade {
  var bl: Blade;
  let s1 = hash12(cell + vec2f(k * 17.3, 0.7));
  let s2 = hash12(cell + vec2f(3.1, k * 29.7 + 5.3));
  let s3 = hash12(cell + vec2f(k * 7.7 + 11.1, 23.9));
  let s4 = hash12(cell + vec2f(41.3, k * 13.1 + 2.2));
  let base = (cell + vec2f(0.5) + (vec2f(s1, s2) - 0.5) * 0.36) * CELL;
  let dist = length(base);
  let patchiness = clamp(0.35 + 1.4 * (fbm3(base * 0.45 + vec2f(9.0, 4.0)) - 0.3), 0.25, 1.4);
  var h = params.grass.y * (0.28 + 0.72 * s3 * s3) * patchiness;
  h *= 1.0 - smoothstep(params.grass.x * 0.62, params.grass.x * 0.98, dist);
  let ang = s4 * 6.2831853;
  let lean = vec2f(cos(ang), sin(ang)) * (0.008 + 0.011 * s2) * (0.4 + 0.6 * h / max(params.grass.y, 0.01));
  bl.a = vec3f(base.x, 0.0, base.y);
  bl.m = bl.a + vec3f(lean.x * 0.55, h * 0.58, lean.y * 0.55);
  bl.b = bl.a + vec3f(lean.x * 1.6, h * 0.97, lean.y * 1.6);
  bl.r0 = 0.0024 + 0.0014 * s1;
  bl.r1 = bl.r0 * 0.45;
  bl.h = h;
  bl.seed = s3 * 0.6 + s4 * 0.4;
  return bl;
}

// Ray / capsule intersection (nearest positive t or -1).
fn iCapsule(ro: vec3f, rd: vec3f, pa: vec3f, pb: vec3f, r: f32) -> f32 {
  let ba = pb - pa;
  let oa = ro - pa;
  let baba = dot(ba, ba);
  let bard = dot(ba, rd);
  let baoa = dot(ba, oa);
  let rdoa = dot(rd, oa);
  let oaoa = dot(oa, oa);
  let a = baba - bard * bard;
  var b = baba * rdoa - baoa * bard;
  var c = baba * oaoa - baoa * baoa - r * r * baba;
  var h = b * b - a * c;
  if (h >= 0.0) {
    let t = (-b - sqrt(h)) / max(a, 1e-9);
    let y = baoa + t * bard;
    if (y > 0.0 && y < baba && t > 0.0) { return t; }
    // End caps.
    let oc = select(oa, ro - pb, y > 0.0);
    b = dot(rd, oc);
    c = dot(oc, oc) - r * r;
    h = b * b - c;
    if (h > 0.0) {
      let tc = -b - sqrt(h);
      if (tc > 0.0) { return tc; }
    }
  }
  return -1.0;
}

struct BladeHit {
  t: f32,
  n: vec3f,
  up: f32,     // 0 at the base, 1 at the tip
  seed: f32,
}

fn segNormal(p: vec3f, a: vec3f, b: vec3f) -> vec3f {
  let ba = b - a;
  let h = clamp(dot(p - a, ba) / dot(ba, ba), 0.0, 1.0);
  return normalize(p - (a + ba * h));
}

// Walks the blade grid along the ray (2D DDA over x/z cells) while the ray is inside the
// blade slab and radius. Exact ray/capsule hits; nearest wins.
fn traceBlades(ro: vec3f, rd: vec3f, tMin: f32, tMax: f32, maxCells: i32) -> BladeHit {
  var res: BladeHit;
  res.t = -1.0;
  res.n = vec3f(0.0, 1.0, 0.0);
  res.up = 0.0;
  res.seed = 0.0;
  let hMax = params.grass.y;
  let radius = params.grass.x;
  var t0 = tMin;
  var t1 = tMax;
  if (abs(rd.y) > 1e-6) {
    let ta = (0.0 - ro.y) / rd.y;
    let tb = (hMax - ro.y) / rd.y;
    t0 = max(t0, min(ta, tb));
    t1 = min(t1, max(ta, tb));
  } else if (ro.y < 0.0 || ro.y > hMax) {
    return res;
  }
  let a2 = dot(rd.xz, rd.xz);
  if (a2 > 1e-8) {
    let b2 = dot(ro.xz, rd.xz);
    let c2 = dot(ro.xz, ro.xz) - radius * radius;
    let disc = b2 * b2 - a2 * c2;
    if (disc < 0.0) { return res; }
    let sq = sqrt(disc);
    t0 = max(t0, (-b2 - sq) / a2);
    t1 = min(t1, (-b2 + sq) / a2);
  }
  if (t1 <= t0) { return res; }
  let start = ro + rd * (t0 + 1e-4);
  var cell = floor(start.xz / CELL);
  let stepDir = vec2f(select(-1.0, 1.0, rd.x >= 0.0), select(-1.0, 1.0, rd.z >= 0.0));
  let safeDir = vec2f(select(rd.x, 1e-6, abs(rd.x) < 1e-6), select(rd.z, 1e-6, abs(rd.z) < 1e-6));
  let invDir = vec2f(1.0) / safeDir;
  let tDelta = abs(CELL * invDir);
  let nextBoundary = (cell + max(stepDir, vec2f(0.0))) * CELL;
  var tNext = (nextBoundary - ro.xz) * invDir;
  var best = t1;
  var bestCell = cell;
  var bestK = -1.0;
  var bestSeg = 0.0;
  for (var i = 0; i < maxCells; i++) {
    let tExit = min(tNext.x, tNext.y);
    for (var k = 0; k < 3; k++) {
      let bl = bladeAt(cell, f32(k));
      if (bl.h > 0.004) {
        let tA = iCapsule(ro, rd, bl.a, bl.m, bl.r0);
        if (tA > t0 - 0.02 && tA < best) { best = tA; bestCell = cell; bestK = f32(k); bestSeg = 0.0; }
        let tB = iCapsule(ro, rd, bl.m, bl.b, bl.r1);
        if (tB > t0 - 0.02 && tB < best) { best = tB; bestCell = cell; bestK = f32(k); bestSeg = 1.0; }
      }
    }
    if (tExit >= best) { break; }
    if (tNext.x < tNext.y) {
      cell.x += stepDir.x;
      tNext.x += tDelta.x;
    } else {
      cell.y += stepDir.y;
      tNext.y += tDelta.y;
    }
  }
  if (bestK < 0.0) { return res; }
  let bl = bladeAt(bestCell, bestK);
  let p = ro + rd * best;
  res.t = best;
  if (bestSeg < 0.5) {
    res.n = segNormal(p, bl.a, bl.m);
  } else {
    res.n = segNormal(p, bl.m, bl.b);
  }
  res.up = clamp((p.y - bl.a.y) / max(bl.h, 0.01), 0.0, 1.0);
  res.seed = bl.seed;
  return res;
}

// Blade occlusion toward a light: blades are thin and translucent, so a hit only dims.
fn bladeShadow(p: vec3f, l: vec3f, maxT: f32, maxCells: i32) -> f32 {
  if (params.grass.z < 0.5) { return 1.0; }
  let hit = traceBlades(p, l, 0.003, maxT, maxCells);
  return select(1.0, 0.3, hit.t > 0.0);
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
  // Broad undulations of the face and a low windward hump to the sides.
  let swell = 0.3 + 0.7 * smoothstep(0.0, 4.0, rise);
  h += 1.8 * (fbm3(p * 0.16 + vec2f(7.0, 1.0)) - 0.5) * swell;
  h += 0.4 * (fbm3(p * 0.42 + vec2f(1.0, 8.0)) - 0.5) * swell;
  h += 0.18 * (fbm3(p * 0.9 + vec2f(3.0, 9.0)) - 0.5) * smoothstep(0.0, 2.0, rise);
  return h;
}

// Wind ripples: asymmetric waves with wandering crests, 8-9 cm apart.
fn ripple(p: vec2f) -> f32 {
  let dir = vec2f(0.22, 0.975);
  // Crests wander with two warps so lines bend, merge and fork like wind ripples do.
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
  albedo *= 1.0 + 0.16 * (hash13(floor(p * 700.0)) - 0.5) * (1.0 - smoothstep(0.002, 0.012, footprint));
  let n0 = duneNormal(p.xz);
  // Per-grain roughness for sparkle; fades with distance so it does not turn to noise.
  let grainFade = 1.0 - smoothstep(0.002, 0.012, footprint);
  let g = vec3f(hash13(p * 431.0), hash13(p * 517.0 + vec3f(3.0)), hash13(p * 619.0 + vec3f(7.0))) - 0.5;
  let n = normalize(n0 + g * 0.35 * grainFade);
  let sun = normalize(vec3f(-0.62, 0.22, -0.55));
  let sunColor = vec3f(1.3, 0.96, 0.56) * 1.2;
  let skyAmb = vec3f(0.34, 0.16, 0.07) * 0.7;
  let ndl = max(dot(n0, sun), 0.0);
  // Ripple crests shade their own troughs: cheap occlusion from the ripple phase.
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
    // Only reachable for rays that skim over the crest: a warm haze instead of sky.
    return rgb8(236.0, 190.0, 120.0) * 0.8;
  }
  let p = ro + rd * t;
  var col = shadeSand(p, rd, footprintScale * t);
  // Dusty haze softens the far face.
  col = mix(col, rgb8(236.0, 190.0, 120.0) * 0.8, 1.0 - exp(-t * 0.012));
  return col;
}

// ---------------------------------------------------------------- lighting

const AMBIENT: vec3f = vec3f(0.035, 0.09, 0.16);

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

fn stars(rd: vec3f) -> f32 {
  let d = normalize(rd);
  let sph = vec2f(atan2(d.z, d.x), asin(clamp(d.y, -1.0, 1.0)));
  let grid = sph * vec2f(260.0, 260.0);
  let cell = floor(grid);
  let local = fract(grid) - 0.5;
  let seed = hash12(cell);
  let jitter = vec2f(hash12(cell + vec2f(7.0, 3.0)), hash12(cell + vec2f(2.0, 9.0))) - 0.5;
  let r = length(local - jitter * 0.8);
  let point = smoothstep(0.13, 0.0, r) * step(0.965, seed);
  return point * (0.3 + 0.7 * hash12(cell + vec2f(5.0, 1.0)));
}

fn skyNight(rd: vec3f) -> vec3f {
  let horizon = rgb8(26.0, 44.0, 84.0);
  let mid = rgb8(9.0, 16.0, 46.0);
  let zenith = rgb8(3.0, 6.0, 24.0);
  let y = max(rd.y, 0.0);
  var c = mix(horizon, mid, smoothstep(0.0, 0.25, y));
  c = mix(c, zenith, smoothstep(0.25, 0.8, y));
  let m = moonDir();
  c += vec3f(0.35, 0.42, 0.6) * pow(max(dot(rd, m), 0.0), 40.0) * 0.12;
  c += vec3f(0.6, 0.7, 0.9) * stars(rd) * 0.8 * smoothstep(0.02, 0.12, rd.y);
  return c;
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

// Light from the sunlit dune pouring through the opening: a warm rectangular area light
// sampled at three points, each with its own shadow ray, so penumbras stay soft.
fn doorLight(p: vec3f, n: vec3f, transl: f32, f: DoorFrame, shadows: bool) -> vec3f {
  let front = dot(p - f.origin, f.fwd);
  if (front > -0.01) { return vec3f(0.0); }
  var sum = 0.0;
  let offsets = array<vec2f, 3>(vec2f(-0.22, 0.42), vec2f(0.21, 1.12), vec2f(-0.07, 1.78));
  for (var i = 0; i < 3; i++) {
    let o = offsets[i];
    let s = f.origin + f.right * o.x + vec3f(0.0, o.y, 0.0);
    let toL = s - p;
    let d = max(length(toL), 0.05);
    let l = toL / d;
    let facing = max(dot(f.fwd, l), 0.0);
    let geom = facing / (d * d + 0.6);
    let ndl = dot(n, l);
    var diffuse = max(ndl, 0.0) + transl * max(-ndl, 0.0);
    // A little wrap so blade edges do not cut to black.
    diffuse += 0.12 * (1.0 - abs(ndl));
    var vis = 1.0;
    if (shadows) {
      vis = objectShadow(p + n * 0.002, l, f, d - 0.3, 12.0);
      if (vis > 0.01) { vis *= bladeShadow(p + n * 0.007 + l * 0.01, l, d - 0.02, 56); }
    }
    sum += geom * diffuse * vis;
  }
  return rgb8(255.0, 214.0, 156.0) * params.look.w * sum / 3.0;
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
  g.albedoMod = 1.0 + strength * (blades * 0.85 + clumps * 0.32 + tufts * 0.26 + patches * 0.18);
  g.warm = max(blades, 0.0) * 0.7 + smoothstep(0.6, 0.9, vnoise(p.xz * 45.0)) * bladeFade * 0.4;
  return g;
}

fn nightBase(albedo: vec3f, n: vec3f) -> vec3f {
  // Scotopic look: the eye loses colour at night, so drift toward a cool grey-green.
  let lumA = dot(albedo, vec3f(0.2126, 0.7152, 0.0722));
  let nightAlbedo = mix(albedo, lumA * vec3f(0.65, 0.85, 1.0), 0.4);
  let moon = moonDir();
  let moonColor = vec3f(0.28, 0.4, 0.68) * 0.13;
  let nightAmbient = vec3f(0.006, 0.01, 0.026);
  let hemi = 0.5 + 0.5 * n.y;
  return nightAlbedo * (moonColor * max(dot(n, moon), 0.0) + nightAmbient * hemi);
}

fn rimGlow(n: vec3f, rim: f32) -> vec3f {
  return vec3f(1.0, 0.7, 0.45) * rim * 4.0 * (0.4 + 0.6 * max(dot(n, normalize(vec3f(0.0, 0.6, 1.0))), 0.0));
}

fn shadeGround(p: vec3f, n0: vec3f, rd: vec3f, t: f32, footprint: f32, f: DoorFrame, dayMix: f32, rim: f32) -> vec3f {
  let tex = grassTexture(p, footprint);
  let dist = length(p.xz);
  // Under the blade grid the ground is dark thatch; out in the field it is the reference lawn.
  let lawn = rgb8(90.0, 132.0, 47.0) * tex.albedoMod;
  let thatch = rgb8(34.0, 46.0, 18.0) * (0.7 + 0.6 * tex.albedoMod);
  let bladeMix = 1.0 - smoothstep(params.grass.x * 0.62, params.grass.x * 0.98, dist);
  var albedo = mix(lawn, thatch, bladeMix);
  albedo = mix(albedo, albedo * vec3f(1.25, 1.08, 0.7), clamp(tex.warm, 0.0, 1.0) * 0.5 * (1.0 - bladeMix));
  let slope = 1.0 - n0.y;
  albedo *= mix(vec3f(1.0), vec3f(0.62, 0.74, 0.7), smoothstep(0.02, 0.22, slope));
  let n = n0;
  let sun = sunDir();
  let v = -rd;

  // --- Day: sun + blue sky ambient tuned so flat ground = albedo.
  let sunCol = sunColor();
  let ndl = max(dot(n, sun), 0.0);
  var sh = terrainShadow(p, sun) * objectShadow(p, sun, f, 8.0, 5.0);
  if (bladeMix > 0.01) { sh *= mix(1.0, bladeShadow(p + n * 0.003, sun, 0.8, 24), bladeMix); }
  let hemi = 0.5 + 0.5 * n.y;
  var day = albedo * (sunCol * ndl * sh + AMBIENT * hemi);
  let h = normalize(sun + v);
  day += albedo * pow(max(dot(n, h), 0.0), 5.0) * 0.10 * sh * sunCol;

  // --- Night: moon fill + the door spill.
  var night = nightBase(albedo, n);
  if (dayMix < 0.999) {
    night += mix(albedo, vec3f(dot(albedo, vec3f(0.33))), 0.2) * doorLight(p, n, 0.0, f, true);
  }
  return mix(night, day, dayMix) + albedo * rimGlow(n, rim);
}

fn shadeBlade(p: vec3f, hit: BladeHit, rd: vec3f, f: DoorFrame, dayMix: f32, rim: f32) -> vec3f {
  // Darker, bluer base fading to a lighter yellow-green tip; each blade gets its own tint.
  let base = rgb8(58.0, 98.0, 32.0);
  let tip = rgb8(152.0, 178.0, 70.0);
  var albedo = mix(base, tip, smoothstep(0.1, 1.0, hit.up));
  albedo *= 0.75 + 0.5 * hit.seed;
  albedo *= mix(vec3f(1.0), vec3f(1.12, 1.0, 0.8), hit.seed * 0.5);
  var n = hit.n;
  let v = -rd;
  // Two-sided thin surface.
  if (dot(n, v) < 0.0) { n = -n; }
  let sun = sunDir();
  let sunCol = sunColor();
  let ndl = dot(n, sun);
  let sh = objectShadow(p + n * 0.003, sun, f, 8.0, 5.0) * bladeShadow(p + n * 0.007 + sun * 0.01, sun, 0.9, 28);
  let hemi = 0.5 + 0.5 * n.y;
  // Diffuse plus transmission: blades between the viewer and the sun glow.
  var day = albedo * (sunCol * (max(ndl, 0.0) + 0.45 * max(-ndl, 0.0)) * sh + AMBIENT * hemi * 1.2);
  let h = normalize(sun + v);
  day += pow(max(dot(n, h), 0.0), 24.0) * 0.08 * sh * sunCol;

  var night = nightBase(albedo, n) * 1.3;
  if (dayMix < 0.999) {
    night += albedo * doorLight(p, n, 0.9, f, true);
  }
  return mix(night, day, dayMix) + albedo * rimGlow(n, rim);
}

fn shadeObject(p: vec3f, n: vec3f, rd: vec3f, mat: f32, f: DoorFrame, dayMix: f32, rim: f32) -> vec3f {
  var albedo = vec3f(0.85, 0.86, 0.86);     // white plastic chair
  if (mat < 1.5) {
    albedo = rgb8(86.0, 94.0, 120.0);        // slate blue door
  }
  let sun = sunDir();
  let v = -rd;
  let sunCol = sunColor() * 0.784;
  let ndl = max(dot(n, sun), 0.0);
  let sh = objectShadow(p + n * 0.003, sun, f, 6.0, 5.0) * terrainShadow(p, sun);
  let hemi = 0.5 + 0.5 * n.y;
  var day = albedo * (sunCol * ndl * sh * 0.85 + AMBIENT * hemi * 1.4);
  let h = normalize(sun + v);
  day += pow(max(dot(n, h), 0.0), 40.0) * 0.25 * sh * sunCol;

  var night = nightBase(albedo, n) * 1.2;
  if (dayMix < 0.999) {
    let spill = doorLight(p, n, 0.0, f, true) * 0.55;
    night += albedo * spill;
    // Glossy paint picks up a highlight of the opening.
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
  let obj = marchObjects(ro, rd, tLimit, f);
  if (obj.t > 0.0) { tLimit = min(tLimit, obj.t); }
  let blade = traceBlades(ro, rd, 0.0, tLimit, 240);

  var t = tTerrain;
  var kind = 0;   // 0 sky, 1 terrain, 2 object, 3 blade
  if (t > 0.0) { kind = 1; }
  if (obj.t > 0.0 && (t < 0.0 || obj.t < t)) { t = obj.t; kind = 2; }
  if (blade.t > 0.0 && (t < 0.0 || blade.t < t)) { t = blade.t; kind = 3; }
  let throughDoor = tPortal > 0.0 && (t < 0.0 || tPortal < t);

  var color = vec3f(0.0);
  if (throughDoor) {
    // The opening is a window onto the dune: continue the ray in door-local space.
    let q = toDoor(ro + rd * tPortal, f);
    let ld = vec3f(dot(rd, f.right), rd.y, dot(rd, f.fwd));
    color = renderSand(vec3f(q.x, q.y, 0.0), normalize(ld), pixelAngle);
    return color;
  }
  if (kind == 0) {
    let far = ro + rd * 120.0;
    let front = dayFront(far, f);
    color = mix(skyNight(rd), skyDay(rd), front.x);
    color += vec3f(1.0, 0.6, 0.35) * front.y * 0.2;
    return color;
  }
  let p = ro + rd * t;
  let front = dayFront(p, f);
  let dayMix = front.x;
  let rim = front.y;
  let footprint = pixelAngle * t;
  if (kind == 3) {
    color = shadeBlade(p, blade, rd, f, dayMix, rim);
  } else if (kind == 2) {
    let n = objectNormal(p, f);
    color = shadeObject(p, n, rd, obj.mat, f, dayMix, rim);
  } else {
    let n = terrainNormal(p, max(0.08, footprint * 0.5));
    color = shadeGround(p, n, rd, t, footprint, f, dayMix, rim);
  }
  // Aerial perspective: night haze is heavier than the crisp day.
  let fogNight = skyNight(vec3f(rd.x, 0.02, rd.z)) * 0.9;
  let fogDay = skyDay(vec3f(rd.x, 0.03, rd.z));
  let fog = mix(fogNight, fogDay, dayMix);
  let fogAmt = 1.0 - exp(-t * 0.0028);
  color = mix(color, fog, fogAmt * mix(0.6, 0.05, dayMix));
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
