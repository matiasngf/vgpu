// Dreamcore: a Bliss-like green field at night with a door that opens into the day.
// Fullscreen raymarcher. Linear HDR output; tone mapping + bloom + grain happen in post.

struct Params {
  resolution: vec2f,
  time: f32,
  phase: f32,        // 0 = night, 1 = day (the day expands out of the door)
  camera: vec4f,     // height, pitch (rad, + looks up), vertical fov (rad), aa samples
  door: vec4f,       // x, z, yaw (rad), leaf angle (rad)
  chair: vec4f,      // x, z, yaw (rad), scale
  look: vec4f,       // sun azimuth (rad), sun elevation (rad), texture strength, door light
}

@group(0) @binding(0) var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const DOOR_W: f32 = 0.92;   // opening width
const DOOR_H: f32 = 2.08;   // opening height
const FRAME_T: f32 = 0.085; // frame member width
const FRAME_D: f32 = 0.07;  // frame half depth
const LEAF_D: f32 = 0.022;  // leaf half thickness
const TMAX: f32 = 170.0;
const PORTAL_PITCH: f32 = 0.30; // rays through the door tilt up so the horizon sits inside the frame
const EXIT_POS: vec2f = vec2f(-15.0, 20.0); // the door opens onto the open plain, looking west
const EXIT_YAW: f32 = 1.5707963;

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

fn bumpR(p: vec2f, c: vec2f, r: vec2f, ang: f32) -> f32 {
  let d = (rot2(ang) * (p - c)) / r;
  return exp(-dot(d, d));
}

fn terrainHeight(p: vec2f) -> f32 {
  // The camera stands on a plateau; beyond its edge the land falls away so the sky
  // dips below eye level between the hills, exactly like the reference.
  let r = length(p);
  var h = -0.28 * max(r - 78.0, 0.0);
  // Portrait view is a narrow corridor (about +-12 degrees), so the hills sit close to x = 0.
  // Left ridge: near foot at the left edge, crest running back toward the centre, its
  // right flank turned away from the sun.
  // Skyline fitted numerically against the reference photo (rms < 1% of frame height).
  h += 3.64 * bump(p, vec2f(-9.3, 40.0), vec2f(4.2, 11.4));
  h += 2.5 * bump(p, vec2f(-4.0, 47.0), vec2f(4.9, 5.2));
  h += 1.98 * bump(p, vec2f(-15.8, 43.7), vec2f(6.7, 8.7));
  h += 2.6 * bump(p, vec2f(-29.8, 52.0), vec2f(10.9, 8.2));
  // Right dome, the tallest silhouette, plus a low ridge in front of it.
  h += 4.99 * bump(p, vec2f(10.2, 54.0), vec2f(9.3, 6.9));
  h += 2.19 * bump(p, vec2f(11.0, 40.0), vec2f(8.2, 4.0));
  // Distant hills peeking over the plateau edge (the second one is only seen through the door).
  h += 12.18 * bump(p, vec2f(-2.0, 125.9), vec2f(15.5, 10.7));
  h += 19.0 * bump(p, vec2f(-130.0, 20.0), vec2f(24.0, 16.0));
  // Wider hills that only show up in landscape framings.
  h += 4.5 * bump(p, vec2f(-34.0, 60.0), vec2f(15.0, 13.0));
  h += 4.8 * bump(p, vec2f(34.0, 64.0), vec2f(15.0, 14.0));
  h += 3.0 * bump(p, vec2f(-52.0, 45.0), vec2f(14.0, 12.0));
  h += 3.4 * bump(p, vec2f(52.0, 48.0), vec2f(14.0, 12.0));
  // Gentle rolling of the plain.
  h += 0.3 * fbm(p * 0.045 + vec2f(3.1, 7.7));
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
    // The heightfield slope never exceeds ~1, so stepping by half the vertical gap is safe;
    // the distance term keeps grazing rays from crawling.
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

// Soft shadow of the terrain itself toward a light direction.
fn terrainShadow(p: vec3f, l: vec3f) -> f32 {
  var s = 1.0;
  var t = 0.6;
  for (var i = 0; i < 24; i++) {
    let q = p + l * t;
    let d = q.y - terrainHeight(q.xz);
    s = min(s, clamp(5.0 * d / t, 0.0, 1.0));
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
  // Big arched panel: a rounded rectangle with a large radius.
  let up = vec2f(q.x - cx, q.y - 1.34);
  var panel = sdBox2(up, vec2f(0.10, 0.34)) - 0.17;
  // Two small lower panels.
  let lx = abs(q.x - cx) - 0.16;
  let low = vec2f(lx, q.y - 0.40);
  let panel2 = sdBox2(low, vec2f(0.055, 0.19)) - 0.03;
  panel = min(panel, panel2);
  // Only the outer 5mm of each face is carved.
  let depth = (LEAF_D - 0.005) - abs(q.z);
  return max(panel, depth);
}

fn sdDoor(p: vec3f, f: DoorFrame) -> f32 {
  let q = toDoor(p, f);
  let hw = DOOR_W * 0.5;
  // Frame: outer slab minus the opening (opening runs through the whole depth).
  let outer = sdBox(q - vec3f(0.0, (DOOR_H + FRAME_T) * 0.5 - 0.05, 0.0),
                    vec3f(hw + FRAME_T, (DOOR_H + FRAME_T) * 0.5 + 0.05, FRAME_D)) - 0.004;
  let opening = sdBox(q - vec3f(0.0, DOOR_H * 0.5 - 0.1, 0.0), vec3f(hw, DOOR_H * 0.5 + 0.1, 1.0));
  var d = max(outer, -opening);
  // Leaf hinged on the right jamb, swung toward the viewer by params.door.w.
  let a = params.door.w;
  let u = vec3f(-cos(a), 0.0, -sin(a));      // along the leaf from the hinge
  let n = vec3f(-u.z, 0.0, u.x);             // leaf normal
  let hinge = vec3f(hw, 0.0, 0.0);
  let r = q - hinge;
  let l = vec3f(dot(r, u), r.y, dot(r, n));
  let leafBox = vec3f(l.x - hw, l.y - DOOR_H * 0.5 - 0.005, l.z);
  var leaf = sdBox(leafBox, vec3f(hw - 0.004, DOOR_H * 0.5 - 0.012, LEAF_D)) - 0.003;
  leaf = max(leaf, -leafPanels(l));
  // Knob on the free edge side.
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
  // Seat.
  var d = sdBox(p - vec3f(0.0, 0.43, 0.0), vec3f(0.235, 0.018, 0.22)) - 0.012;
  // Backrest, tilted back 13 degrees, rounded top, five slots.
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
  // Armrests.
  let ax = abs(p.x);
  let arm = sdBox(vec3f(ax - 0.24, p.y - 0.63, p.z + 0.02), vec3f(0.028, 0.012, 0.19)) - 0.008;
  let armFront = sdBox(vec3f(ax - 0.24, p.y - 0.53, p.z + 0.2), vec3f(0.018, 0.1, 0.014)) - 0.006;
  d = min(d, min(arm, armFront));
  // Legs (front straight, back splayed).
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

// Object scene SDF. Returns (distance, material id).
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
  // Bounding spheres keep the SDF evaluation off most rays.
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
    if (dm.x < 0.0006 * max(t, 1.0)) {
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
  let e = 0.0012;
  let k = vec2f(1.0, -1.0);
  return normalize(
    k.xyy * sdObjects(p + k.xyy * e, f).x +
    k.yyx * sdObjects(p + k.yyx * e, f).x +
    k.yxy * sdObjects(p + k.yxy * e, f).x +
    k.xxx * sdObjects(p + k.xxx * e, f).x);
}

fn objectShadow(p: vec3f, l: vec3f, f: DoorFrame, maxT: f32) -> f32 {
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
    s = min(s, clamp(14.0 * d / t, 0.0, 1.0));
    if (s < 0.005) { break; }
    t += clamp(d, 0.01, 0.4);
    if (t > min(tEnd, maxT)) { break; }
  }
  return s;
}

// ---------------------------------------------------------------- portal

// Distance along the ray to the door opening plane, or -1 when the ray misses the opening.
fn portalHit(ro: vec3f, rd: vec3f, f: DoorFrame) -> f32 {
  let denom = dot(rd, f.fwd);
  if (abs(denom) < 1e-5) { return -1.0; }
  let t = dot(f.origin - ro, f.fwd) / denom;
  if (t < 0.0) { return -1.0; }
  let q = toDoor(ro + rd * t, f);
  if (abs(q.x) < DOOR_W * 0.5 && q.y > 0.0 && q.y < DOOR_H) { return t; }
  return -1.0;
}

// ---------------------------------------------------------------- lighting

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

const AMBIENT: vec3f = vec3f(0.035, 0.09, 0.16);

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
  let horizon = rgb8(24.0, 42.0, 82.0);
  let mid = rgb8(8.0, 15.0, 44.0);
  let zenith = rgb8(3.0, 6.0, 24.0);
  let y = max(rd.y, 0.0);
  var c = mix(horizon, mid, smoothstep(0.0, 0.25, y));
  c = mix(c, zenith, smoothstep(0.25, 0.8, y));
  // Faint moon glow.
  let m = moonDir();
  c += vec3f(0.35, 0.42, 0.6) * pow(max(dot(rd, m), 0.0), 40.0) * 0.12;
  c += vec3f(0.6, 0.7, 0.9) * stars(rd) * 0.8 * smoothstep(0.02, 0.12, rd.y);
  return c;
}

// Expanding day front, centred on the door. Returns day mix in [0,1] and the rim glow.
fn dayFront(p: vec3f, f: DoorFrame) -> vec2f {
  let ph = clamp(params.phase, 0.0, 1.0);
  // Ease so the wave starts slow, then sweeps.
  let e = ph * ph * (3.0 - 2.0 * ph);
  let radius = e * 210.0 - 0.6;
  let d = length(p - (f.origin + vec3f(0.0, 1.0, 0.0)));
  let edge = 1.5 + radius * 0.08;
  let day = 1.0 - smoothstep(radius - edge, radius + edge, d);
  let rim = exp(-pow((d - radius) / (edge * 1.6), 2.0)) * (1.0 - smoothstep(0.85, 1.0, ph)) * smoothstep(0.0, 0.05, ph);
  return vec2f(day, rim);
}

struct Grass {
  albedoMod: f32,
  warm: f32,
  bend: vec2f,
}

// Grass micro texture. footprint = world size of one pixel at the shading point.
fn grassTexture(p: vec3f, footprint: f32) -> Grass {
  let strength = params.look.z;
  var g: Grass;
  // Fixed depth squash: the plain is seen at a grazing angle, so world-isotropic noise
  // would smear into horizontal streaks on screen.
  let sq = vec2f(1.0, 0.45);
  // Blades: anisotropic high-frequency noise, stretched along the view depth (z) so the
  // foreshortened result reads as short vertical strokes.
  let fBlade = 70.0;
  let bladeFade = 1.0 - smoothstep(0.3, 1.3, footprint * fBlade);
  var blades = 0.0;
  if (bladeFade > 0.001) {
    let n1 = vnoise(vec2f(p.x * fBlade, p.z * fBlade * 0.12));
    let n2 = vnoise(vec2f(p.x * fBlade * 2.1 + 13.1, p.z * fBlade * 0.3 + 7.3));
    blades = ((n1 - 0.5) * 1.0 + (n2 - 0.5) * 0.6) * bladeFade;
  }
  // Clumps: mid frequency, isotropic.
  let fClump = 6.0;
  let clumpFade = 1.0 - smoothstep(0.3, 1.2, footprint * fClump);
  var clumps = 0.0;
  if (clumpFade > 0.001) {
    clumps = (fbm3(p.xz * sq * fClump) - 0.5) * clumpFade;
  }
  // Tufts: the velvet texture that survives on the far hills.
  let fTuft = 4.5;
  let tuftFade = 1.0 - smoothstep(0.35, 1.3, footprint * fTuft);
  var tufts = 0.0;
  var bend = vec2f(0.0);
  if (tuftFade > 0.001) {
    let e = 0.12;
    let t0 = fbm3(p.xz * fTuft + vec2f(5.0, 2.0));
    let tx = fbm3((p.xz + vec2f(e, 0.0)) * fTuft + vec2f(5.0, 2.0));
    let tz = fbm3((p.xz + vec2f(0.0, e)) * fTuft + vec2f(5.0, 2.0));
    tufts = (t0 - 0.5) * tuftFade;
    bend = vec2f(tx - t0, tz - t0) / e * tuftFade;
  }
  // Patches: very low frequency colour drift.
  let patches = fbm3(p.xz * 0.35 + vec2f(11.0, 3.0)) - 0.5;
  g.albedoMod = 1.0 + strength * (blades * 0.85 + clumps * 0.32 + tufts * 0.26 + patches * 0.18);
  // Yellow flecks: a warm shift where the blade noise peaks.
  g.warm = max(blades, 0.0) * 0.7 + smoothstep(0.6, 0.9, vnoise(p.xz * 45.0)) * bladeFade * 0.4;
  g.bend = bend;
  return g;
}

fn shadeGround(p: vec3f, n0: vec3f, rd: vec3f, t: f32, footprint: f32, f: DoorFrame, dayMix: f32, rim: f32) -> vec3f {
  let tex = grassTexture(p, footprint);
  // Base grass albedo measured from the reference (sRGB 86,128,45).
  var albedo = rgb8(90.0, 132.0, 47.0) * tex.albedoMod;
  albedo = mix(albedo, albedo * vec3f(1.25, 1.08, 0.7), clamp(tex.warm, 0.0, 1.0) * 0.5);
  let slope = 1.0 - n0.y;
  albedo *= mix(vec3f(1.0), vec3f(0.62, 0.74, 0.7), smoothstep(0.02, 0.22, slope));
  // Bend the normal with the tuft field so the velvet catches the sun.
  let n = normalize(n0 + vec3f(-tex.bend.x, 0.0, -tex.bend.y) * 0.06);
  let sun = sunDir();
  let v = -rd;

  // --- Day lighting: sun + blue sky ambient tuned so flat ground = albedo.
  let sunColor = sunColor();
  let ambient = AMBIENT;
  let ndl = max(dot(n, sun), 0.0);
  var sh = terrainShadow(p, sun);
  sh *= objectShadow(p, sun, f, 8.0);
  let hemi = 0.5 + 0.5 * n.y;
  var day = albedo * (sunColor * ndl * sh + ambient * hemi);
  // Grass sheen: forward scatter that brightens ground seen against the sun.
  let h = normalize(sun + v);
  let sheen = pow(max(dot(n, h), 0.0), 5.0) * 0.10 * sh;
  day += albedo * sheen * sunColor;
  if (params.time < -0.5 && params.time > -1.5) { return vec3f(sh); }
  if (params.time < -1.5 && params.time > -2.5) { return vec3f(rim); }
  if (params.time < -2.5 && params.time > -3.5) { return vec3f(dayMix); }

  // --- Night lighting: moon + dark blue sky + the door spill.
  let moon = moonDir();
  let moonColor = vec3f(0.3, 0.42, 0.68) * 0.26;
  let nightAmbient = vec3f(0.010, 0.015, 0.036);
  let mdl = max(dot(n, moon), 0.0);
  let msh = terrainShadow(p, moon) * objectShadow(p, moon, f, 8.0);
  // Scotopic look: the eye loses colour at night, so drift the grass toward a cool grey-green.
  let lumA = dot(albedo, vec3f(0.2126, 0.7152, 0.0722));
  let nightAlbedo = mix(albedo, lumA * vec3f(0.65, 0.85, 1.0), 0.4);
  var night = nightAlbedo * (moonColor * mdl * msh + nightAmbient * hemi);
  // Door light: the opening is a bright rectangle facing -fwd (toward the viewer side).
  let doorCenter = f.origin + vec3f(0.0, DOOR_H * 0.55, 0.0);
  let toL = doorCenter - p;
  let dist = max(length(toL), 0.3);
  let l = toL / dist;
  let facing = max(dot(-f.fwd, -l), 0.0);   // only the near side of the door is lit
  let spill = params.look.w * facing * max(dot(n, l), 0.0) / (dist * dist + 0.6);
  let spillSh = objectShadow(p, l, f, dist - 0.1);
  let doorColor = vec3f(1.0, 0.95, 0.85);
  night += albedo * doorColor * spill * spillSh;
  // Rim of the expanding day front.
  let rimLight = vec3f(1.0, 0.72, 0.5) * rim * 1.8 * (0.4 + 0.6 * max(dot(n, normalize(vec3f(0.0, 0.6, 1.0))), 0.0));
  return mix(night, day, dayMix) + albedo * rimLight;
}

fn shadeObject(p: vec3f, n: vec3f, rd: vec3f, mat: f32, f: DoorFrame, dayMix: f32, rim: f32) -> vec3f {
  var albedo = vec3f(0.85, 0.86, 0.86);     // white plastic chair
  if (mat < 1.5) {
    albedo = rgb8(122.0, 132.0, 160.0);      // slate blue door
  }
  let sun = sunDir();
  let v = -rd;
  let sunColor = sunColor() * 0.784;
  let ambient = AMBIENT;
  let ndl = max(dot(n, sun), 0.0);
  let sh = objectShadow(p + n * 0.003, sun, f, 6.0) * terrainShadow(p, sun);
  let hemi = 0.5 + 0.5 * n.y;
  var day = albedo * (sunColor * ndl * sh * 0.85 + ambient * hemi * 1.4);
  let h = normalize(sun + v);
  let spec = pow(max(dot(n, h), 0.0), 40.0) * 0.25 * sh;
  day += spec * sunColor;

  let moon = moonDir();
  let moonColor = vec3f(0.3, 0.4, 0.65) * 0.16;
  let nightAmbient = vec3f(0.008, 0.012, 0.03);
  let mdl = max(dot(n, moon), 0.0);
  var night = albedo * (moonColor * mdl + nightAmbient * hemi * 1.4);
  let doorCenter = f.origin + vec3f(0.0, DOOR_H * 0.55, 0.0);
  let toL = doorCenter - p;
  let dist = max(length(toL), 0.3);
  let l = toL / dist;
  let facing = max(dot(-f.fwd, -l), 0.0);
  let spill = params.look.w * (facing + 0.06) * max(dot(n, l), 0.0) / (dist * dist + 0.6);
  let spillSh = objectShadow(p + n * 0.003, l, f, dist - 0.1);
  night += albedo * vec3f(1.0, 0.95, 0.85) * spill * spillSh;
  let hn = normalize(l + v);
  night += pow(max(dot(n, hn), 0.0), 30.0) * 0.4 * spill * spillSh * vec3f(1.0, 0.95, 0.85);
  night += albedo * vec3f(1.0, 0.62, 0.4) * rim * 1.2;
  return mix(night, day, dayMix);
}

// ---------------------------------------------------------------- render

fn noDoor() -> DoorFrame {
  var f: DoorFrame;
  f.right = vec3f(1.0, 0.0, 0.0);
  f.fwd = vec3f(0.0, 0.0, 1.0);
  f.origin = vec3f(0.0, -1000.0, 0.0);
  return f;
}

fn render(ro: vec3f, rd: vec3f, pixelAngle: f32) -> vec3f {
  let f = doorFrame();
  let tPortal = portalHit(ro, rd, f);
  let tTerrain = marchTerrain(ro, rd, 0.0, TMAX);
  let tLimit = select(tTerrain, TMAX, tTerrain < 0.0);
  let obj = marchObjects(ro, rd, tLimit, f);

  var t = tTerrain;
  var isObject = false;
  if (obj.t > 0.0 && (tTerrain < 0.0 || obj.t < tTerrain)) {
    t = obj.t;
    isObject = true;
  }
  let throughDoor = tPortal > 0.0 && (t < 0.0 || tPortal < t);

  var color = vec3f(0.0);
  if (throughDoor) {
    // Beyond the opening the same field is seen in daylight, but the portal tilts the view
    // up toward the horizon so sky and hills fill the frame: a window to somewhere else.
    let q = toDoor(ro + rd * tPortal, f);
    let ld = vec3f(dot(rd, f.right), rd.y, dot(rd, f.fwd));
    let ax = vec3f(cos(EXIT_YAW), 0.0, sin(EXIT_YAW));
    let exitFwd = vec3f(-sin(EXIT_YAW), 0.0, cos(EXIT_YAW));
    let ro2 = vec3f(EXIT_POS.x, terrainHeight(EXIT_POS) + 0.02, EXIT_POS.y) + ax * q.x + vec3f(0.0, q.y, 0.0);
    let rd1 = ax * ld.x + vec3f(0.0, ld.y, 0.0) + exitFwd * ld.z;
    let c = cos(PORTAL_PITCH);
    let s = sin(PORTAL_PITCH);
    let rd2 = normalize(rd1 * c - cross(ax, rd1) * s + ax * dot(ax, rd1) * (1.0 - c));
    let t2 = marchTerrain(ro2, rd2, 0.02, TMAX);
    if (t2 < 0.0) {
      color = skyDay(rd2);
    } else {
      let p = ro2 + rd2 * t2;
      let footprint = pixelAngle * (tPortal + t2);
      let n = terrainNormal(p, max(0.08, footprint * 0.5));
      color = shadeGround(p, n, rd2, t2, footprint, noDoor(), 1.0, 0.0);
      let fogAmt = 1.0 - exp(-t2 * 0.0028);
      color = mix(color, skyDay(vec3f(rd2.x, 0.03, rd2.z)), fogAmt * 0.05);
    }
    return color;
  }
  if (t < 0.0) {
    // Sky. The day front reaches the sky through a far point along the ray.
    let far = ro + rd * 120.0;
    let front = dayFront(far, f);
    let dayMix = front.x;
    color = mix(skyNight(rd), skyDay(rd), dayMix);
    // A faint warm dawn glow rides the front, fading quickly away from the horizon.
    color += vec3f(1.0, 0.6, 0.35) * front.y * 0.05 * (1.0 - smoothstep(0.0, 0.25, rd.y));
  } else {
    let p = ro + rd * t;
    let front = dayFront(p, f);
    let dayMix = front.x;
    let rim = front.y;
    let footprint = pixelAngle * t;
    if (isObject) {
      let n = objectNormal(p, f);
      color = shadeObject(p, n, rd, obj.mat, f, dayMix, rim);
    } else {
      let n = terrainNormal(p, max(0.08, footprint * 0.5));
      color = shadeGround(p, n, rd, t, footprint, f, dayMix, rim);
    }
    // Aerial perspective: a touch of sky colour with distance (mostly at night).
    let fogNight = skyNight(vec3f(rd.x, 0.02, rd.z)) * 0.9;
    let fogDay = skyDay(vec3f(rd.x, 0.03, rd.z));
    let fog = mix(fogNight, fogDay, dayMix);
    let fogAmt = 1.0 - exp(-t * 0.0028);
    color = mix(color, fog, fogAmt * mix(0.55, 0.05, dayMix));
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
  let pixelAngle = (2.0 / focal) / res.y;   // radians per pixel (approx)

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
