// Dreamcore: a Bliss-like meadow at night with a door standing in the grass. Through the
// opening there is a sunlit sand dune; as `phase` rises the day pours out of the door and
// sweeps across the hills.
//
// Fullscreen raymarcher: heightfield hills, a relief-mapped blade layer (a tile of thousands
// of bent blades rasterised top-down, then ray marched as a heightfield with Kajiya-Kay fibre
// shading, depth occlusion and light transmittance), an SDF door, a rectangular area light for
// the door spill, single scattering in the night air, and a second heightfield world (the
// dune) behind the portal.
//
// Grass techniques follow Habel et al. (ray-cast grass layers), Kajiya & Kay (fibre shading)
// and Boulanger et al. (lit grass volume with occlusion and shadows through the layer).

struct Params {
  resolution: vec2f,
  time: f32,
  phase: f32,        // 0 = night, 1 = day (the day expands out of the door)
  camera: vec4f,     // height, pitch (rad, + looks up), vertical fov (rad), aa samples
  door: vec4f,       // x, z, yaw (rad), leaf angle (rad)
  look: vec4f,       // sun azimuth (rad), sun elevation (rad), texture strength, door light
  grass: vec4f,      // blade radius (m), blade height (m), blade shadow rays (0/1), wind
  debug: vec4f,      // debug view (0 off, 1 free camera, 2 top-down map, 3 main camera, 4 main camera clean), camera xyz
  dune: vec4f,       // first dune: start (m behind the sill), stoss slope (tan), crest (m), far field level (m)
  dune2: vec4f,      // second dune: gap after the first crest (m), stoss slope (tan), crest (m), crest line skew (tan)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tileColor: texture_2d<f32>;    // rgb sRGB albedo, a height / TILE_HEIGHT
@group(0) @binding(2) var tileTangent: texture_2d<f32>;  // xyz blade tangent * 0.5 + 0.5 (tile space), w along-blade
@group(0) @binding(3) var tileSamp: sampler;

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

// ---------------------------------------------------------------- grass layer

// The grass is a relief-mapped layer: a repeating tile of thousands of bent blades rendered
// top-down (colour + height + tangent) is ray marched as a heightfield above the ground, so
// blades overlap, occlude and bend sideways like a real lawn. The tile is combed along +x and
// the world comb field rotates it locally.
const TILE_SIZE: f32 = 3.0;
const TILE_HEIGHT: f32 = 0.32;

// Local grass coverage: clumpy, only on the flat plain (the hills keep the far-field
// texture), fading with distance from the camera where blades would be sub-pixel anyway.
fn grassCoverage(xz: vec2f) -> f32 {
  let clump = clamp(0.55 + 0.9 * (fbm3(xz * 0.45 + vec2f(9.0, 4.0)) - 0.3), 0.35, 1.2);
  let radius = params.grass.x;
  let inPatch = 1.0 - smoothstep(radius * 0.7, radius * 0.98, length(xz - params.door.xy));
  let near = 1.0 - smoothstep(26.0, 42.0, length(xz));
  return clump * inPatch * near;
}

// Prevailing comb direction of the flattened grass (shared with the far-field streaks).
fn combDir() -> vec2f {
  let a = PI * 0.78;
  return vec2f(sin(a), cos(a));
}

// Slow drift of the lay: a displacement of the tile domain rather than a rotation, so the
// distortion stays the same everywhere instead of growing with the distance to the door.
fn combWarp(xz: vec2f) -> vec2f {
  return 0.6 * vec2f(fbm3(xz * 0.045 + vec2f(3.0, 1.0)) - 0.5, fbm3(xz * 0.045 + vec2f(8.0, 5.0)) - 0.5)
       + 0.08 * vec2f(sin(xz.x * 0.9 + xz.y * 0.4), cos(xz.x * 0.5 - xz.y * 0.7));
}

struct GrassFrame {
  dir: vec2f,      // world direction the tile's +x maps to
  offset: vec2f,   // tile-space displacement of the lay at this ray
  cover: f32,
  scale: f32,      // tile height to scene height
}

fn grassFrame(xz: vec2f) -> GrassFrame {
  var g: GrassFrame;
  g.dir = combDir();
  g.offset = combWarp(xz);
  g.cover = grassCoverage(xz);
  g.scale = params.grass.y / TILE_HEIGHT;
  return g;
}

fn grassUV(xz: vec2f, g: GrassFrame) -> vec2f {
  let rel = xz - params.door.xy;
  let local = vec2f(dot(rel, g.dir), dot(rel, vec2f(-g.dir.y, g.dir.x))) + g.offset;
  return vec2f(local.x / TILE_SIZE, 1.0 - local.y / TILE_SIZE);
}

fn grassHeight(xz: vec2f, g: GrassFrame) -> f32 {
  // Tufts: the sward is taller and shorter in 40 cm patches, so low light picks out relief.
  let tuft = 0.7 + 0.6 * vnoise(xz * 2.4 + vec2f(3.0, 11.0));
  return textureSampleLevel(tileColor, tileSamp, grassUV(xz, g), 0.0).a * TILE_HEIGHT * g.scale * g.cover * tuft;
}

// Blade layer surface height above the local terrain.
fn grassSurface(p: vec3f, g: GrassFrame) -> f32 {
  return p.y - terrainHeight(p.xz) - grassHeight(p.xz, g);
}

// Relief march of the blade heightfield, which rides on the terrain, between tEnter and the
// terrain hit tExit.
fn grassMarch(ro: vec3f, rd: vec3f, tEnter: f32, tExit: f32, g: GrassFrame) -> f32 {
  let steps = select(28, 56, params.grass.z > 0.5);
  let dt = (tExit - tEnter) / f32(steps);
  let jitter = hash13(rd * 977.0) * dt;
  var tPrev = tEnter;
  for (var i = 0; i < steps; i++) {
    let t = tEnter + f32(i) * dt + jitter;
    if (t > tExit) { break; }
    if (grassSurface(ro + rd * t, g) < 0.0) {
      // Bisect between the last sample above the surface and this one.
      var a = tPrev;
      var b = t;
      for (var k = 0; k < 4; k++) {
        let m = 0.5 * (a + b);
        if (grassSurface(ro + rd * m, g) < 0.0) { b = m; } else { a = m; }
      }
      return 0.5 * (a + b);
    }
    tPrev = t;
  }
  return -1.0;
}

// Ray interval for the grass layer: from one layer height above the terrain hit (the
// terrain is locally gentle) down to the hit, restricted to the patch radius.
fn grassInterval(ro: vec3f, rd: vec3f, tHit: f32) -> vec2f {
  if (tHit <= 0.0 || rd.y > -0.02) { return vec2f(1.0, 0.0); }
  let hMax = params.grass.y;
  var t0 = max(tHit - hMax * 1.6 / (-rd.y), 0.0);
  var t1 = tHit;
  let radius = params.grass.x;
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

// Light reaching a point at relative height hf inside the blade layer.
fn grassTransmittance(hf: f32, l: vec3f, cover: f32) -> f32 {
  let depth = pow(1.0 - hf, 1.5);
  return exp(-1.6 * cover * depth / max(l.y, 0.12));
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
  let z = p.y;
  let toe = 1.2;
  // Flat sand level with the field at the threshold, gently undulating further in.
  var h = 0.05 * (fbm3(p * 0.3 + vec2f(7.0, 1.0)) - 0.5) * smoothstep(0.0, 2.0, z);
  // First dune (params.dune): a soft toe `start` metres in, a stoss face of slope `slope` up to
  // a rounded crest near `crest`, then a 31 degree slip face down to a trough behind it.
  let start = params.dune.x + 0.8 * sin(p.x * 0.35 + 1.0) + 0.8 * (fbm3(p * 0.15 + vec2f(2.0, 5.0)) - 0.5);
  let rise = z - start;
  let face = max(rise, 0.0) + toe * log(1.0 + exp(-abs(rise) / toe));
  let face0 = toe * log(1.0 + exp(-start / toe));
  let crest = max(params.dune.z, 0.1);
  let slope = max(params.dune.y, 0.05);
  var h1 = crest * tanh(slope * (face - face0) / crest);
  let zc = start + 1.8 * crest / slope;
  let drop = 0.6 * 0.45 * log(1.0 + exp((z - zc) / 0.45));
  h1 -= min(drop, crest * 0.7);
  h += h1;
  // Second dune (params.dune2): taller, rising from the trough with its crest running at a
  // skew so its face turns partly away from the sand sun.
  let start2 = zc + 1.0 + params.dune2.x;
  let rise2 = z - start2 + params.dune2.w * p.x + 0.6 * (fbm3(p * 0.12 + vec2f(8.0, 3.0)) - 0.5);
  let face2 = max(rise2, 0.0) + toe * log(1.0 + exp(-abs(rise2) / toe));
  let crest2 = max(params.dune2.z, 0.1);
  h += crest2 * tanh(max(params.dune2.y, 0.05) * face2 / crest2);
  // Dune field beyond the second crest: asymmetric dunes (a 10 degree stoss side, a 31 degree
  // slip face) about 34 m apart with wandering crests, sitting at the far field level.
  let far = smoothstep(start2 + 6.0, start2 + 18.0, z);
  let fieldDir = vec2f(0.85, 0.53);
  let fw = 6.0 * (fbm3(p * 0.03 + vec2f(9.0, 2.0)) - 0.5);
  let u = fract((dot(p, fieldDir) + fw) / 34.0 + 0.3);
  let stoss = 0.78;
  let prof = select((1.0 - u) / (1.0 - stoss), u / stoss, u < stoss);
  let fieldAmp = 4.5 * (0.55 + 0.45 * fbm3(p * 0.02 + vec2f(4.0, 6.0)));
  h += far * (fieldAmp * prof - 2.2 + 0.5 * (fbm3(p * 0.14 + vec2f(1.0, 8.0)) - 0.5) + params.dune.w);
  // Small-scale roughness on the slopes only; the flat sand stays smooth.
  h += 0.08 * (fbm3(p * 0.9 + vec2f(3.0, 9.0)) - 0.5) * smoothstep(0.0, 3.0, rise);
  return h;
}

// Large-scale dune slope (dh/dx, dh/dz).
fn duneGradient(p: vec2f) -> vec2f {
  let e = 0.05;
  let hx = duneHeight(p + vec2f(e, 0.0)) - duneHeight(p - vec2f(e, 0.0));
  let hz = duneHeight(p + vec2f(0.0, e)) - duneHeight(p - vec2f(0.0, e));
  return vec2f(hx, hz) / (2.0 * e);
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

const SAND_HORIZON: vec3f = vec3f(0.78, 0.50, 0.30);   // sunset haze at the desert horizon

// Desert sky seen through the opening: warm haze low, blue above, the low sun glowing.
fn sandSky(rd: vec3f) -> vec3f {
  let t = clamp(rd.y, 0.0, 1.0);
  var col = mix(SAND_HORIZON, vec3f(0.30, 0.45, 0.70), pow(t, 0.45));
  let s = max(dot(rd, sandSun()), 0.0);
  col += vec3f(1.0, 0.9, 0.7) * (0.18 * pow(s, 8.0) + 0.5 * pow(s, 200.0));
  return col;
}

fn sandSun() -> vec3f {
  // Low sun (12 degrees) from the left: it rakes across the flat sand, lights the first dune
  // and leaves the skewed face of the second one in half shadow, so the two read apart.
  return normalize(vec3f(-0.9, 0.2, -0.3));
}

fn shadeSand(p: vec3f, rd: vec3f, footprint: f32) -> vec3f {
  // Orange desert sand at sunset, lit sand near sRGB (212,120,40), shadows deep brown.
  var albedo = rgb8(214.0, 142.0, 62.0);
  albedo *= 0.94 + 0.12 * fbm3(p.xz * 0.7 + vec2f(4.0, 2.0));
  let grainFade = 1.0 - smoothstep(0.002, 0.014, footprint);
  // Grains: a fine speckle plus a coarser clumping, both fading before they can alias.
  let grain = (hash13(floor(p * 900.0)) - 0.5) * 0.10 + (vnoise(p.xz * 260.0) - 0.5) * 0.14;
  albedo *= 1.0 + grain * grainFade;

  let sun = sandSun();
  let sunCol = vec3f(1.0, 0.84, 0.62) * 1.5;      // sunset sun
  let amb = vec3f(0.28, 0.15, 0.09) * 1.5;        // bounce off the surrounding sand
  let slope = duneGradient(p.xz);
  let n = normalize(vec3f(-slope.x, 1.0, -slope.y));
  let ndl = max(dot(n, sun), 0.0);
  var col = albedo * (sunCol * ndl + amb * (0.6 + 0.4 * n.y));
  let v = -rd;
  let h = normalize(sun + v);
  // Broad sheen toward the sun, and a few grains catching it.
  col += albedo * pow(max(dot(n, h), 0.0), 8.0) * 0.05 * sunCol;
  let g = vec3f(hash13(p * 431.0), hash13(p * 517.0 + vec3f(3.0)), hash13(p * 619.0 + vec3f(7.0))) - 0.5;
  let gn = normalize(n + g * 0.6 * grainFade);
  let glint = pow(max(dot(gn, h), 0.0), 160.0) * step(0.965, hash13(floor(p * 900.0))) * grainFade;
  col += sunCol * glint * 0.5;
  return col;
}

fn renderSand(ro: vec3f, rd: vec3f, pixelAngle: f32, tBase: f32) -> vec3f {
  let t = marchDune(ro, rd);
  if (t < 0.0) {
    return select(SAND_HORIZON, sandSky(rd), rd.y > 0.0);
  }
  let p = ro + rd * t;
  let col = shadeSand(p, rd, pixelAngle * (tBase + t));
  return mix(col, SAND_HORIZON, 1.0 - exp(-t * 0.013));
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
    let facing = pow(max(dot(f.fwd, l), 0.0), 1.6);
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
      vis = doorShadow(p + n * 0.002, l, f, d - 0.03, 40.0);
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
      let facing = pow(max(dot(f.fwd, l), 0.0), 1.6);
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
  sh *= grassTransmittance(0.0, sun, cover);
  let hemi = 0.5 + 0.5 * n.y;
  var day = albedo * (sunCol * ndl * sh + AMBIENT * hemi * ao);
  let h = normalize(sun + v);
  day += albedo * pow(max(dot(n, h), 0.0), 5.0) * 0.10 * sh * sunCol;

  // --- Night: moon fill + the door spill.
  var night = nightBase(albedo, n) * ao;
  if (dayMix < 0.999) {
    night += mix(albedo, vec3f(dot(albedo, vec3f(0.33))), 0.2) * doorLight(p, n, vec3f(0.0), 0.0, f, true) * grassTransmittance(0.0, normalize(f.origin + vec3f(0.0, 1.0, 0.0) - p), cover);
  }
  return mix(night, day, dayMix) + albedo * rimGlow(n, rim);
}

// Blades shadowing blades: walk toward the light through the heightfield. The plain is
// flat under the patch, so only the blade height changes along the short walk.
fn grassSelfShadow(p: vec3f, l: vec3f, g: GrassFrame, base: f32, seed: f32) -> f32 {
  let dt = 0.045;
  var s = 1.0;
  for (var i = 0; i < 8; i++) {
    let q = p + l * ((f32(i) + 0.35 + 0.65 * seed) * dt);
    let surf = base + grassHeight(q.xz, g);
    s = min(s, clamp((q.y - surf) / 0.025 + 0.5, 0.0, 1.0));
    if (s <= 0.0) { break; }
  }
  return s;
}

fn shadeGrass(p: vec3f, rd: vec3f, g: GrassFrame, f: DoorFrame, dayMix: f32, rim: f32) -> vec3f {
  let uv = grassUV(p.xz, g);
  let colH = textureSampleLevel(tileColor, tileSamp, uv, 0.0);
  let tanS = textureSampleLevel(tileTangent, tileSamp, uv, 0.0);
  let base = terrainHeight(p.xz);
  let hf = clamp((p.y - base) / max(params.grass.y, 0.01), 0.0, 1.0);
  let seed = hash12(p.xz * 311.0 + vec2f(p.y * 57.0));
  var albedo = srgb2lin(colH.rgb);
  // Tile tangent lives in comb space (+x = comb direction); rotate it into the world.
  let tt = tanS.xyz * 2.0 - vec3f(1.0);
  let perp = vec2f(-g.dir.y, g.dir.x);
  let tangent = normalize(vec3f(tt.x * g.dir.x + tt.z * perp.x, tt.y, tt.x * g.dir.y + tt.z * perp.y));
  // Heightfield normal from the tile, softened: blades are thin so raw gradients are spiky.
  let e = 0.006;
  let hx = grassHeight(p.xz + vec2f(e, 0.0), g) - grassHeight(p.xz - vec2f(e, 0.0), g);
  let hz = grassHeight(p.xz + vec2f(0.0, e), g) - grassHeight(p.xz - vec2f(0.0, e), g);
  var n = normalize(vec3f(-hx * 0.7, 2.0 * e, -hz * 0.7));
  let v = -rd;
  let sun = sunDir();
  let sunCol = sunColor();
  let ao = 0.3 + 0.7 * hf;
  let sh = doorShadow(p + vec3f(0.0, 0.02, 0.0), sun, f, 8.0, 5.0) * terrainShadow(p, sun)
    * mix(1.0, grassSelfShadow(p, sun, g, base, seed), 0.6);
  let tlSun = dot(tangent, sun);
  let kkSun = sqrt(max(1.0 - tlSun * tlSun, 0.0));
  let lambertSun = max(dot(n, sun), 0.0);
  let trSun = grassTransmittance(hf, sun, g.cover);
  var day = albedo * (sunCol * mix(lambertSun, kkSun, 0.45) * trSun * sh + AMBIENT * (0.5 + 0.5 * n.y) * ao);
  let hSun = normalize(sun + v);
  let thSun = dot(tangent, hSun);
  day += sunCol * albedo * pow(sqrt(max(1.0 - thSun * thSun, 0.0)), 22.0) * 0.18 * trSun * sh;

  let lumA = dot(albedo, vec3f(0.2126, 0.7152, 0.0722));
  let nightAlbedo = mix(albedo, lumA * vec3f(0.65, 0.85, 1.0), 0.4);
  let moon = moonDir();
  let tlMoon = dot(tangent, moon);
  let kkMoon = sqrt(max(1.0 - tlMoon * tlMoon, 0.0));
  var night = nightAlbedo * (vec3f(0.28, 0.4, 0.68) * 0.16 * mix(max(dot(n, moon), 0.0), kkMoon, 0.5) * grassTransmittance(hf, moon, g.cover) + vec3f(0.006, 0.01, 0.026) * ao);
  if (dayMix < 0.999) {
    let doorCenter = f.origin + vec3f(0.0, DOOR_H * 0.5, 0.0);
    let toDoorC = doorCenter - p;
    let sMid = f.origin + f.right * DOOR_SAMPLES[1].x + vec3f(0.0, DOOR_SAMPLES[1].y, 0.0);
    let doorSh = doorShadow(p + vec3f(0.0, 0.02, 0.0), normalize(toDoorC), f, length(toDoorC) - 0.03, 40.0)
      * grassSelfShadow(p, normalize(sMid - p), g, base, seed);
    var e2 = 0.0;
    var spec = 0.0;
    for (var j = 0; j < 3; j++) {
      let o = DOOR_SAMPLES[j];
      let s = f.origin + f.right * o.x + vec3f(0.0, o.y, 0.0);
      let toL = s - p;
      let d = max(length(toL), 0.05);
      let l = toL / d;
      let facing = pow(max(dot(f.fwd, l), 0.0), 1.6);
      let geom = facing / (d * d + 0.6);
      let tl = dot(tangent, l);
      let kk = sqrt(max(1.0 - tl * tl, 0.0));
      let lam = max(dot(n, l), 0.0) + 0.05;
      let tr = grassTransmittance(hf, l, g.cover);
      e2 += geom * mix(lam, kk, 0.3) * tr;
      let hl = normalize(l + v);
      let th = dot(tangent, hl);
      spec += geom * pow(sqrt(max(1.0 - th * th, 0.0)), 16.0) * tr;
    }
    let front = dot(p - f.origin, f.fwd);
    let inFront = select(0.0, 1.0, front < -0.01);
    night += albedo * DOOR_COLOR * params.look.w * (e2 * 0.333 + spec * 0.1) * doorSh * inFront;
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
  var kind = 0;   // 0 sky, 1 terrain, 2 door, 3 blade layer
  if (t > 0.0) { kind = 1; }
  if (tDoor > 0.0 && (t < 0.0 || tDoor < t)) { t = tDoor; kind = 2; }
  var throughDoor = tPortal > 0.0 && (t < 0.0 || tPortal < t);

  // The blade layer rides on the terrain: march it in front of whatever the ray reached
  // first, so blades also overlap the sill, the leaf and the bottom of the opening.
  var g: GrassFrame;
  var iv = grassInterval(ro, rd, tTerrain);
  let tBlock = select(t, tPortal, throughDoor);
  if (tBlock > 0.0) { iv.y = min(iv.y, tBlock); }
  if (iv.y > iv.x) {
    g = grassFrame((ro + rd * iv.y).xz);
    if (g.cover > 0.01) {
      let tg = grassMarch(ro, rd, iv.x, iv.y, g);
      if (tg > 0.0) { t = tg; kind = 3; throughDoor = false; }
    }
  }

  var color = vec3f(0.0);
  if (throughDoor) {
    // The opening is a window onto the dune: continue the ray in door-local space.
    let q = toDoor(ro + rd * tPortal, f);
    let ld = vec3f(dot(rd, f.right), rd.y, dot(rd, f.fwd));
    return renderSand(vec3f(q.x, q.y, 0.0), normalize(ld), pixelAngle, tPortal);
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
    } else if (kind == 3) {
      color = shadeGrass(p, rd, g, f, dayMix, rim);
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
  }
  if (dayMix < 0.999) {
    color += doorScatter(ro, rd, t, f) * (1.0 - dayMix);
  }
  return color;
}

// ---------------------------------------------------------------- debug views

// The sand world lives in door-local space: x along the frame, z away from the door, the
// threshold at the origin. These views show that space whole.
fn identityFrame() -> DoorFrame {
  var f: DoorFrame;
  f.right = vec3f(1.0, 0.0, 0.0);
  f.fwd = vec3f(0.0, 0.0, 1.0);
  f.origin = vec3f(0.0);
  return f;
}

// Is a point of the sand world seen by the real camera through the opening?
fn portalVisible(p: vec3f) -> f32 {
  let cam = toDoor(vec3f(0.0, params.camera.x, 0.0), doorFrame());
  let dz = p.z - cam.z;
  if (dz <= 1e-4) { return 0.0; }
  let s = -cam.z / dz;
  if (s < 0.0 || s > 1.0) { return 0.0; }
  let q = cam + (p - cam) * s;
  if (abs(q.x) >= DOOR_W * 0.5 || q.y <= 0.0 || q.y >= DOOR_H) { return 0.0; }
  // Hidden behind a nearer part of the dune?
  for (var i = 1; i < 24; i++) {
    let m = q + (p - q) * (f32(i) / 24.0);
    if (m.y < duneHeight(m.xz) - 0.01) { return 0.0; }
  }
  return 1.0;
}

fn debugGrid(xz: vec2f, footprint: f32) -> f32 {
  let w = max(footprint * 1.5, 0.004);
  let gx = smoothstep(w, 0.0, abs(fract(xz.x + 0.5) - 0.5));
  let gz = smoothstep(w, 0.0, abs(fract(xz.y + 0.5) - 0.5));
  return max(gx, gz);
}

// Free camera in the sand world: the door as solid geometry on the flat toe, the slip face
// behind it, a 1 m grid, the door plane in blue and the part seen through the opening in green.
// overlays: 0 bare sand, 1 the door and the part seen through it, 2 also the grid and door plane.
fn renderDebugWorld(ro: vec3f, rd: vec3f, pixelAngle: f32, overlays: i32) -> vec3f {
  let f = identityFrame();
  let tDune = marchDune(ro, rd);
  var tDoor = -1.0;
  if (overlays > 0) { tDoor = marchDoor(ro, rd, select(tDune, 200.0, tDune < 0.0), f); }
  if (tDoor > 0.0) {
    let p = ro + rd * tDoor;
    let n = doorNormal(p, f);
    return rgb8(86.0, 94.0, 120.0) * (0.35 + 0.65 * max(dot(n, sandSun()), 0.0));
  }
  if (tDune < 0.0) { return select(SAND_HORIZON, sandSky(rd), rd.y > 0.0); }
  let p = ro + rd * tDune;
  let footprint = pixelAngle * tDune;
  var col = shadeSand(p, rd, footprint);
  if (overlays > 1) {
    col = mix(col, vec3f(0.05, 0.05, 0.08), debugGrid(p.xz, footprint) * 0.6);
    col = mix(col, vec3f(0.1, 0.3, 1.0), smoothstep(max(footprint * 2.0, 0.006), 0.0, abs(p.z)) * 0.9);
  }
  if (overlays > 0) {
    col = mix(col, vec3f(0.15, 0.9, 0.25), 0.4 * portalVisible(p));
  }
  return mix(col, SAND_HORIZON, 1.0 - exp(-tDune * 0.013));
}

// Top-down map of the sand world: x in [-6, 6], z in [-2, 22] (camera side at the bottom),
// shaded from above with 25 cm contours, the door cut at 1 m height and the same overlays.
fn renderDebugMap(uv: vec2f) -> vec3f {
  let x = mix(-6.0, 6.0, uv.x);
  let z = mix(22.0, -2.0, uv.y);
  let footprint = 12.0 / params.resolution.x;
  let p2 = vec2f(x, z);
  let h = duneHeight(p2);
  let p = vec3f(x, h, z);
  var col = shadeSand(p, vec3f(0.0, -1.0, 0.0), footprint);
  let gm = max(length(duneGradient(p2)), 1e-3);
  let dist = abs(fract(h / 0.25 + 0.5) - 0.5) * 0.25 / gm;
  col = mix(col, vec3f(0.35, 0.15, 0.05), smoothstep(footprint * 2.0, 0.0, dist) * 0.75);
  col = mix(col, vec3f(0.05, 0.05, 0.08), debugGrid(p2, footprint) * 0.35);
  col = mix(col, vec3f(0.15, 0.9, 0.25), 0.45 * portalVisible(p));
  col = mix(col, vec3f(0.1, 0.3, 1.0), smoothstep(footprint * 2.0, 0.0, abs(z)) * 0.9);
  let dd = sdDoor(vec3f(x, 1.0, z), identityFrame());
  col = mix(col, vec3f(0.05, 0.08, 0.3), smoothstep(footprint, 0.0, dd));
  return col;
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let res = params.resolution;
  if (params.debug.x > 1.5 && params.debug.x < 2.5) {
    return vec4f(renderDebugMap(uv), 1.0);
  }
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
    if (params.debug.x > 2.5) {
      // The main camera carried into the sand world: same place, pitch and lens, so the
      // composition of the sand behind the door can be read as the door frames it.
      let fD = doorFrame();
      let camD = toDoor(ro, fD);
      let fwdW = vec3f(dot(forward, fD.right), forward.y, dot(forward, fD.fwd));
      let upW = vec3f(dot(up, fD.right), up.y, dot(up, fD.fwd));
      let rightW = vec3f(dot(right, fD.right), right.y, dot(right, fD.fwd));
      let rdM = normalize(fwdW * focal + rightW * sx + upW * sy);
      acc += renderDebugWorld(camD, rdM, pixelAngle, select(1, 0, params.debug.x > 3.5));
      continue;
    }
    if (params.debug.x > 0.5) {
      // Free camera in the sand world, looking at the foot of the slip face.
      let cam = params.debug.yzw;
      let fwdD = normalize(vec3f(0.0, 1.2, 7.0) - cam);
      let rightD = normalize(cross(vec3f(0.0, 1.0, 0.0), fwdD));
      let upD = cross(fwdD, rightD);
      let focalD = 1.0 / tan(0.45);
      let rdD = normalize(fwdD * focalD + rightD * sx + upD * sy);
      acc += renderDebugWorld(cam, rdD, (2.0 / focalD) / res.y, 2);
      continue;
    }
    let rd = normalize(forward * focal + right * sx + up * sy);
    acc += render(ro, rd, pixelAngle);
  }
  return vec4f(acc / f32(count), 1.0);
}
