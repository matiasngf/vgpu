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
  plain: vec4f,      // flat sand: where it starts falling away (m behind the sill), fall (tan), unused, hollow depth (m, negative)
  dune: vec4f,       // the dune: start (m behind the sill), stoss slope (tan), crest (m), crest line skew (tan)
  sand: vec4f,       // wind ripples on the flat sand: amplitude (m), wavelength (m), fade distance from the camera (m), crest position (0..1 of the period)
  blades: vec4f,     // sub-pixel sample offset (x, y in 0..1), geometric blade zone radius (m, 0 = none), sample weight
  shadow: vec4f,     // blade shadow map: size (px, 0 = none), light height above the sill (m), depth bias (m), unused
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tileColor: texture_2d<f32>;    // rgb sRGB albedo, a height / TILE_HEIGHT
@group(0) @binding(2) var tileTangent: texture_2d<f32>;  // xyz blade tangent * 0.5 + 0.5 (tile space), w along-blade
@group(0) @binding(3) var tileSamp: sampler;
@group(0) @binding(4) var bladeDist: texture_2d<f32>;    // geometric blades: distance along the ray (0 = none), tangent
@group(0) @binding(5) var bladeColor: texture_2d<f32>;   // geometric blades: linear albedo, height above the terrain (m)
@group(0) @binding(6) var bladeShadow: texture_2d<f32>;  // the blades seen from the door light: distance from it (0 = none)

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
  // Leaf hinged on the right jamb, swung toward the viewer by params.door.w. The hinge axis
  // sits at the front inner corner of the jamb, like a real door: closed, the leaf lies flush
  // with the front face inside the opening; open, it stays on the jamb instead of floating off
  // its outer edge.
  let a = params.door.w;
  let u = vec3f(-cos(a), 0.0, -sin(a));
  let n = vec3f(-u.z, 0.0, u.x);
  let hinge = vec3f(hw - 0.006, 0.0, -FRAME_D + LEAF_D);
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

// Inside the geometric blade zone the relief steps aside for the real blades (their carpet
// stays: the tile colour at ground level); the zone radius is 0 when there are no blades.
fn reliefLod(xz: vec2f) -> f32 {
  let zone = params.blades.z;
  return smoothstep(zone - 2.0, zone + 2.0, length(xz));
}

fn grassHeight(xz: vec2f, g: GrassFrame) -> f32 {
  // Tufts: the sward is taller and shorter in 40 cm patches, so low light picks out relief.
  let tuft = 0.7 + 0.6 * vnoise(xz * 2.4 + vec2f(3.0, 11.0));
  return textureSampleLevel(tileColor, tileSamp, grassUV(xz, g), 0.0).a * TILE_HEIGHT * g.scale * g.cover * tuft * reliefLod(xz);
}

// Blade layer surface height above the local terrain.
fn grassSurface(p: vec3f, g: GrassFrame) -> f32 {
  return p.y - terrainHeight(p.xz) - grassHeight(p.xz, g);
}

// Relief march of the blade heightfield, which rides on the terrain, between tEnter and the
// terrain hit tExit.
fn grassMarch(ro: vec3f, rd: vec3f, tEnter: f32, tExit: f32, g: GrassFrame) -> f32 {
  let steps = select(28, 96, params.grass.z > 0.5);
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
const DUNE_TOE: f32 = 1.2;

// Rise coordinate of the dune: positive on its face and beyond. The toe line runs at a skew
// (params.dune.w): positive brings the dune closer on the +x side, so its face turns away
// from a sun on that side and sits backlit, its shadow falling across the sand in front.
fn duneRise(p: vec2f) -> f32 {
  return p.y - params.dune.x + params.dune.w * p.x + 1.2 * (fbm3(p * 0.08 + vec2f(8.0, 3.0)) - 0.5);
}

fn duneHeight(p: vec2f) -> f32 {
  let z = p.y;
  let toe = DUNE_TOE;
  // Flat sand level with the field at the threshold, gently undulating further in.
  var h = 0.10 * (fbm3(p * 0.3 + vec2f(7.0, 1.0)) - 0.5) * smoothstep(0.0, 2.0, z);
  // From params.plain.x on the sand falls away at params.plain.y and settles into a hollow
  // params.plain.w below the sill, so the dunes beyond stand up out of it at a distance.
  let t = z - params.plain.x;
  let ramp = max(t, 0.0) + toe * log(1.0 + exp(-abs(t) / toe)) - toe * log(1.0 + exp(-params.plain.x / toe));
  let depth = max(-params.plain.w, 0.1);
  h -= depth * tanh(ramp * params.plain.y / depth);
  // The near dune (params.dune): a soft toe `start` metres in, its toe line skewed so it runs in
  // diagonally from the right, a face of slope `slope` up to a rounded crest `crest` above the
  // hollow, then down its back.
  let rise = duneRise(p);
  let face = max(rise, 0.0) + toe * log(1.0 + exp(-abs(rise) / toe));
  let crest = max(params.dune.z, 0.1);
  let slope = max(params.dune.y, 0.05);
  let back = 1.5 * crest / slope;
  h += crest * tanh(slope * face / crest) * (1.0 - smoothstep(back, back + 30.0, rise));
  // A tall dune off to the right, outside the door's view, standing against the sun: the band of
  // shadow that crosses the hollow is its.
  let side = p.x - 10.0 + 0.15 * (z - 20.0) + 2.0 * (fbm3(p * 0.05 + vec2f(2.0, 9.0)) - 0.5);
  let sideFace = max(side, 0.0) + toe * log(1.0 + exp(-abs(side) / toe));
  h += 12.0 * tanh(sideFace / 12.0) * smoothstep(16.0, 26.0, z) * (1.0 - smoothstep(60.0, 110.0, z));
  // A big dune far behind the door, its face toward us, so the opening looks at dunes all the
  // way up instead of sky.
  let farRise = z - 100.0 + 0.25 * p.x + 6.0 * (fbm3(p * 0.012 + vec2f(5.0, 1.0)) - 0.5);
  let farFace = max(farRise, 0.0) + toe * log(1.0 + exp(-abs(farRise) / toe));
  h += 16.0 * tanh(0.5 * farFace / 16.0) * (1.0 - smoothstep(60.0, 120.0, farRise));
  // Dune field beyond: asymmetric dunes (a 10 degree stoss side, a 30 degree slip face) about
  // 34 m apart with wandering crests, and a second family three times bigger further out, in
  // the same proportions so no face gets steeper than sand allows.
  let far = smoothstep(params.dune.x + 8.0, params.dune.x + 30.0, z);
  let fieldDir = vec2f(0.85, 0.53);
  let fw = 6.0 * (fbm3(p * 0.03 + vec2f(9.0, 2.0)) - 0.5);
  let u = fract((dot(p, fieldDir) + fw) / 34.0 + 0.3);
  let stoss = 0.78;
  let prof = select((1.0 - u) / (1.0 - stoss), u / stoss, u < stoss);
  let fieldAmp = 4.5 * (0.55 + 0.45 * fbm3(p * 0.02 + vec2f(4.0, 6.0)));
  h += far * (fieldAmp * prof + 0.5 * (fbm3(p * 0.14 + vec2f(1.0, 8.0)) - 0.5));
  let far2 = smoothstep(90.0, 140.0, z);
  let dir2 = vec2f(0.6, 0.8);
  let fw2 = 15.0 * (fbm3(p * 0.01 + vec2f(3.0, 5.0)) - 0.5);
  let u2 = fract((dot(p, dir2) + fw2) / 110.0 + 0.6);
  let prof2 = select((1.0 - u2) / (1.0 - stoss), u2 / stoss, u2 < stoss);
  h += far2 * 14.0 * (0.6 + 0.4 * fbm3(p * 0.008 + vec2f(7.0, 2.0))) * prof2;
  // Sweeping undulations and small roughness on the near dune's slope only; the flat sand stays smooth.
  let onSlope = smoothstep(0.0, 3.0, rise) * (1.0 - smoothstep(back, back + 30.0, rise));
  h += 0.18 * (fbm3(p * 0.22 + vec2f(4.0, 7.0)) - 0.5) * onSlope;
  h += 0.04 * (fbm3(p * 0.9 + vec2f(3.0, 9.0)) - 0.5) * onSlope;
  return h;
}

// Large-scale dune slope (dh/dx, dh/dz).
fn duneGradient(p: vec2f) -> vec2f {
  let e = 0.05;
  let hx = duneHeight(p + vec2f(e, 0.0)) - duneHeight(p - vec2f(e, 0.0));
  let hz = duneHeight(p + vec2f(0.0, e)) - duneHeight(p - vec2f(0.0, e));
  return vec2f(hx, hz) / (2.0 * e);
}

// Wind ripples on the flat sand, from the reference: two sets of crests a few degrees apart
// take turns across the sand, so where they hand over the lines split and join like a
// fingerprint. Each ripple has a long gentle side toward the sun and a short steep side toward
// the door, so the low sun lights the crest tops and drops the steep sides and the troughs
// behind them into shadow. They fade with distance until the sand reads smooth, and never
// climb the dune faces.
const RIPPLE_DIR_A: vec2f = vec2f(0.20, 0.98);   // across the crests, roughly away from the door
const RIPPLE_DIR_B: vec2f = vec2f(0.36, 0.93);   // the second set, 9 degrees off
const RIPPLE_LEN_B: f32 = 0.86;                  // its wavelength relative to params.sand.y
const RIPPLE_PHASE_B: f32 = 1.7;

struct RippleField {
  warp: f32,
  ampA: f32,
  ampB: f32,
  amp: f32,     // ampA + ampB
}

struct Ripples {
  warp: f32,
  ampA: f32,
  ampB: f32,
  amp: f32,     // ampA + ampB
  h: f32,       // height above the trough line
  grad: vec2f,
}

// Crests wander over a few metres and wobble over a few centimetres (the beaded crest lines).
fn rippleWarp(p: vec2f) -> f32 {
  return 22.0 * (fbm3(p * 0.3 + vec2f(1.0, 4.0)) - 0.5) + 14.0 * (fbm3(p * 1.2 + vec2f(6.0, 2.0)) - 0.5)
       + 3.0 * (fbm3(p * 3.0 + vec2f(9.0, 6.0)) - 0.5)
       + 0.8 * (vnoise(p * 7.0 + vec2f(2.0, 5.0)) - 0.5) + 0.5 * (vnoise(p * 21.0 + vec2f(8.0, 1.0)) - 0.5);
}

// Ripple cross-section over one period of phase u: rounded troughs and a sharp crest at
// params.sand.w of the period (so the short steep side faces the door), heights in [0,1].
// Each flank is a quarter cosine, flat in the trough and still climbing at the crest, so the
// two flanks meet in a ridge the way avalanching sand does. Returns (height, dheight/du).
fn rippleShape(u: f32) -> vec2f {
  let a = clamp(params.sand.w, 0.1, 0.9);
  let v = fract(u / (2.0 * PI));
  if (v < a) {
    let s = v / a;
    return vec2f(1.0 - cos(0.5 * PI * s), 0.5 * PI * sin(0.5 * PI * s) / (a * 2.0 * PI));
  }
  let s = (1.0 - v) / (1.0 - a);
  return vec2f(1.0 - cos(0.5 * PI * s), -0.5 * PI * sin(0.5 * PI * s) / ((1.0 - a) * 2.0 * PI));
}

// Camera position in door-local space (the sand world), for distance fades.
fn sandCamera() -> vec3f {
  return toDoor(vec3f(0.0, params.camera.x, 0.0), doorFrame());
}

fn rippleHeightAt(q: vec2f, warp: f32, ampA: f32, ampB: f32) -> f32 {
  let len = max(params.sand.y, 0.02);
  let kA = 2.0 * PI / len;
  let kB = 2.0 * PI / (len * RIPPLE_LEN_B);
  return ampA * rippleShape(dot(q, RIPPLE_DIR_A) * kA + warp).x
       + ampB * rippleShape(dot(q, RIPPLE_DIR_B) * kB + warp * 0.8 + RIPPLE_PHASE_B).x;
}

// The ripple field at p: how strong the ripples are there (patchy, on the flat sand only, gone
// by the dune's toe, fading with the distance to the camera), which set of crests, and the
// crest-line warp. Shared by the surface the rays hit and by the shading, so both agree.
fn rippleField(p: vec2f) -> RippleField {
  var f: RippleField;
  f.warp = rippleWarp(p);
  let onPlain = 1.0 - smoothstep(-1.5, 0.5, duneRise(p));
  let dist = length(p - sandCamera().xz);
  let near = 1.0 - smoothstep(params.sand.z * 0.5, params.sand.z, dist);
  let patchy = smoothstep(0.3, 0.7, fbm3(p * 1.1 + vec2f(3.0, 7.0)));
  f.amp = params.sand.x * (0.7 + 0.3 * patchy) * onPlain * near;
  // Which set of crests: patches about a metre across hand over from one set to the other, so
  // crest lines split and rejoin every few wavelengths; the handovers widen further in.
  let width = mix(0.12, 0.22, smoothstep(2.0, 10.0, p.y));
  let m = smoothstep(0.5 - width, 0.5 + width, fbm3(p * 1.0 + vec2f(5.0, 9.0)));
  f.ampA = f.amp * (1.0 - m);
  f.ampB = f.amp * m;
  return f;
}

fn ripples(p: vec2f, footprint: f32) -> Ripples {
  var r: Ripples;
  let len = max(params.sand.y, 0.02);
  let kA = 2.0 * PI / len;
  let kB = 2.0 * PI / (len * RIPPLE_LEN_B);
  let f = rippleField(p);
  // Ripples much finer than a pixel shade as smooth sand.
  let aa = 1.0 - smoothstep(len * 0.08, len * 0.35, footprint);
  r.warp = f.warp;
  r.amp = f.amp * aa;
  r.ampA = f.ampA * aa;
  r.ampB = f.ampB * aa;
  let sA = rippleShape(dot(p, RIPPLE_DIR_A) * kA + r.warp);
  let sB = rippleShape(dot(p, RIPPLE_DIR_B) * kB + r.warp * 0.8 + RIPPLE_PHASE_B);
  r.h = r.ampA * sA.x + r.ampB * sB.x;
  let e = 0.004;
  let dw = vec2f(rippleWarp(p + vec2f(e, 0.0)) - r.warp, rippleWarp(p + vec2f(0.0, e)) - r.warp) / e;
  r.grad = r.ampA * sA.y * (RIPPLE_DIR_A * kA + dw) + r.ampB * sB.y * (RIPPLE_DIR_B * kB + dw * 0.8);
  return r;
}

// Cast shadow of the ripples: walk toward the sun over one wavelength through the local
// ripple field, riding on the local slope of the sand.
fn rippleShadow(p: vec2f, r: Ripples, sun: vec3f, slope: vec2f) -> f32 {
  if (r.amp < 1e-5) { return 1.0; }
  let len = max(params.sand.y, 0.02);
  let dt = len * 0.08;
  var s = 1.0;
  for (var i = 1; i <= 12; i++) {
    let t = f32(i) * dt;
    let q = p + sun.xz * t;
    let hq = rippleHeightAt(q, r.warp, r.ampA, r.ampB);
    let ray = r.h + (sun.y - dot(slope, sun.xz)) * t;
    s = min(s, clamp((ray - hq) / (0.35 * r.amp) + 0.5, 0.0, 1.0));
  }
  return s;
}

// Soft shadow of the dunes themselves: march toward the sun over the heightfield.
fn duneShadow(p: vec3f, sun: vec3f) -> f32 {
  var s = 1.0;
  var t = 0.4;
  for (var i = 0; i < 40; i++) {
    let q = p + sun * t;
    let d = q.y - duneHeight(q.xz);
    s = min(s, clamp(32.0 * d / t, 0.0, 1.0));   // a penumbra of about 2 degrees
    if (s < 0.01 || t > 90.0) { break; }
    t += max(0.25, t * 0.15);
  }
  return s;
}

// How much of the ripple height the sand surface actually rises: the shading normal carries the
// full profile (that is what draws the crest lines), the relief a fraction of it, so from the
// low camera the crests stand up and shift with parallax without hiding the lit sand behind
// them; at full height every crest would occlude the whole period behind it.
const RIPPLE_RELIEF: f32 = 0.35;

// The sand surface the rays actually hit: the dunes with the wind ripples as real relief, so
// near the door the crests stand up and break the silhouette instead of being painted onto a
// flat surface. The relief is the same field the shading normal uses, so the two agree
// everywhere and the relief simply flattens out with the ripples' own distance fade: there is
// no boundary between "displaced" and "flat" sand to hide.
fn sandHeight(p: vec2f) -> f32 {
  let h = duneHeight(p);
  if (length(p - sandCamera().xz) > params.sand.z) { return h; }
  let f = rippleField(p);
  if (f.amp < 1e-5) { return h; }
  return h + RIPPLE_RELIEF * rippleHeightAt(p, f.warp, f.ampA, f.ampB);
}

fn marchDune(ro: vec3f, rd: vec3f) -> f32 {
  var t = 0.02;
  var lastT = 0.0;
  var hit = -1.0;
  // Enough steps for rays that skim along the far dunes: running out early would leave
  // holes of sky along their crests.
  for (var i = 0; i < 480; i++) {
    let p = ro + rd * t;
    let d = p.y - sandHeight(p.xz);
    // Sand within about a pixel of the ray counts as hit, so distant crests keep clean
    // silhouettes instead of sawing between the samples.
    if (d < 0.0005 * t) { hit = t; break; }
    if (t > 260.0) { break; }
    lastT = t;
    t += clamp(d * 0.4, 0.01 + 0.006 * t, 4.0);
  }
  if (hit < 0.0) { return -1.0; }
  var a = lastT;
  var b = hit;
  for (var i = 0; i < 6; i++) {
    let m = 0.5 * (a + b);
    let p = ro + rd * m;
    if (p.y - sandHeight(p.xz) < 0.0005 * m) { b = m; } else { a = m; }
  }
  return 0.5 * (a + b);
}

const SAND_HORIZON: vec3f = vec3f(0.78, 0.60, 0.34);   // golden haze at the desert horizon

// Desert sky seen through the opening: warm haze low, blue above, the low sun glowing.
fn sandSky(rd: vec3f) -> vec3f {
  let t = clamp(rd.y, 0.0, 1.0);
  var col = mix(SAND_HORIZON, vec3f(0.62, 0.72, 0.80), pow(t, 0.5));
  let s = max(dot(rd, sandSun()), 0.0);
  col += vec3f(1.0, 0.9, 0.7) * (0.35 * pow(s, 6.0) + 1.5 * pow(s, 300.0));
  return col;
}

fn sandSun() -> vec3f {
  // Low sun (about 24 degrees) from the right and behind the dunes: the near faces sit in
  // shadow, their shadows fall across the flat sand, and the ripples light up on their far sides.
  return normalize(vec3f(0.85, 0.44, 0.45));
}

// Oren-Nayar rough diffuse: sand grains scatter back toward the light, so a rough surface
// stays flatter-lit near the terminator and brightens at grazing angles instead of falling off
// like Lambert. Returns the factor that multiplies albedo * light.
fn orenNayar(n: vec3f, l: vec3f, v: vec3f, sigma: f32) -> f32 {
  let ndl = max(dot(n, l), 0.0);
  let ndv = max(dot(n, v), 1e-3);
  let s2 = sigma * sigma;
  let a = 1.0 - 0.5 * s2 / (s2 + 0.33);
  let b = 0.45 * s2 / (s2 + 0.09);
  let s = dot(l, v) - ndl * ndv;
  let t = select(1.0, max(ndl, ndv), s > 0.0);
  return ndl * (a + b * s / t);
}

// GGX specular lobe with Schlick Fresnel and Smith-Schlick masking: the glassy quartz
// grains give sand a soft sheen that grows toward grazing angles and low sun.
fn ggxSheen(n: vec3f, l: vec3f, v: vec3f, roughness: f32, f0: f32) -> f32 {
  let hv = normalize(l + v);
  let ndh = max(dot(n, hv), 0.0);
  let ndv = max(dot(n, v), 1e-3);
  let ndl = max(dot(n, l), 0.0);
  let vdh = max(dot(v, hv), 0.0);
  let alpha = roughness * roughness;
  let a2 = alpha * alpha;
  let dd = ndh * ndh * (a2 - 1.0) + 1.0;
  let d = a2 / (PI * dd * dd);
  let k = alpha * 0.5;
  let g = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k));
  let f = f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
  return d * g * f / (4.0 * ndv);
}

// The desert is lit for its own daylight; through the night's exposure it would clip to a flat
// yellow, so its light is scaled down before the shared tone curve.
const SAND_EXPOSURE: f32 = 1.3;

fn shadeSand(p: vec3f, rd: vec3f, footprint: f32) -> vec3f {
  // Golden sand at low sun, from the reference: lit smooth sand near sRGB (200,136,72), the
  // shadowed dune faces near (83,68,52) under a pale sky, ripple crests flaring to (191,128,70)
  // with black troughs.
  var albedo = rgb8(205.0, 140.0, 72.0);
  albedo *= 0.93 + 0.14 * fbm3(p.xz * 0.7 + vec2f(4.0, 2.0));
  let sun = sandSun();
  let sunCol = vec3f(1.0, 0.86, 0.66) * 1.9;
  let amb = vec3f(0.14, 0.19, 0.34) * 0.75;           // pale sky in the shadows
  let slope = duneGradient(p.xz);
  let r = ripples(p.xz, footprint);
  // Grains: speckle about two pixels wide wherever the sand is, so the crest lines read grainy
  // near the door and the texture melts into smooth sand further in instead of aliasing.
  let gsize = max(0.0008, footprint * 1.7);
  let gcell = floor(p.xz / gsize);
  let crestness = clamp(r.h / max(r.amp, 1e-4), 0.0, 1.0);
  let grainAmp = mix(1.0, 0.4, smoothstep(0.003, 0.015, footprint));
  let grain = ((hash12(gcell) - 0.5) * 0.28 + (hash12(floor(p.xz / (gsize * 2.6)) + vec2f(13.0, 5.0)) - 0.5) * 0.22) * (0.5 + 1.0 * crestness) * grainAmp
            + (vnoise(p.xz * 180.0) - 0.5) * 0.08 + (vnoise(p.xz * 45.0 + vec2f(9.0, 2.0)) - 0.5) * 0.06;
  albedo *= 1.0 + grain;

  // Grain-scale unevenness: the sand between the crests is never flat, so under the grazing
  // sun it shows a faint pebbling (about 1 mm over 4 cm), gone once a pixel spans a cell.
  let e2 = 0.004;
  let b0 = vnoise(p.xz * 25.0 + vec2f(3.0, 8.0));
  let micro = vec2f(vnoise((p.xz + vec2f(e2, 0.0)) * 25.0 + vec2f(3.0, 8.0)) - b0,
                    vnoise((p.xz + vec2f(0.0, e2)) * 25.0 + vec2f(3.0, 8.0)) - b0) / e2
              * 0.0008 * (1.0 - smoothstep(0.01, 0.03, footprint));
  let n = normalize(vec3f(-(slope.x + r.grad.x + micro.x), 1.0, -(slope.y + r.grad.y + micro.y)));
  let shadow = duneShadow(p + vec3f(0.0, 0.05, 0.0), sun) * rippleShadow(p.xz, r, sun, slope);
  // The lip of each crest catches the low sun as a thin bright line.
  let lip = smoothstep(0.85, 1.0, crestness) * 0.25;
  let v = -rd;
  let diffuse = orenNayar(n, sun, v, 0.6) * shadow;
  // The troughs between the ripples see less sky than the crests.
  let ao = mix(0.78, 1.0, crestness);
  // Bounce off the sunlit sand around: warm fill that reaches the faces turned away from the
  // sun (the steep sides and the dune's shadowed flanks) more than the flat sand.
  let bounce = rgb8(205.0, 140.0, 72.0) * sunCol * (0.05 + 0.4 * (1.0 - n.y));
  var col = albedo * (sunCol * diffuse * (1.0 + lip) + (amb * (0.6 + 0.4 * n.y) + bounce) * ao);
  let h = normalize(sun + v);
  // Sheen of the quartz grains toward the sun: crests and rims catch the light.
  col += ggxSheen(n, sun, v, 0.45, 0.04) * sunCol * shadow * 0.35;
  // A few grains catching the sun.
  let g = vec3f(hash12(gcell * 1.7 + vec2f(11.0, 3.0)), hash12(gcell * 2.3 + vec2f(3.0, 7.0)), hash12(gcell * 3.1 + vec2f(7.0, 11.0))) - 0.5;
  let gn = normalize(n + g * 1.2);
  let sparkle = step(0.985, hash12(gcell + vec2f(5.0, 9.0)));
  col += sunCol * albedo * pow(max(dot(gn, h), 0.0), 30.0) * sparkle * grainAmp * shadow * 2.0;
  return col;
}

// Haze toward the horizon, then the desert's own exposure: the sand world is lit for its
// daylight and scaled into the night's exposure here, so through the door it reads bright,
// a little over, without collapsing into the tone curve's shoulder. Every view of the sand
// world goes through this, so the debug views show exactly what the door shows.
fn sandFinish(col: vec3f, t: f32) -> vec3f {
  return mix(col, SAND_HORIZON, 1.0 - exp(-max(t - 30.0, 0.0) * 0.0025)) * SAND_EXPOSURE;
}

fn renderSand(ro: vec3f, rd: vec3f, pixelAngle: f32, tBase: f32) -> vec3f {
  let t = marchDune(ro, rd);
  if (t < 0.0) {
    return select(SAND_HORIZON, sandSky(rd), rd.y > 0.0) * SAND_EXPOSURE;
  }
  let p = ro + rd * t;
  return sandFinish(shadeSand(p, rd, pixelAngle * (tBase + t)), t);
}

// ---------------------------------------------------------------- lighting

const AMBIENT: vec3f = vec3f(0.035, 0.09, 0.16);
// Night fill over the meadow: a low blue moon and a faint sky ambient (both 1.6x the first tuning).
const MOON_COLOR: vec3f = vec3f(0.0717, 0.1024, 0.1741);
const NIGHT_AMBIENT: vec3f = vec3f(0.0096, 0.016, 0.0416);
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

const DOOR_SAMPLE_COUNT: i32 = 6;
const DOOR_SAMPLES: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-0.23, 0.35), vec2f(0.23, 0.35), vec2f(-0.23, 1.04), vec2f(0.23, 1.04), vec2f(-0.23, 1.73), vec2f(0.23, 1.73));

// Light from the sunlit dune pouring through the opening: a warm rectangular area light
// sampled at six points, each with its own shadow ray, so penumbras stay soft.
// `tangent` enables Kajiya-Kay fibre shading for blades (zero vector for surfaces).
fn doorLight(p: vec3f, n: vec3f, tangent: vec3f, transl: f32, f: DoorFrame, shadows: bool) -> vec3f {
  let front = dot(p - f.origin, f.fwd);
  if (front > -0.01) { return vec3f(0.0); }
  let fibre = dot(tangent, tangent) > 0.5;
  var sum = 0.0;
  for (var i = 0; i < DOOR_SAMPLE_COUNT; i++) {
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
  return DOOR_COLOR * params.look.w * sum / f32(DOOR_SAMPLE_COUNT);
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
    for (var j = 0; j < DOOR_SAMPLE_COUNT; j++) {
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
    acc += e / f32(DOOR_SAMPLE_COUNT) * dt;
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
  let hemi = 0.5 + 0.5 * n.y;
  return nightAlbedo * (MOON_COLOR * max(dot(n, moon), 0.0) + NIGHT_AMBIENT * hemi);
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
  let dt = 0.025;
  var s = 1.0;
  for (var i = 0; i < 16; i++) {
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
  var night = nightAlbedo * (MOON_COLOR * mix(max(dot(n, moon), 0.0), kkMoon, 0.5) * grassTransmittance(hf, moon, g.cover) + NIGHT_AMBIENT * ao);
  if (dayMix < 0.999) {
    let doorCenter = f.origin + vec3f(0.0, DOOR_H * 0.5, 0.0);
    let toDoorC = doorCenter - p;
    // Near the door every light sample gets its own shadow ray through the leaf and the
    // blades, so the leaf's shadow has a real penumbra; further out, where the spill is
    // faint, one ray toward the centre is enough.
    let near = length(toDoorC) < 14.0;
    var doorSh = 1.0;
    if (!near) {
      let lc = normalize(toDoorC);
      doorSh = doorShadow(p + vec3f(0.0, 0.02, 0.0), lc, f, length(toDoorC) - 0.03, 40.0) * grassSelfShadow(p, lc, g, base, seed);
    }
    var e2 = 0.0;
    var spec = 0.0;
    for (var j = 0; j < DOOR_SAMPLE_COUNT; j++) {
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
      var vis = doorSh;
      if (near) {
        vis = doorShadow(p + vec3f(0.0, 0.02, 0.0), l, f, d - 0.03, 40.0) * grassSelfShadow(p, l, g, base, seed);
      }
      e2 += geom * mix(lam, kk, 0.3) * tr * vis;
      let hl = normalize(l + v);
      let th = dot(tangent, hl);
      spec += geom * pow(sqrt(max(1.0 - th * th, 0.0)), 16.0) * tr * vis;
    }
    let front = dot(p - f.origin, f.fwd);
    let inFront = select(0.0, 1.0, front < -0.01);
    night += albedo * DOOR_COLOR * params.look.w * (e2 + spec * 0.3) / f32(DOOR_SAMPLE_COUNT) * inFront;
  }
  return mix(night, day, dayMix) + albedo * rimGlow(n, rim);
}

// Blades shadowing blades under the door light: the blades rasterised from the door into a
// paraboloid map (grass-blades.wgsl), looked up over 3x3 taps. The door faces -z (yaw 0).
fn bladeShadowAt(p: vec3f, f: DoorFrame) -> f32 {
  let size = params.shadow.x;
  if (size < 1.0) { return 1.0; }
  let d = p - (f.origin + vec3f(0.0, params.shadow.y, 0.0));
  let r = length(d);
  let dir = d / max(r, 1e-4);
  if (dir.z > -0.02) { return 1.0; }
  let uv = dir.xy / (1.0 - dir.z);
  let px = vec2f(uv.x * 0.5 + 0.5, 0.5 - uv.y * 0.5) * size;
  var lit = 0.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let c = vec2i(clamp(px + vec2f(f32(i), f32(j)) * 1.6, vec2f(0.0), vec2f(size - 1.0)));
      let stored = textureLoad(bladeShadow, c, 0).x;
      lit += select(1.0, 0.0, stored > 0.0 && r - params.shadow.z > stored);
    }
  }
  return lit / 9.0;
}

// A geometric blade from the G-buffer: the same fibre shading as the relief, lit by the sun
// (day) or the moon and the door (night), with the light attenuated through the sward by the
// blade's position along its length.
fn shadeBlade(p: vec3f, rd: vec3f, tangent: vec3f, albedo: vec3f, height: f32, f: DoorFrame, dayMix: f32, rim: f32) -> vec3f {
  let cover = grassCoverage(p.xz);
  // Where this point sits in the sward: its height against the local canopy (the tallest
  // blades here), so short blades and low parts of tall ones are deep inside it and get the
  // light attenuated the way the relief's layer did.
  let tuft = 0.7 + 0.6 * vnoise(p.xz * 2.4 + vec2f(3.0, 11.0));
  let canopy = 1.4 * params.grass.y * clamp(cover, 0.35, 1.2) * tuft;   // the tallest blades reach past the relief's ceiling
  let hf = clamp(height / max(canopy, 0.05), 0.0, 1.0);
  // Mean normal of a thin blade: as far up as its tangent allows.
  var n = vec3f(0.0, 1.0, 0.0) - tangent * tangent.y;
  n = normalize(select(n, vec3f(0.0, 0.0, -1.0), dot(n, n) < 1e-4));
  let v = -rd;
  let sun = sunDir();
  let sunCol = sunColor();
  let ao = 0.3 + 0.7 * hf;
  let sh = doorShadow(p + vec3f(0.0, 0.02, 0.0), sun, f, 8.0, 5.0) * terrainShadow(p, sun);
  let tlSun = dot(tangent, sun);
  let kkSun = sqrt(max(1.0 - tlSun * tlSun, 0.0));
  let lambertSun = max(dot(n, sun), 0.0);
  let trSun = grassTransmittance(hf, sun, cover);
  var day = albedo * (sunCol * mix(lambertSun, kkSun, 0.45) * trSun * sh + AMBIENT * (0.5 + 0.5 * n.y) * ao);
  let hSun = normalize(sun + v);
  let thSun = dot(tangent, hSun);
  day += sunCol * albedo * pow(sqrt(max(1.0 - thSun * thSun, 0.0)), 22.0) * 0.18 * trSun * sh;

  let lumA = dot(albedo, vec3f(0.2126, 0.7152, 0.0722));
  let nightAlbedo = mix(albedo, lumA * vec3f(0.65, 0.85, 1.0), 0.4);
  let moon = moonDir();
  let tlMoon = dot(tangent, moon);
  let kkMoon = sqrt(max(1.0 - tlMoon * tlMoon, 0.0));
  var night = nightAlbedo * (MOON_COLOR * mix(max(dot(n, moon), 0.0), kkMoon, 0.5) * grassTransmittance(hf, moon, cover) + NIGHT_AMBIENT * ao);
  if (dayMix < 0.999) {
    let doorCenter = f.origin + vec3f(0.0, DOOR_H * 0.5, 0.0);
    let toDoorC = doorCenter - p;
    let near = length(toDoorC) < 14.0;
    // The blades in front of this one, from the door's shadow map.
    let selfSh = bladeShadowAt(p, f);
    var doorSh = selfSh;
    if (!near) {
      doorSh *= doorShadow(p + vec3f(0.0, 0.02, 0.0), normalize(toDoorC), f, length(toDoorC) - 0.03, 40.0);
    }
    var e2 = 0.0;
    var spec = 0.0;
    for (var j = 0; j < DOOR_SAMPLE_COUNT; j++) {
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
      // The shadow map already holds the blades in the way; keep only a little of the
      // statistical attenuation for the sward finer than the map resolves.
      let tr = mix(1.0, grassTransmittance(hf, l, cover), 0.5);
      var vis = doorSh;
      if (near) {
        vis = selfSh * doorShadow(p + vec3f(0.0, 0.02, 0.0), l, f, d - 0.03, 40.0);
      }
      e2 += geom * mix(lam, kk, 0.3) * tr * vis;
      let hl = normalize(l + v);
      let th = dot(tangent, hl);
      spec += geom * pow(sqrt(max(1.0 - th * th, 0.0)), 16.0) * tr * vis;
    }
    let front = dot(p - f.origin, f.fwd);
    let inFront = select(0.0, 1.0, front < -0.01);
    night += albedo * DOOR_COLOR * params.look.w * (e2 + spec * 0.3) / f32(DOOR_SAMPLE_COUNT) * inFront * 0.45;
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

fn render(ro: vec3f, rd: vec3f, pixelAngle: f32, pixel: vec2i) -> vec3f {
  let f = doorFrame();
  let tPortal = portalHit(ro, rd, f);
  let tTerrain = marchTerrain(ro, rd, 0.0, TMAX);
  var tLimit = select(tTerrain, TMAX, tTerrain < 0.0);
  let tDoor = marchDoor(ro, rd, tLimit, f);

  var t = tTerrain;
  var kind = 0;   // 0 sky, 1 terrain, 2 door, 3 blade layer, 4 geometric blade
  if (t > 0.0) { kind = 1; }
  if (tDoor > 0.0 && (t < 0.0 || tDoor < t)) { t = tDoor; kind = 2; }
  // The geometric blades were rasterised for this very sub-pixel sample.
  var bladeTangent = vec3f(0.0, 1.0, 0.0);
  var bladeAlbedo = vec3f(0.0);
  var bladeAlong = 0.0;
  if (params.blades.z > 0.0) {
    let bd = textureLoad(bladeDist, pixel, 0);
    if (bd.x > 0.0 && (t < 0.0 || bd.x < t)) {
      let bc = textureLoad(bladeColor, pixel, 0);
      t = bd.x;
      kind = 4;
      bladeTangent = bd.yzw;
      bladeAlbedo = bc.rgb;
      bladeAlong = bc.a;
    }
  }
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
    } else if (kind == 4) {
      color = shadeBlade(p, rd, bladeTangent, bladeAlbedo, bladeAlong, f, dayMix, rim);
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
// overlays: 0 bare sand, 1 the door, 2 also the part seen through it, 3 also the grid and door plane.
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
  if (tDune < 0.0) { return select(SAND_HORIZON, sandSky(rd), rd.y > 0.0) * SAND_EXPOSURE; }
  let p = ro + rd * tDune;
  let footprint = pixelAngle * tDune;
  var col = shadeSand(p, rd, footprint);
  if (overlays > 2) {
    col = mix(col, vec3f(0.05, 0.05, 0.08), debugGrid(p.xz, footprint) * 0.6);
    col = mix(col, vec3f(0.1, 0.3, 1.0), smoothstep(max(footprint * 2.0, 0.006), 0.0, abs(p.z)) * 0.9);
  }
  if (overlays > 1) {
    col = mix(col, vec3f(0.15, 0.9, 0.25), 0.4 * portalVisible(p));
  }
  return sandFinish(col, tDune);
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
  var col = shadeSand(p, vec3f(0.0, -1.0, 0.0), footprint) * SAND_EXPOSURE;
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
  let weight = params.blades.w;
  if (params.debug.x > 1.5 && params.debug.x < 2.5) {
    return vec4f(renderDebugMap(uv), 1.0) * weight;
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

  // One sub-pixel sample per pass (params.blades.xy); the passes add up in the target.
  let pixel = vec2i(uv * res);
  let px = uv * res + params.blades.xy - vec2f(0.5);
  let ndc = (px / res) * 2.0 - 1.0;
  let sx = ndc.x * aspect;
  let sy = -ndc.y;
  var color = vec3f(0.0);
  if (params.debug.x > 4.5) {
    // The geometric blade G-buffer: distance (red, 50 m = 1) and the blades' shadow on each
    // other from the door light (green, lit = 0.5).
    let bd = textureLoad(bladeDist, pixel, 0);
    let rdB = normalize(forward * focal + right * sx + up * sy);
    var sh = 0.0;
    if (bd.x > 0.0) { sh = bladeShadowAt(ro + rdB * bd.x, doorFrame()); }
    color = vec3f(bd.x / 50.0, sh * 0.5, 0.0);
  } else if (params.debug.x > 2.5) {
    // The main camera carried into the sand world: same place, pitch and lens, so the
    // composition of the sand behind the door can be read as the door frames it.
    let fD = doorFrame();
    let camD = toDoor(ro, fD);
    let fwdW = vec3f(dot(forward, fD.right), forward.y, dot(forward, fD.fwd));
    let upW = vec3f(dot(up, fD.right), up.y, dot(up, fD.fwd));
    let rightW = vec3f(dot(right, fD.right), right.y, dot(right, fD.fwd));
    let rdM = normalize(fwdW * focal + rightW * sx + upW * sy);
    color = renderDebugWorld(camD, rdM, pixelAngle, select(2, 1, params.debug.x > 3.5));
  } else if (params.debug.x > 0.5) {
    // Free camera in the sand world, looking at the foot of the slip face.
    let cam = params.debug.yzw;
    let fwdD = normalize(vec3f(0.0, 1.2, 7.0) - cam);
    let rightD = normalize(cross(vec3f(0.0, 1.0, 0.0), fwdD));
    let upD = cross(fwdD, rightD);
    let focalD = 1.0 / tan(0.45);
    let rdD = normalize(fwdD * focalD + rightD * sx + upD * sy);
    color = renderDebugWorld(cam, rdD, (2.0 / focalD) / res.y, 3);
  } else {
    let rd = normalize(forward * focal + right * sx + up * sy);
    color = render(ro, rd, pixelAngle, pixel);
  }
  // One NaN would spread through the bloom chain to the whole frame: drop it here.
  let safe = select(clamp(color, vec3f(0.0), vec3f(1e4)), vec3f(0.0), color != color);
  return vec4f(safe, 1.0) * weight;
}
