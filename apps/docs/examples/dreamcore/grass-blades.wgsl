// Geometric grass: hundreds of thousands of blades generated in the vertex shader, one
// instance each, rasterised into a G-buffer (distance + tangent, albedo + position along the
// blade) that the scene raymarcher composites and shades. Placement, shape and colour follow
// the relief tile's recipe (grass-tile.ts / grass-tile.wgsl) so near and far grass agree.
//
// Blades fill the camera's view wedge: a near zone at full density up to the knee radius,
// then density falling as 1/r^2 to the outer radius, so the number of blades per pixel stays
// roughly even with distance.
struct Blades {
  camera: vec4f,   // height (m), pitch (rad, + looks up), vertical fov (rad), aspect (w/h)
  jitter: vec4f,   // sub-pixel offset in ndc (x, y), wind amplitude, time (s)
  door: vec4f,     // door x, door z, grass patch radius (m), blade height scale
  zone: vec4f,     // inner radius (m), knee radius (m), outer radius (m), fraction of instances inside the knee
  count: vec4f,    // instance count, unused
}

@group(0) @binding(0) var<uniform> blades: Blades;

const PI: f32 = 3.14159265359;
const SPINE: i32 = 7;            // samples along a blade, as in the tile mesh
const FAR: f32 = 200.0;          // depth range of the pass (m)

// ---- noise, shared with scene.wgsl (keep identical)

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

// ---- terrain and coverage, the parts of scene.wgsl that reach the blade zone (keep in sync)

fn bump(p: vec2f, c: vec2f, r: vec2f) -> f32 {
  let d = (p - c) / r;
  return exp(-dot(d, d));
}

fn terrainHeight(p: vec2f) -> f32 {
  var h = 0.0;
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
  return h;
}

fn grassCoverage(xz: vec2f) -> f32 {
  let clump = clamp(0.55 + 0.9 * (fbm3(xz * 0.45 + vec2f(9.0, 4.0)) - 0.3), 0.35, 1.2);
  let radius = blades.door.z;
  let inPatch = 1.0 - smoothstep(radius * 0.7, radius * 0.98, length(xz - blades.door.xy));
  let near = 1.0 - smoothstep(26.0, 42.0, length(xz));
  return clump * inPatch * near;
}

// ---- the blades

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) tangent: vec3f,
  @location(2) extra: vec2f,     // (fraction along the blade, per-blade seed)
}

fn bladeHash(ii: u32, salt: f32) -> f32 {
  // Two small coordinates keep the hash in float precision for any instance count.
  return hash12(vec2f(f32(ii % 4096u), f32(ii / 4096u)) + vec2f(salt * 17.0, salt * 3.0));
}

@vertex fn vs_main(@builtin(instance_index) ii: u32, @builtin(vertex_index) vi: u32) -> VertexOut {
  // Where the blade stands: polar coordinates around the camera, inside its view wedge.
  let u = bladeHash(ii, 1.0);
  let v = bladeHash(ii, 2.0);
  let inner = blades.zone.x;
  let knee = blades.zone.y;
  let outer = blades.zone.z;
  var r = 0.0;
  if (f32(ii) < blades.count.x * blades.zone.w) {
    r = sqrt(mix(inner * inner, knee * knee, u));        // even per square metre
  } else {
    r = knee * pow(outer / knee, u);                      // density falling as 1 / r^2
  }
  let halfAngle = atan(tan(blades.camera.z * 0.5) * blades.camera.w) + atan(0.6 / r);
  let theta = (v - 0.5) * 2.0 * halfAngle;
  let base = vec2f(r * sin(theta), r * cos(theta));

  let cover = grassCoverage(base);
  let tuft = 0.7 + 0.6 * vnoise(base * 2.4 + vec2f(3.0, 11.0));
  let keep = bladeHash(ii, 3.0) < clamp(cover, 0.0, 1.0);

  // The blade recipe of grass-tile.ts: a spine that starts a little off vertical and arcs
  // over toward the comb direction, drifting in yaw and twisting along its length.
  let gauss = (bladeHash(ii, 4.0) + bladeHash(ii, 5.0) + bladeHash(ii, 6.0) - 1.5) * 1.15;
  let comb = vec2f(sin(PI * 0.78), cos(PI * 0.78));
  let yaw0 = atan2(comb.y, comb.x) + gauss * 0.55;
  let curl = (bladeHash(ii, 7.0) - 0.5) * 0.7;
  let r1 = bladeHash(ii, 8.0);
  let r2 = bladeHash(ii, 9.0);
  let len = (0.3 + 0.3 * r1 * r1 + 0.1 * r2) * blades.door.w * clamp(cover, 0.35, 1.2) * tuft;
  let width = 0.005 + 0.006 * bladeHash(ii, 10.0);
  let theta0 = 0.25 + 0.4 * bladeHash(ii, 11.0);
  let theta1 = 1.25 + 0.5 * bladeHash(ii, 12.0);
  let twist = (bladeHash(ii, 13.0) - 0.5) * 1.4;
  let seed = bladeHash(ii, 14.0);
  let wind = blades.jitter.z * (0.5 + 0.5 * sin(blades.jitter.w * 1.6 + base.x * 0.7 + base.y * 0.4 + seed * 6.28));

  // Six vertices per quad between spine samples: (a, a+1, a+3, a, a+3, a+2).
  let quad = i32(vi) / 6;
  let k = i32(vi) % 6;
  var corner = 0;
  if (k == 1) { corner = 1; } else if (k == 2 || k == 4) { corner = 3; } else if (k == 5) { corner = 2; }
  let vtx = quad * 2 + corner;
  let i = vtx / 2;
  let side = f32(vtx % 2) * 2.0 - 1.0;

  // Integrate the spine up to this sample.
  let s = f32(i) / f32(SPINE - 1);
  let step = len / f32(SPINE - 1);
  var p = vec3f(base.x, terrainHeight(base), base.y);
  var d = vec3f(0.0, 1.0, 0.0);
  for (var j = 0; j <= i; j++) {
    let sj = f32(j) / f32(SPINE - 1);
    let th = theta0 + (theta1 - theta0) * sj + wind * 0.3 * sj;
    let yaw = yaw0 + curl * sj + wind * 0.2;
    d = vec3f(sin(th) * cos(yaw), cos(th), sin(th) * sin(yaw));
    if (j < i) { p += d * step; }
  }
  // Side vector: horizontal, perpendicular to the tangent, then twisted around it.
  let sideV = normalize(vec3f(d.z, 0.0, -d.x));
  let tau = twist * s;
  let wv = sideV * cos(tau) + cross(d, sideV) * sin(tau);
  let w = width * (1.0 - pow(s, 1.4) * 0.93);
  var pos = p + wv * w * side;
  if (!keep) { pos = vec3f(0.0, -100.0, 0.0); }   // collapsed: nothing rasterises

  // The scene camera: at (0, height, 0), pitched, looking along +z. The jitter shifts the
  // projection so the pixel centre coincides with the raymarcher's sub-pixel sample.
  let ro = vec3f(0.0, blades.camera.x, 0.0);
  let cp = cos(blades.camera.y);
  let sp = sin(blades.camera.y);
  let fwd = vec3f(0.0, sp, cp);
  let up = vec3f(0.0, cp, -sp);
  let focal = 1.0 / tan(blades.camera.z * 0.5);
  let dd = pos - ro;
  let vx = dd.x;
  let vy = dot(dd, up);
  let vz = dot(dd, fwd);
  var out: VertexOut;
  out.position = vec4f(vx * focal / blades.camera.w - blades.jitter.x * vz, vy * focal + blades.jitter.y * vz, vz * vz / FAR, vz);
  out.world = pos;
  out.tangent = d;
  out.extra = vec2f(s, seed);
  return out;
}

struct FragmentOut {
  @location(0) dist: vec4f,    // distance from the camera along the ray, blade tangent
  @location(1) color: vec4f,   // linear albedo, fraction along the blade
}

fn srgb2lin(c: vec3f) -> vec3f {
  return pow((c + vec3f(0.055)) / 1.055, vec3f(2.4));
}

@fragment fn fs_main(in: VertexOut) -> FragmentOut {
  let s = in.extra.x;
  let seed = in.extra.y;
  // Same colours as the tile: dark base to a lighter tip, per-blade variation, a few dead
  // straw-coloured blades.
  let base = srgb2lin(vec3f(46.0, 86.0, 27.0) / 255.0);
  let tip = srgb2lin(vec3f(148.0, 176.0, 64.0) / 255.0);
  var albedo = mix(base, tip, smoothstep(0.05, 1.0, s));
  albedo *= 0.72 + 0.56 * seed;
  albedo *= mix(vec3f(1.0), vec3f(1.1, 1.0, 0.82), fract(seed * 7.31) * 0.6);
  if (fract(seed * 13.7) < 0.07) {
    albedo = mix(albedo, srgb2lin(vec3f(158.0, 128.0, 66.0) / 255.0), 0.85);
  }
  let ro = vec3f(0.0, blades.camera.x, 0.0);
  var out: FragmentOut;
  out.dist = vec4f(length(in.world - ro), normalize(in.tangent));
  out.color = vec4f(albedo, s);
  return out;
}
