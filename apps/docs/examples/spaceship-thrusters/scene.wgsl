import { MAT_SIZE, MAT_LEVELS, atlasUv } from "./material-common.wgsl";

// Lit geometry pass for the engine, test stand and pad. Writes scene-linear
// radiance to @location(0), camera distance to @location(1) so the plume
// raymarch can stop at surfaces and composite over the scene, and the world
// normal plus the "occludable" share of the radiance to @location(2) for the
// screen-space ambient occlusion pass of the social pipeline.

struct Camera {
  viewProj: mat4x4f,
  position: vec3f,
  time: f32,
  pixelAngle: f32,      // radians per pixel, for fading baked detail with distance
}

// Night: no sun. Two floodlights on poles (the key one casts the shadow map)
// and the plume light the scene; a faint night-sky ambient fills the rest.
struct Lighting {
  skyColor: vec3f,
  ambient: f32,
  groundColor: vec3f,
  shadowTexel: f32,     // 1 / shadow map size
  shadowViewProj: mat4x4f,  // key floodlight, perspective
  fogColor: vec3f,
  fogDensity: f32,
  keyLight: vec4f,      // xyz position, w intensity (key floodlight, shadowed)
  keyColor: vec3f,
  shadowBias: f32,      // world units
  keySpot: vec4f,       // xyz unit direction the lamp points, w cos(half angle)
  fillLight: vec4f,     // xyz position, w intensity (fill floodlight)
  fillColor: vec3f,
  shadowFov: f32,       // tan(fov / 2) of the shadow camera, for the normal offset
  fillSpot: vec4f,
}

// The plume lights the engine and pad as a line segment: the closest point
// on the axis illuminates each fragment, with intensity and colour that
// follow the exhaust (blue-white at the exit, pink further out).
struct PlumeLight {
  nozzle: vec3f,
  length: f32,
  axis: vec3f,
  intensity: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var<uniform> plumeLight: PlumeLight;
@group(0) @binding(3) var detail: texture_2d<f32>;
@group(0) @binding(4) var detailSamp: sampler;
@group(0) @binding(5) var shadowMap: texture_2d<f32>;
// Baked ground materials (see bake-material-*.wgsl): level atlases of albedo +
// roughness and tangent normal + height + cavity, sampled with a clamp
// sampler through atlasUv (the tiles carry their own periodic border).
@group(0) @binding(6) var concreteColor: texture_2d<f32>;
@group(0) @binding(7) var concreteNormal: texture_2d<f32>;
@group(0) @binding(8) var gravelColor: texture_2d<f32>;
@group(0) @binding(9) var gravelNormal: texture_2d<f32>;
@group(0) @binding(10) var atlasSamp: sampler;

const CONCRETE_TILE = 3.0;   // world units per texture repeat
const GRAVEL_TILE = 0.9;
/** Normal-map strength per ground material (the gravel relief is baked strong; only a little is wanted). */
const CONCRETE_BUMP = 1.0;
const GRAVEL_BUMP = 0.35;

struct VertexIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) material: f32,
}

struct VertexOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) @interpolate(flat, either) material: u32,
}

struct FragOut {
  @location(0) color: vec4f,
  @location(1) depth: vec4f,
  // xyz: world normal * 0.5 + 0.5; w: share of the radiance that ambient
  // occlusion is allowed to darken (ambient fully, the wide local lights partly).
  @location(2) aux: vec4f,
}

@vertex fn vs_main(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  out.clip = camera.viewProj * vec4f(in.position, 1.0);
  out.world = in.position;
  out.normal = in.normal;
  out.uv = in.uv;
  out.material = u32(in.material + 0.5);
  return out;
}

struct Material {
  albedo: vec3f,
  roughness: f32,
  metallic: f32,
  normal: vec3f,    // shading normal (normal-mapped for the ground)
}

struct GroundSample {
  albedo: vec3f,    // ~1.0 mean, multiplied by the material's base colour
  roughness: f32,
  normal: vec3f,    // world space
  cavity: f32,
}

// One trilinear tap of a baked ground material: colour + tangent normal from
// the two atlas levels around `lod`, with the tangent frame rotated by `rot`
// (the anti-tiling second layer is sampled on a rotated uv set, so its
// normal has to be rotated back).
fn groundTap(colorTex: texture_2d<f32>, normalTex: texture_2d<f32>, uv: vec2f, lod: f32, rot: mat2x2f) -> array<vec4f, 2> {
  let l0 = i32(floor(lod));
  let l1 = min(l0 + 1, MAT_LEVELS - 1);
  let t = fract(lod);
  let c = mix(textureSampleLevel(colorTex, atlasSamp, atlasUv(uv, l0), 0.0), textureSampleLevel(colorTex, atlasSamp, atlasUv(uv, l1), 0.0), t);
  let nm = mix(textureSampleLevel(normalTex, atlasSamp, atlasUv(uv, l0), 0.0), textureSampleLevel(normalTex, atlasSamp, atlasUv(uv, l1), 0.0), t);
  let tangent = rot * (nm.xy * 2.0 - 1.0);
  // Reconstruct the up component; the atlas only stores xy.
  let up = sqrt(max(1.0 - dot(tangent, tangent), 0.0));
  return array<vec4f, 2>(c, vec4f(tangent, up, nm.w));
}

// Two layers of the same tile, the second rotated and rescaled, blended by a
// macro mask (the "randomized tiling" trick) so the repeat never lines up;
// then the tangent normal is applied on the world XZ frame.
fn groundMaterial(colorTex: texture_2d<f32>, normalTex: texture_2d<f32>, world: vec3f, n: vec3f, tile: f32, bump: f32) -> GroundSample {
  let uvA = world.xz / tile;
  let ca = 0.8; let sa = 0.6; // 37 degrees
  let rotB = mat2x2f(ca, sa, -sa, ca);
  let scaleB = 0.83;
  let uvB = rotB * world.xz / (tile * scaleB) + vec2f(0.37, 0.71);
  let macroMask = textureSampleLevel(detail, detailSamp, world.xz * 0.021 + vec2f(0.13, 0.57), 0.0).r;
  let m = smoothstep(0.38, 0.62, macroMask);
  // Level-0 texels per pixel; each atlas level is 4x coarser (2 in lod units).
  let footprint = distance(camera.position, world) * camera.pixelAngle / (tile / MAT_SIZE);
  let lod = clamp(0.5 * log2(max(footprint, 1.0)), 0.0, f32(MAT_LEVELS - 1));
  let a = groundTap(colorTex, normalTex, uvA, lod, mat2x2f(1.0, 0.0, 0.0, 1.0));
  let b = groundTap(colorTex, normalTex, uvB, clamp(lod - 0.5 * log2(scaleB), 0.0, f32(MAT_LEVELS - 1)), transpose(rotB));
  let color = mix(a[0], b[0], m);
  let nm = mix(a[1], b[1], m);
  // Tangent frame of the flat ground: +X, +Z, up; `bump` scales the tilt.
  let tilt = nm.xy * bump;
  let up = sqrt(max(1.0 - dot(tilt, tilt), 0.0));
  var out: GroundSample;
  out.albedo = color.rgb * 2.0;
  out.roughness = color.a;
  out.normal = normalize(vec3f(tilt.x, up, tilt.y));
  out.cavity = nm.w;
  return out;
}

// Soot streak the exhaust leaves on the pad: darkens along +X from the nozzle.
fn scorch(world: vec3f) -> f32 {
  let along = smoothstep(0.3, 3.5, world.x) * (1.0 - smoothstep(24.0, 40.0, world.x));
  let across = 1.0 - smoothstep(0.8, 1.1 + world.x * 0.07, abs(world.z));
  let breakup = textureSampleLevel(detail, detailSamp, world.xz * vec2f(0.03, 0.09), 0.0).r;
  return along * across * (0.55 + 0.6 * breakup);
}

fn materialFor(id: u32, world: vec3f, n: vec3f) -> Material {
  switch (id) {
    case 0u: { return Material(vec3f(0.022, 0.021, 0.02), 0.55, 0.1, n); }   // matte black (nozzle, insulated lines)
    case 1u: { return Material(vec3f(0.14, 0.145, 0.15), 0.45, 0.8, n); }    // dark steel housings
    case 2u: { return Material(vec3f(0.42, 0.43, 0.44), 0.32, 0.95, n); }    // stainless lines and valves
    case 3u: {                                                                // concrete pad
      let g = groundMaterial(concreteColor, concreteNormal, world, n, CONCRETE_TILE, CONCRETE_BUMP);
      // Slabs: joints with a bevelled edge, a random tone per slab, and broad
      // damp / weathered patches across several slabs.
      let cell = world.xz / 8.0 + vec2f(0.25, 0.5);
      let toJoint = vec2f(fract(cell.x) - 0.5, fract(cell.y) - 0.5);
      let jointDist = min(abs(toJoint.x), abs(toJoint.y));
      // Sawn joints ~4 cm wide with a ~2 cm chamfer sloping into them
      // (cell = 8 world units, so 0.005 cells = 4 cm).
      let joints = 1.0 - 0.3 * (1.0 - smoothstep(0.003, 0.0055, jointDist));
      let bevel = (1.0 - smoothstep(0.005, 0.0075, jointDist)) * step(0.0045, jointDist);
      var normal = g.normal;
      if (bevel > 0.0) {
        let axis = select(vec3f(0.0, 0.0, -sign(toJoint.y)), vec3f(-sign(toJoint.x), 0.0, 0.0), abs(toJoint.x) < abs(toJoint.y));
        normal = normalize(mix(normal, axis, bevel * 0.25));
      }
      let slab = floor(cell + 0.5);
      let tone = 0.92 + 0.14 * fract(sin(dot(slab, vec2f(12.9898, 78.233))) * 43758.5453);
      let patches = textureSampleLevel(detail, detailSamp, world.xz * 0.012 + vec2f(0.3, 0.7), 0.0).r;
      let damp = smoothstep(0.55, 0.8, textureSampleLevel(detail, detailSamp, world.xz * 0.02 + vec2f(0.6, 0.2), 0.0).r);
      var albedo = vec3f(0.33, 0.315, 0.285) * g.albedo * (0.82 + 0.25 * patches) * joints * tone;
      albedo = mix(albedo, albedo * vec3f(0.72, 0.7, 0.68), damp * 0.6);
      let burn = scorch(world);
      albedo = mix(albedo, vec3f(0.09, 0.08, 0.075), burn * 0.8);
      let roughness = g.roughness + 0.08 * burn - 0.15 * damp + 0.1 * (1.0 - joints);
      return Material(albedo, clamp(roughness, 0.3, 1.0), 0.0, normal);
    }
    case 4u: {                                                                // gravel apron
      let g = groundMaterial(gravelColor, gravelNormal, world, n, GRAVEL_TILE, GRAVEL_BUMP);
      // Broad colour drift across the apron (wetter and darker in places).
      let patches = textureSampleLevel(detail, detailSamp, world.xz * 0.015 + vec2f(0.1, 0.4), 0.0).r;
      var albedo = vec3f(0.36, 0.32, 0.26) * g.albedo * (0.8 + 0.35 * patches);
      albedo = mix(albedo, vec3f(0.09, 0.08, 0.075), scorch(world) * 0.6);
      return Material(albedo, clamp(g.roughness, 0.3, 1.0), 0.0, g.normal);
    }
    case 5u: {                                                                // painted dark steel (stand), worn
      let wear = textureSampleLevel(detail, detailSamp, world.xz * 0.7 + world.y * 0.37, 0.0).g;
      return Material(vec3f(0.075, 0.08, 0.085) * (0.75 + 0.5 * wear), 0.5 + 0.25 * wear, 0.35, n);
    }
    case 6u: { return Material(vec3f(0.85, 0.85, 0.82), 0.7, 0.0, n); }      // white decal
    case 7u: { return Material(vec3f(0.85, 0.6, 0.08), 0.5, 0.2, n); }       // safety yellow
    default: { return Material(vec3f(1.0, 0.95, 0.85), 0.3, 0.0, n); }       // lamp face (emissive, see fs_main)
  }
}

// Bilinear-weighted 2x2 PCF against the light distance from shadow.wgsl.
fn keyVisibility(world: vec3f, n: vec3f) -> f32 {
  let toLight = lighting.keyLight.xyz - world;
  let dist = length(toLight);
  let ndl = clamp(dot(n, toLight / dist), 0.0, 1.0);
  // Normal-offset + slope-scaled bias, in shadow texels at this distance.
  let texelWorld = lighting.shadowTexel * 2.0 * lighting.shadowFov * dist;
  let offsetWorld = world + n * texelWorld * (1.5 + 3.0 * (1.0 - ndl));
  let clip = lighting.shadowViewProj * vec4f(offsetWorld, 1.0);
  if (clip.w <= 0.0) { return 0.0; }
  let ndc = clip.xyz / clip.w;
  if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z > 1.0) { return 1.0; }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let size = vec2f(textureDimensions(shadowMap));
  let base = uv * size - 0.5;
  let i0 = vec2i(floor(base));
  let f = fract(base);
  let bias = lighting.shadowBias * (1.0 + 2.0 * (1.0 - ndl));
  let offsetDist = distance(offsetWorld, lighting.keyLight.xyz);
  var taps = array<f32, 4>();
  for (var k = 0; k < 4; k++) {
    let texel = clamp(i0 + vec2i(k & 1, k >> 1), vec2i(0), vec2i(size) - 1);
    taps[k] = select(0.0, 1.0, offsetDist - bias <= textureLoad(shadowMap, texel, 0).r);
  }
  return mix(mix(taps[0], taps[1], f.x), mix(taps[2], taps[3], f.x), f.y);
}

// Floodlight: inverse-square point light with a soft spot cone.
fn floodlight(light: vec4f, color: vec3f, spot: vec4f, world: vec3f, n: vec3f, v: vec3f, m: Material) -> vec3f {
  let toLight = light.xyz - world;
  let dist2 = max(dot(toLight, toLight), 0.5);
  let l = toLight * inverseSqrt(dist2);
  let cone = smoothstep(spot.w, spot.w + 0.35, dot(-l, spot.xyz));
  return shade(n, v, l, color * (light.w * cone / dist2), m);
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  // Clamp before the power: a dot product can round to slightly above 1 and
  // pow() of a negative base is NaN, which the bloom blur then smears into a
  // black rectangle.
  let f = clamp(1.0 - cosTheta, 0.0, 1.0);
  let f2 = f * f;
  return f0 + (vec3f(1.0) - f0) * (f2 * f2 * f);
}

fn ggx(n: vec3f, h: vec3f, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let ndh = max(dot(n, h), 0.0);
  let d = ndh * ndh * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265 * d * d + 1e-5);
}

fn shade(n: vec3f, v: vec3f, l: vec3f, radiance: vec3f, m: Material) -> vec3f {
  let ndl = max(dot(n, l), 0.0);
  if (ndl <= 0.0) { return vec3f(0.0); }
  // l == -v (light exactly behind the fragment on the view ray) would make
  // the half vector undefined; fall back to the normal.
  let lv = l + v;
  let h = select(n, normalize(lv), dot(lv, lv) > 1e-8);
  let f0 = mix(vec3f(0.04), m.albedo, m.metallic);
  let f = fresnelSchlick(max(dot(h, v), 0.0), f0);
  let d = ggx(n, h, max(m.roughness, 0.05));
  let k = (m.roughness + 1.0) * (m.roughness + 1.0) / 8.0;
  let ndv = max(dot(n, v), 1e-3);
  let g = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k));
  let spec = f * d * g / (4.0 * ndv * ndl + 1e-4);
  let kd = (vec3f(1.0) - f) * (1.0 - m.metallic);
  return (kd * m.albedo / 3.14159265 + spec) * radiance * ndl;
}

@fragment fn fs_main(in: VertexOut, @builtin(front_facing) frontFacing: bool) -> FragOut {
  var geometricNormal = normalize(in.normal);
  // No cull mode: both faces rasterize, so flip normals seen from behind
  // (the inside of the bell, the underside of pipes).
  if (!frontFacing) { geometricNormal = -geometricNormal; }
  let v = normalize(camera.position - in.world);
  let m = materialFor(in.material, in.world, geometricNormal);
  // Shading uses the (possibly normal-mapped) material normal; the shadow
  // lookup and the occlusion pass keep the geometric one.
  let n = m.normal;

  // Key floodlight, shadowed by the baked map.
  let key = floodlight(lighting.keyLight, lighting.keyColor, lighting.keySpot, in.world, n, v, m) * keyVisibility(in.world, geometricNormal);
  var color = key;
  // Radiance that screen-space occlusion may darken: the hemisphere ambient
  // fully, and the lamps and plume partly (all are wide sources whose light
  // also arrives from the sides, so creases receive less of them).
  var occludable = key * 0.5;

  // Hemisphere ambient: sky from above, warm bounce from the pad below.
  let up = n.y * 0.5 + 0.5;
  let f0 = mix(vec3f(0.04), m.albedo, m.metallic);
  let ambientSpec = fresnelSchlick(max(dot(n, v), 0.0), f0) * (1.0 - m.roughness) * 0.5;
  let ambient = mix(lighting.groundColor, lighting.skyColor, up) * lighting.ambient * (m.albedo * (1.0 - m.metallic) + ambientSpec);
  color += ambient;
  occludable += ambient;

  // Plume glow: closest point on the exhaust segment, colour following the
  // exhaust (blue at the exit, magenta-violet downstream), intensity peaking
  // in the afterburning zone.
  {
    let s = clamp(dot(in.world - plumeLight.nozzle, plumeLight.axis), 0.0, plumeLight.length);
    let p = plumeLight.nozzle + plumeLight.axis * s;
    let toLight = p - in.world;
    let dist2 = max(dot(toLight, toLight), 0.25);
    let l = toLight * inverseSqrt(dist2);
    let profile = 0.15 + smoothstep(0.0, 6.0, s) * (1.0 - smoothstep(18.0, 32.0, s));
    let tint = mix(vec3f(0.55, 0.65, 1.0), vec3f(0.95, 0.6, 1.0), smoothstep(1.0, 8.0, s));
    let glow = shade(n, v, l, tint * (plumeLight.intensity * profile / dist2), m);
    color += glow;
    occludable += glow * 0.7;
  }

  // Fill floodlight on the far side, unshadowed.
  {
    let flood = floodlight(lighting.fillLight, lighting.fillColor, lighting.fillSpot, in.world, n, v, m);
    color += flood;
    occludable += flood * 0.5;
  }
  // The lamp faces themselves glow.
  if (in.material == 8u) { color += vec3f(1.0, 0.98, 0.92) * 12.0; }

  // Night haze: distant ground fades toward the (near black) sky.
  let viewDistance = distance(camera.position, in.world);
  let fog = 1.0 - exp(-viewDistance * lighting.fogDensity);
  let occludableShare = luminance(occludable) * (1.0 - fog) / max(luminance(color), 1e-4);
  color = mix(color, lighting.fogColor, fog);

  var out: FragOut;
  out.color = vec4f(color, 1.0);
  out.depth = vec4f(viewDistance, 0.0, 0.0, 1.0);
  out.aux = vec4f(geometricNormal * 0.5 + 0.5, clamp(occludableShare, 0.0, 1.0));
  return out;
}
