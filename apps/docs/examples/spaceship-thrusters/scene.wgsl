// Lit geometry pass for the engine, test stand and pad. Writes scene-linear
// radiance to @location(0) and camera distance to @location(1) so the plume
// raymarch can stop at surfaces and composite over the scene.

struct Camera {
  viewProj: mat4x4f,
  position: vec3f,
  time: f32,
}

struct Lighting {
  sunDir: vec3f,        // unit vector TOWARD the sun
  sunIntensity: f32,
  sunColor: vec3f,
  ambient: f32,
  skyColor: vec3f,
  shadowTexel: f32,     // 1 / shadow map size
  groundColor: vec3f,
  shadowBias: f32,
  sunViewProj: mat4x4f,
  shadowExtent: f32,    // world units covered by the shadow map
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
}

// Soot streak the exhaust leaves on the pad: darkens along +X from the nozzle.
fn scorch(world: vec3f) -> f32 {
  let along = smoothstep(0.3, 3.5, world.x) * (1.0 - smoothstep(24.0, 40.0, world.x));
  let across = 1.0 - smoothstep(0.8, 1.1 + world.x * 0.07, abs(world.z));
  let breakup = textureSampleLevel(detail, detailSamp, world.xz * vec2f(0.03, 0.09), 0.0).r;
  return along * across * (0.55 + 0.6 * breakup);
}

fn materialFor(id: u32, world: vec3f) -> Material {
  switch (id) {
    case 0u: { return Material(vec3f(0.022, 0.021, 0.02), 0.55, 0.1); }      // matte black (nozzle, insulated lines)
    case 1u: { return Material(vec3f(0.14, 0.145, 0.15), 0.45, 0.8); }       // dark steel housings
    case 2u: { return Material(vec3f(0.42, 0.43, 0.44), 0.32, 0.95); }       // stainless lines and valves
    case 3u: {                                                                // concrete pad
      let grain = textureSampleLevel(detail, detailSamp, world.xz * 0.045, 0.0);
      let fine = textureSampleLevel(detail, detailSamp, world.xz * 0.6, 0.0).r;
      let cell = world.xz / 8.0 + vec2f(0.25, 0.5);
      let joints = 1.0 - 0.22 * (1.0 - smoothstep(0.0, 0.014, min(abs(fract(cell.x) - 0.5), abs(fract(cell.y) - 0.5))));
      let slab = floor(cell + 0.5);
      let tone = 0.92 + 0.14 * fract(sin(dot(slab, vec2f(12.9898, 78.233))) * 43758.5453);
      let patches = textureSampleLevel(detail, detailSamp, world.xz * 0.012 + vec2f(0.3, 0.7), 0.0).r;
      let albedo = vec3f(0.33, 0.315, 0.285) * (0.72 + 0.3 * grain.r + 0.1 * fine + 0.25 * patches) * joints * tone;
      let burn = scorch(world);
      return Material(mix(albedo, vec3f(0.09, 0.08, 0.075), burn * 0.8), 0.9 + 0.08 * burn, 0.0);
    }
    case 4u: {                                                                // gravel apron
      let grain = textureSampleLevel(detail, detailSamp, world.xz * 0.12, 0.0);
      let fine = textureSampleLevel(detail, detailSamp, world.xz * 1.1, 0.0).g;
      let stone = textureSampleLevel(detail, detailSamp, world.xz * 2.5, 0.0).b;
      var albedo = vec3f(0.36, 0.32, 0.26) * (0.85 + 0.25 * grain.r + 0.3 * fine);
      albedo *= mix(0.65, 1.45, smoothstep(0.35, 0.75, stone));
      return Material(mix(albedo, vec3f(0.09, 0.08, 0.075), scorch(world) * 0.6), 0.95, 0.0);
    }
    case 5u: {                                                                // painted dark steel (stand), worn
      let wear = textureSampleLevel(detail, detailSamp, world.xz * 0.7 + world.y * 0.37, 0.0).g;
      return Material(vec3f(0.075, 0.08, 0.085) * (0.75 + 0.5 * wear), 0.5 + 0.25 * wear, 0.35);
    }
    case 6u: { return Material(vec3f(0.85, 0.85, 0.82), 0.7, 0.0); }         // white decal
    default: { return Material(vec3f(0.85, 0.6, 0.08), 0.5, 0.2); }          // safety yellow
  }
}

// Bilinear-weighted 2x2 PCF against the light-space depth from shadow.wgsl.
fn sunVisibility(world: vec3f, n: vec3f) -> f32 {
  // Normal-offset + slope-scaled bias, both in shadow texels.
  let ndl = clamp(dot(n, lighting.sunDir), 0.0, 1.0);
  let texelWorld = lighting.shadowTexel * lighting.shadowExtent;
  let offsetWorld = world + n * texelWorld * (1.5 + 3.0 * (1.0 - ndl));
  let clip = lighting.sunViewProj * vec4f(offsetWorld, 1.0);
  let ndc = clip.xyz / clip.w;
  if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z > 1.0) { return 1.0; }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let size = vec2f(textureDimensions(shadowMap));
  let base = uv * size - 0.5;
  let i0 = vec2i(floor(base));
  let f = fract(base);
  let bias = lighting.shadowBias * (1.0 + 2.0 * (1.0 - ndl));
  var taps = array<f32, 4>();
  for (var k = 0; k < 4; k++) {
    let texel = clamp(i0 + vec2i(k & 1, k >> 1), vec2i(0), vec2i(size) - 1);
    taps[k] = select(0.0, 1.0, ndc.z - bias <= textureLoad(shadowMap, texel, 0).r);
  }
  return mix(mix(taps[0], taps[1], f.x), mix(taps[2], taps[3], f.x), f.y);
}

fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  return f0 + (vec3f(1.0) - f0) * pow(1.0 - cosTheta, 5.0);
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
  let h = normalize(l + v);
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
  var n = normalize(in.normal);
  // No cull mode: both faces rasterize, so flip normals seen from behind
  // (the inside of the bell, the underside of pipes).
  if (!frontFacing) { n = -n; }
  let v = normalize(camera.position - in.world);
  let m = materialFor(in.material, in.world);

  var color = shade(n, v, lighting.sunDir, lighting.sunColor * lighting.sunIntensity, m) * sunVisibility(in.world, n);

  // Hemisphere ambient: sky from above, warm bounce from the pad below.
  let up = n.y * 0.5 + 0.5;
  let f0 = mix(vec3f(0.04), m.albedo, m.metallic);
  let ambientSpec = fresnelSchlick(max(dot(n, v), 0.0), f0) * (1.0 - m.roughness) * 0.5;
  color += mix(lighting.groundColor, lighting.skyColor, up) * lighting.ambient * (m.albedo * (1.0 - m.metallic) + ambientSpec);

  // Plume glow: closest point on the exhaust segment, colour following the
  // exhaust (blue-white at the exit, pink downstream), intensity peaking in
  // the afterburning zone.
  {
    let s = clamp(dot(in.world - plumeLight.nozzle, plumeLight.axis), 0.0, plumeLight.length);
    let p = plumeLight.nozzle + plumeLight.axis * s;
    let toLight = p - in.world;
    let dist2 = max(dot(toLight, toLight), 0.25);
    let l = toLight * inverseSqrt(dist2);
    let profile = 0.25 + smoothstep(0.0, 6.0, s) * (1.0 - smoothstep(18.0, 32.0, s));
    let tint = mix(vec3f(0.75, 0.8, 1.0), vec3f(1.0, 0.55, 0.42), smoothstep(1.0, 8.0, s));
    color += shade(n, v, l, tint * (plumeLight.intensity * profile / dist2), m);
  }

  var out: FragOut;
  out.color = vec4f(color, 1.0);
  out.depth = vec4f(distance(camera.position, in.world), 0.0, 0.0, 1.0);
  return out;
}
