import { sample_env } from "./env-common.wgsl";

// Glass cube shaded entirely from the environment map: a Fresnel-weighted mix of one
// reflected ray and three refracted rays (one per channel, which is what makes the
// dispersion rainbow). No lights, no shadow maps — just the 360° texture.
struct Uniforms {
  view_projection: mat4x4f,
  model: mat4x4f,
  camera_position: vec3f,
  ior: f32,
  dispersion: f32,
  absorption: f32,
  half_extent: f32,
  edge_tint: f32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) world_position: vec3f,
  @location(1) world_normal: vec3f,
  @location(2) local_position: vec3f,
};

@vertex
fn vs_main(@location(0) position: vec3f, @location(1) normal: vec3f) -> VertexOut {
  let world = uniforms.model * vec4f(position, 1.0);
  var out: VertexOut;
  out.position = uniforms.view_projection * world;
  out.world_position = world.xyz;
  // `model` is rotation-only, so the normal needs no inverse-transpose.
  out.world_normal = (uniforms.model * vec4f(normal, 0.0)).xyz;
  out.local_position = position;
  return out;
}

fn rotation() -> mat3x3f {
  return mat3x3f(uniforms.model[0].xyz, uniforms.model[1].xyz, uniforms.model[2].xyz);
}

// Distance from a point inside the cube to the wall it leaves through (slab test,
// exit side only, so the smallest positive slab hit wins).
fn exit_distance(origin: vec3f, direction: vec3f, half_extent: f32) -> f32 {
  let safe = select(direction, vec3f(1e-5), abs(direction) < vec3f(1e-5));
  let planes = (sign(safe) * half_extent - origin) / safe;
  return max(min(min(planes.x, planes.y), planes.z), 0.0);
}

fn box_normal(point: vec3f, half_extent: f32) -> vec3f {
  let d = abs(point) / max(half_extent, 1e-5);
  let axis = step(max(d.y, d.z), d.x) * vec3f(1.0, 0.0, 0.0)
    + step(max(d.x, d.z), d.y) * vec3f(0.0, 1.0, 0.0)
    + step(max(d.x, d.y), d.z) * vec3f(0.0, 0.0, 1.0);
  return normalize(sign(point) * axis);
}

struct Refraction {
  direction: vec3f,
  distance: f32,
};

// Two-interface refraction: bend on the way in, walk to the far wall in local space,
// bend again on the way out. A single-interface approximation looks like a bubble;
// this looks like a solid block.
fn refract_through_cube(entry_point: vec3f, incident: vec3f, normal: vec3f, ior: f32) -> Refraction {
  let inside = refract(incident, normal, 1.0 / ior);
  if (dot(inside, inside) < 1e-6) {
    return Refraction(reflect(incident, normal), 0.0);
  }

  let rot = rotation();
  let local_direction = normalize(transpose(rot) * inside);
  let travel = exit_distance(entry_point, local_direction, uniforms.half_extent);
  let exit_local = entry_point + local_direction * travel;
  let exit_normal = normalize(rot * box_normal(exit_local, uniforms.half_extent));

  // At the exit the ray leaves the dense medium, so the usable normal faces inward.
  var outgoing = refract(inside, -exit_normal, ior);
  if (dot(outgoing, outgoing) < 1e-6) {
    outgoing = reflect(inside, -exit_normal);
  }
  return Refraction(normalize(outgoing), travel);
}

fn refracted_sample(entry_point: vec3f, incident: vec3f, normal: vec3f, ior: f32) -> vec4f {
  let path = refract_through_cube(entry_point, incident, normal, ior);
  return vec4f(sample_env(env_tex, env_samp, path.direction), path.distance);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
  let normal = normalize(in.world_normal);
  let incident = normalize(in.world_position - uniforms.camera_position);
  let facing = clamp(dot(-incident, normal), 0.0, 1.0);

  // Schlick, with glass's usual 4% base reflectance.
  let fresnel = 0.04 + 0.96 * pow(1.0 - facing, 5.0);

  let reflected = sample_env(env_tex, env_samp, reflect(incident, normal));

  // One chain per channel: shorter wavelengths bend harder, so red/green/blue exit
  // along slightly different directions and the edges split into a spectrum.
  let red = refracted_sample(in.local_position, incident, normal, uniforms.ior - uniforms.dispersion);
  let green = refracted_sample(in.local_position, incident, normal, uniforms.ior);
  let blue = refracted_sample(in.local_position, incident, normal, uniforms.ior + uniforms.dispersion);
  var transmitted = vec3f(red.r, green.g, blue.b);

  // Beer-Lambert tint: the longer the ray stays in the glass, the greener it gets.
  let path = (red.a + green.a + blue.a) / 3.0;
  transmitted *= exp(-uniforms.absorption * path * vec3f(1.15, 0.72, 0.9));

  let color = mix(transmitted, reflected, fresnel)
    + vec3f(0.35, 0.55, 0.85) * pow(1.0 - facing, 8.0) * uniforms.edge_tint;
  return vec4f(color, 1.0);
}
