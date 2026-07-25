import { sample_env } from "./env-common.wgsl";

// Mirror-metal cube shaded entirely from the environment map: one reflected ray per
// pixel, weighted by a conductor Fresnel term. No lights, no shadow maps — a polished
// metal surface is nothing but the environment seen from a different angle.
struct Uniforms {
  view_projection: mat4x4f,
  model: mat4x4f,
  camera_position: vec3f,
  roughness: f32,
  base_color: vec3f,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) world_position: vec3f,
  @location(1) world_normal: vec3f,
};

@vertex
fn vs_main(@location(0) position: vec3f, @location(1) normal: vec3f) -> VertexOut {
  let world = uniforms.model * vec4f(position, 1.0);
  var out: VertexOut;
  out.position = uniforms.view_projection * world;
  out.world_position = world.xyz;
  // `model` is rotation-only, so the normal needs no inverse-transpose.
  out.world_normal = (uniforms.model * vec4f(normal, 0.0)).xyz;
  return out;
}

// A small cone of taps stands in for a prefiltered roughness mip chain: enough to take
// the hard edge off the reflected checker floor without building a whole pyramid.
// Set roughness to 0 for a perfect mirror.
fn glossy_env(direction: vec3f, roughness: f32) -> vec3f {
  let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(direction.y) > 0.95);
  let tangent = normalize(cross(helper, direction));
  let bitangent = cross(direction, tangent);

  var offsets = array<vec2f, 4>(vec2f(1.0, 0.0), vec2f(-1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, -1.0));
  var sum = sample_env(env_tex, env_samp, direction);
  for (var i = 0; i < 4; i++) {
    let offset = offsets[i] * roughness;
    sum += sample_env(env_tex, env_samp, direction + tangent * offset.x + bitangent * offset.y);
  }
  return sum * 0.2;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
  let normal = normalize(in.world_normal);
  let view = normalize(uniforms.camera_position - in.world_position);
  let facing = clamp(dot(view, normal), 0.0, 1.0);

  // Conductor Fresnel: `base_color` is the metal's normal-incidence reflectance, and
  // every metal turns into a white mirror at grazing angles.
  let fresnel = uniforms.base_color + (vec3f(1.0) - uniforms.base_color) * pow(1.0 - facing, 5.0);

  return vec4f(glossy_env(reflect(-view, normal), uniforms.roughness) * fresnel, 1.0);
}
