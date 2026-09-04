// Sun shadow map: rasterizes the scene from the light's orthographic camera
// and writes light-space depth (clip z in [0, 1]) into an r32float colour
// attachment, since depth textures cannot be bound for sampling.
struct Light {
  viewProj: mat4x4f,
}
@group(0) @binding(0) var<uniform> light: Light;

// The mesh carries normal/uv/material streams too; vgpu requires every
// mesh attribute to have a shader input, so they are declared and ignored.
struct VertexIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) material: f32,
}
struct VertexOut {
  @builtin(position) clip: vec4f,
  @location(0) depth: f32,
}

@vertex fn vs_main(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  out.clip = light.viewProj * vec4f(in.position, 1.0);
  out.depth = out.clip.z / out.clip.w;
  return out;
}

@fragment fn fs_main(in: VertexOut) -> @location(0) vec4f {
  return vec4f(in.depth, 0.0, 0.0, 1.0);
}
