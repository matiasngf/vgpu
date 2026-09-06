import { Plume, GRID_TILE, GRID_BORDER, GRID_STRIDE, GRID_COLS, GRID_SLICES, plumeFrame, boundRadius, blackbody, evaluatePlume } from "./plume-volume.wgsl";

// Fills the plume grid once per frame (froxel-style, but aligned to the plume
// instead of the camera): each atlas texel is one voxel of the cone-fitted
// grid, and stores emission (rgb) and extinction (a) per world unit. The
// raymarch then reads two texels per step instead of evaluating the volume.

struct Params {
  time: f32,
  motion: f32,
  // Temporal amortization: -1 refills every slice; otherwise the frame index.
  // Near slices (the exit capsule and shock structure) refresh every other
  // frame, far slices (big, slow, blurry) every fourth, and the pass runs with
  // clear: false so the rest is kept. Neighbouring slices then differ by a
  // frame or two, which the trilinear lookup blends into a slight motion blur
  // along the axis.
  frame: i32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var detail: texture_2d<f32>;
@group(0) @binding(3) var atlasSamp: sampler;
@group(0) @binding(4) var detailSamp: sampler;
@group(0) @binding(5) var<uniform> plume: Plume;

@fragment fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let px = vec2i(position.xy);
  let stride = i32(GRID_STRIDE);
  let tile = px / stride;
  let slice = tile.y * GRID_COLS + tile.x;
  if (slice >= GRID_SLICES) { return vec4f(0.0); }
  if (params.frame >= 0) {
    let period = select(4, 2, slice < GRID_SLICES / 4);
    if ((slice % period) != (params.frame % period)) { discard; }
  }
  // Local texel -1..TILE (the border texels evaluate just outside the cone,
  // where the volume is empty, so filtering fades to zero at the edge).
  let local = vec2f(px - tile * stride) - GRID_BORDER + 0.5;
  let g = local / GRID_TILE * 2.0 - 1.0;
  let s = (f32(slice) + 0.5) / f32(GRID_SLICES) * plume.length;
  let bound = boundRadius(plume, s);
  let frame = plumeFrame(plume.axis);
  let p = plume.nozzle + plume.axis * s + frame[0] * (g.x * bound) + frame[1] * (g.y * bound);
  let time = params.time * params.motion;
  return evaluatePlume(atlas, atlasSamp, detail, detailSamp, plume, frame, p, time, blackbody(2900.0), blackbody(1900.0), blackbody(2600.0));
}
