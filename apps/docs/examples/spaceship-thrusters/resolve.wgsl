// Temporal resolve for the interleaved plume (Nubis-style): each frame the
// march pass renders one phase of a 2x2 pattern into a quarter-size target;
// this pass scatters it into the half-resolution history and carries the
// other three phases over from the previous frame. With a static camera the
// reprojection is the identity, so no motion vectors are needed.
struct Resolve {
  phase: u32,
  blend: f32,    // how much of the fresh sample replaces its 4-frame-old history
  neighbor: f32, // how much stale phases lean on the fresh sample of their 2x2 block
}

@group(0) @binding(0) var marchFire: texture_2d<f32>;
@group(0) @binding(1) var marchAux: texture_2d<f32>;
@group(0) @binding(2) var historyFire: texture_2d<f32>;
@group(0) @binding(3) var historyAux: texture_2d<f32>;
@group(0) @binding(4) var<uniform> resolve: Resolve;

struct Out {
  @location(0) fire: vec4f,
  @location(1) aux: vec4f,
}

@fragment fn fs_main(@builtin(position) position: vec4f) -> Out {
  let pixel = vec2i(position.xy);
  let phase = u32(pixel.x & 1) + 2u * u32(pixel.y & 1);
  var out: Out;
  let oldFire = textureLoad(historyFire, pixel, 0);
  let oldAux = textureLoad(historyAux, pixel, 0);
  if (phase == resolve.phase) {
    let small = pixel / 2;
    let fresh = textureLoad(marchFire, small, 0);
    let freshAux = textureLoad(marchAux, small, 0);
    // A history that was never written (transmittance 0 after clear) is
    // replaced outright.
    let weight = select(resolve.blend, 1.0, oldFire.a <= 0.0);
    out.fire = mix(oldFire, fresh, weight);
    out.aux = freshAux;
  } else {
    // Stale phase: pull it slightly toward the freshly marched pixel of the
    // same 2x2 block so fast-moving fire does not read as a mosaic.
    let fresh = textureLoad(marchFire, pixel / 2, 0);
    let weight = select(resolve.neighbor, 1.0, oldFire.a <= 0.0 && oldFire.r <= 0.0);
    out.fire = mix(oldFire, fresh, weight);
    out.aux = oldAux;
  }
  return out;
}
