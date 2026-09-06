// Packs a baked 1024² material tile into its level atlas (see
// material-common.wgsl): level 0 is copied with a periodic border, the
// coarser levels are box averages of level 0. Runs once per material.

import { MAT_SIZE, MAT_LEVELS, levelSize, levelOffset } from "./material-common.wgsl";

@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var normal: texture_2d<f32>;

struct AtlasOut {
  @location(0) color: vec4f,
  @location(1) normal: vec4f,
}

fn wrapLoad(tex: texture_2d<f32>, texel: vec2i) -> vec4f {
  let size = i32(MAT_SIZE);
  return textureLoad(tex, ((texel % size) + size) % size, 0);
}

@fragment fn fs_main(@builtin(position) position: vec4f) -> AtlasOut {
  let x = position.x;
  var out: AtlasOut;
  out.color = vec4f(0.0);
  out.normal = vec4f(0.0);
  for (var level = 0; level < MAT_LEVELS; level++) {
    let size = levelSize(level);
    let offset = levelOffset(level);
    if (x < offset || x >= offset + size + 2.0 || position.y >= size + 2.0) { continue; }
    // Local texel inside the bordered tile; -1 and size wrap around.
    let local = vec2i(i32(x - offset) - 1, i32(position.y) - 1);
    let scale = i32(MAT_SIZE / size);           // level-0 texels per texel here
    let base = local * scale;
    // Box filter over the level-0 block, subsampled to at most 16x16 taps.
    let step = max(scale / 16, 1);
    var sumColor = vec4f(0.0);
    var sumNormal = vec4f(0.0);
    var count = 0.0;
    for (var y = 0; y < scale; y += step) {
      for (var xx = 0; xx < scale; xx += step) {
        let texel = base + vec2i(xx, y);
        sumColor += wrapLoad(color, texel);
        let n = wrapLoad(normal, texel);
        sumNormal += vec4f(n.xy * 2.0 - 1.0, n.zw);
        count += 1.0;
      }
    }
    // Averaging alone leaves stone-sized bumps in the coarse levels, which
    // shade as pixel noise at a distance; flatten the normal further per level
    // and hand the lost micro-relief to roughness instead (Toksvig-style).
    let flatten = pow(0.55, f32(level));
    let n = sumNormal / count;
    let color = sumColor / count;
    out.color = vec4f(color.rgb, clamp(color.a + 0.04 * f32(level), 0.0, 1.0));
    out.normal = vec4f(n.xy * flatten * 0.5 + 0.5, n.zw);
  }
  return out;
}
