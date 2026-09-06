// Vertical pass of the depth-aware blur, folded into the apply: the lit scene
// is darkened by the blurred occlusion, but only the share of its radiance the
// scene pass flagged as occludable (ambient and the wide local lights), so
// direct sunlight and the emissive lamp are left alone.

@group(0) @binding(0) var src: texture_2d<f32>;        // horizontally blurred occlusion
@group(0) @binding(1) var sceneDepth: texture_2d<f32>; // camera distance
@group(0) @binding(2) var scene: texture_2d<f32>;      // lit geometry
@group(0) @binding(3) var sceneAux: texture_2d<f32>;   // normal, occludable share

@fragment fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(position.xy);
  let color = textureLoad(scene, pixel, 0);
  let dist = textureLoad(sceneDepth, pixel, 0).r;
  if (dist <= 0.0) { return color; }
  let size = vec2i(textureDimensions(src));
  let sigma = 0.02 * dist + 0.05;
  var weights = array<f32, 5>(0.2270, 0.1946, 0.1216, 0.0541, 0.0162);
  var sum = textureLoad(src, pixel, 0).r * weights[0];
  var weightSum = weights[0];
  for (var i = 1; i < 5; i++) {
    for (var side = -1; side <= 1; side += 2) {
      let tap = clamp(pixel + vec2i(0, i * side), vec2i(0), size - 1);
      let tapDist = textureLoad(sceneDepth, tap, 0).r;
      let dd = (tapDist - dist) / sigma;
      let w = weights[i] * exp(-0.5 * dd * dd) * select(0.0, 1.0, tapDist > 0.0);
      sum += textureLoad(src, tap, 0).r * w;
      weightSum += w;
    }
  }
  let visibility = sum / weightSum;
  let share = textureLoad(sceneAux, pixel, 0).w;
  return vec4f(color.rgb * (1.0 - share * (1.0 - visibility)), color.a);
}
