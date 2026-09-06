struct Composite {
  exposure: f32,
  bloomStrength: f32,
  grain: f32,
  time: f32,
  skyColor: vec3f,
}

// Depth-aware (bilateral) upsample of the half-resolution plume: the four
// nearest plume texels are weighted bilinearly, but a texel whose clip distance
// disagrees with this pixel's surface is discarded so the plume does not bleed
// over the edge of the bell.
fn upsampleFire(pixel: vec2f, surfaceDistance: f32) -> vec4f {
  let fireSize = vec2f(textureDimensions(fire));
  let sceneSize = vec2f(textureDimensions(scene));
  let coord = pixel * (fireSize / sceneSize) - 0.5;
  let base = vec2i(floor(coord));
  let f = fract(coord);
  var sum = vec4f(0.0);
  var weightSum = 0.0;
  var fallback = vec4f(0.0);
  for (var k = 0; k < 4; k++) {
    let offset = vec2i(k & 1, k >> 1);
    let texel = clamp(base + offset, vec2i(0), vec2i(fireSize) - 1);
    let w = select(1.0 - f.x, f.x, offset.x == 1) * select(1.0 - f.y, f.y, offset.y == 1);
    let sample = textureLoad(fire, texel, 0);
    let clipDistance = textureLoad(fireAux, texel, 0).z;
    let agree = abs(clipDistance - surfaceDistance) < 0.35 + 0.02 * surfaceDistance;
    fallback += sample * w;
    if (agree) { sum += sample * w; weightSum += w; }
  }
  return select(fallback, sum / weightSum, weightSum > 0.05);
}

@group(0) @binding(0) var fire: texture_2d<f32>;      // premultiplied plume, half resolution
@group(0) @binding(1) var fireAux: texture_2d<f32>;   // haze offset, clip distance, shimmer
@group(0) @binding(2) var scene: texture_2d<f32>;     // lit geometry, full resolution
@group(0) @binding(3) var sceneDepth: texture_2d<f32>;
@group(0) @binding(4) var bloom: texture_2d<f32>;
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var<uniform> composite: Composite;

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += vec2f(dot(q, q + vec2f(45.32)));
  return fract(q.x * q.y);
}

@fragment fn fs_main(@builtin(position) position: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
  // Background: the lit scene, refracted by the plume's heat haze, or a flat
  // dusk sky where nothing was drawn.
  let pixel = vec2i(position.xy);
  let surfaceDistance = textureLoad(sceneDepth, pixel, 0).r;
  let hasSurface = surfaceDistance > 0.0;
  let aux = textureSampleLevel(fireAux, samp, uv, 0.0);
  let hazePixel = clamp(pixel + vec2i(aux.xy), vec2i(0), vec2i(textureDimensions(scene)) - 1);
  let hazeSurface = textureLoad(sceneDepth, hazePixel, 0).r > 0.0;
  var background = select(composite.skyColor, textureLoad(scene, select(pixel, hazePixel, hazeSurface), 0).rgb, hasSurface);
  // Exit gas over the sky reads as hot glass: slightly brighter and cooler.
  let shimmer = aux.w * select(1.0, 0.0, hasSurface);
  background = mix(background, background * vec3f(1.15, 1.2, 1.25) + vec3f(0.03, 0.05, 0.06), shimmer);

  // Plume over background. Radiance is scene-linear; halation from the bloom
  // chain is added with the red bias film shows, then the sensor response
  // soft-clips each channel independently: an orange core saturates R first,
  // then G, then B, which is what turns the hottest part of a flame white.
  let plume = upsampleFire(position.xy, surfaceDistance);
  let radiance = plume.rgb + plume.a * background;
  let halation = textureSampleLevel(bloom, samp, uv, 0.0).rgb * vec3f(1.0, 0.85, 0.78);
  let exposed = (radiance + halation * composite.bloomStrength) * composite.exposure;
  var color = vec3f(1.0) - exp(-exposed);

  // Gentle film-style contrast: lift mids slightly, keep the toe.
  color = mix(color, color * color * (3.0 - 2.0 * color), 0.35);

  let centered = uv - vec2f(0.5);
  let vignette = 1.0 - smoothstep(0.5, 1.2, length(centered) * 1.55);
  color *= mix(0.72, 1.0, vignette);

  color = pow(color, vec3f(1.0 / 2.2));
  color += (hash21(uv * 1024.0 + fract(composite.time) * 17.0) - 0.5) * composite.grain;
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
