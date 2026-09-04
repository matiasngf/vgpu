import { DETAIL_SIZE, DETAIL_PERIOD, pnoise2 } from "./thruster-common.wgsl";

// Bakes a tileable 2D detail texture (sampled with a repeat sampler). Runs once.
// r: 6-octave fbm      — fine grain along the plume surface
// g: 5-octave ridged   — wispy filaments at the shear layer
// ba: 2-octave warp vector

@fragment fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let p = position.xy / DETAIL_SIZE;
  let period = vec2i(DETAIL_PERIOD);

  var fbm = 0.0;
  var ridged = 0.0;
  var amp = 0.5;
  var freq = 1;
  var weight = 1.0;
  for (var o = 0; o < 6; o++) {
    let q = p * vec2f(period * freq);
    fbm += amp * pnoise2(q, period * freq, 0x2468u + u32(o) * 7919u);
    if (o < 5) {
      let r = 1.0 - abs(pnoise2(q + vec2f(0.5, 0.25), period * freq, 0xabcdu + u32(o) * 104729u));
      ridged += amp * r * r * weight;
      weight = clamp(r * r * 1.5, 0.0, 1.0);
    }
    amp *= 0.5;
    freq *= 2;
  }
  let warpX = pnoise2(p * vec2f(period), period, 0x1111u) + 0.5 * pnoise2(p * vec2f(period * 2), period * 2, 0x2222u);
  let warpY = pnoise2(p * vec2f(period), period, 0x3333u) + 0.5 * pnoise2(p * vec2f(period * 2), period * 2, 0x4444u);
  return vec4f(
    clamp(0.5 + 0.5 * fbm * 3.0, 0.0, 1.0),
    clamp(ridged * 1.2, 0.0, 1.0),
    clamp(0.5 + 0.5 * warpX * 2.0, 0.0, 1.0),
    clamp(0.5 + 0.5 * warpY * 2.0, 0.0, 1.0),
  );
}
