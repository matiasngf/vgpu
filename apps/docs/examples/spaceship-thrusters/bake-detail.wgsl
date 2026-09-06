import { DETAIL_SIZE, DETAIL_PERIOD, pnoise2 } from "./thruster-common.wgsl";

// Bakes a tileable 2D detail texture (sampled with a repeat sampler). Runs once.
// r: 6-octave fbm      — fine grain along the plume surface
// g: 5-octave ridged   — wispy filaments at the shear layer
// b: 2-octave warp scalar
// a: ridged stack — the g channel plus the same ridged field at 3x2 and 6x4
//    (integer multiples keep it tileable), so the fire's fibre lookup is one
//    fetch instead of three

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
  // Ridged field at the two finer anisotropic scales used by the fire fibres.
  var stack = 0.0;
  for (var k = 0; k < 2; k++) {
    let scaleUv = select(vec2f(3.0, 2.0), vec2f(6.0, 4.0), k == 1);
    let scalePeriod = select(vec2i(3, 2), vec2i(6, 4), k == 1);
    let offset = select(vec2f(0.37, 0.11), vec2f(0.71, 0.53), k == 1);
    var r2 = 0.0;
    var amp2 = 0.5;
    var freq2 = 1;
    var weight2 = 1.0;
    for (var o = 0; o < 5; o++) {
      let q = (p * scaleUv + offset) * vec2f(period * freq2);
      let r = 1.0 - abs(pnoise2(q, period * scalePeriod * freq2, 0xabcdu + u32(o) * 104729u));
      r2 += amp2 * r * r * weight2;
      weight2 = clamp(r * r * 1.5, 0.0, 1.0);
      amp2 *= 0.5;
      freq2 *= 2;
    }
    stack += clamp(r2 * 1.2, 0.0, 1.0) * select(0.35, 0.22, k == 1);
  }
  let ridgedOut = clamp(ridged * 1.2, 0.0, 1.0);
  return vec4f(
    clamp(0.5 + 0.5 * fbm * 3.0, 0.0, 1.0),
    ridgedOut,
    clamp(0.5 + 0.5 * warpX * 2.0, 0.0, 1.0),
    clamp(ridgedOut * 0.43 + stack, 0.0, 1.0),
  );
}
