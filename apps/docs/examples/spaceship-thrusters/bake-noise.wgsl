import { TILE, BORDER, STRIDE, COLS, SLICES, PERIOD_XY, PERIOD_Z, pnoise3 } from "./thruster-common.wgsl";

// Bakes a tileable 3D noise volume into a 2D slice atlas. Runs once.
// r: 5-octave fbm       (signed noise remapped to [0, 1])
// g: 4-octave billow    (|noise| sum, cauliflower-like lumps)
// b: 4-octave ridged    (1 - |noise|, sharp filaments)
// a: 1-octave low-frequency noise for domain warping

@fragment fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let px = vec2i(position.xy);
  let stride = i32(STRIDE);
  let tile = px / stride;
  let local = vec2f(px - tile * stride) - BORDER; // -1 .. TILE
  let slice = tile.y * COLS + tile.x;
  let p = vec3f(local / TILE, f32(slice) / f32(SLICES));
  let period = vec3i(PERIOD_XY, PERIOD_XY, PERIOD_Z);

  var fbm = 0.0;
  var billow = 0.0;
  var ridged = 0.0;
  var amp = 0.5;
  var freq = 1;
  var ridgeWeight = 1.0;
  for (var o = 0; o < 5; o++) {
    let q = p * vec3f(period * freq);
    let n = pnoise3(q, period * freq, 0x1234u + u32(o) * 7919u);
    fbm += amp * n;
    if (o < 4) {
      billow += amp * abs(pnoise3(q + vec3f(0.37, 0.11, 0.73), period * freq, 0x5151u + u32(o) * 104729u));
      let r = 1.0 - abs(pnoise3(q + vec3f(0.21, 0.83, 0.42), period * freq, 0x9e37u + u32(o) * 15485863u));
      ridged += amp * r * r * ridgeWeight;
      ridgeWeight = clamp(r * r, 0.0, 1.0);
    }
    amp *= 0.5;
    freq *= 2;
  }
  let low = pnoise3(p * vec3f(period), period, 0x7777u);
  return vec4f(
    clamp(0.5 + 0.5 * fbm * 3.0, 0.0, 1.0),
    clamp(billow * 2.6, 0.0, 1.0),
    clamp(ridged * 1.1, 0.0, 1.0),
    clamp(0.5 + 0.5 * low * 2.4, 0.0, 1.0),
  );
}
