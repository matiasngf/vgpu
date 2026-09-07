import type { Draw, Frame, Gpu, Target } from 'vgpu';

import tileWgsl from './grass-tile.wgsl';

/** Edge of the repeating grass tile in metres. */
export const TILE_SIZE = 3;
/** Tallest blade the tile encodes, in metres. */
export const TILE_HEIGHT = 0.32;

export interface GrassTile {
  readonly target: Target;
  readonly draw: Draw;
}

export interface BladeMesh {
  /** Interleaved position(3) tangent(3) extra(2) floats. */
  readonly vertices: Float32Array<ArrayBuffer>;
  readonly indices: Uint32Array<ArrayBuffer>;
}

/**
 * Builds the tile as a two-attachment render target (colour+height, tangent+along) and the
 * draw that rasterises thousands of bent blades into it from above.
 */
export function createGrassTile(gpu: Gpu, resolution = 2048, label = 'dreamcore-grass-tile'): GrassTile {
  const target = gpu.target({
    size: [resolution, resolution],
    // 8-bit attachments so MSAA is available everywhere; the shader packs height and tangent.
    colors: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }],
    depth: true,
    msaa: 4,
    label,
  });
  const blades = buildBladeMesh(TILE_SIZE, 6500, 0x9e3779b9);
  const mesh = gpu.mesh({
    buffers: [{ attributes: { position: 'float32x3', tangent: 'float32x3', extra: 'float32x2' }, data: blades.vertices }],
    indices: blades.indices,
    label,
  });
  const draw = gpu.draw({ shader: tileWgsl, mesh, label });
  draw.set({ tile: { size: TILE_SIZE, height: TILE_HEIGHT } });
  return { target, draw };
}

/** Encodes the one-off tile rasterisation into a frame. */
export function renderGrassTile(frame: Frame, tile: GrassTile): void {
  frame.pass({ target: tile.target, clear: [0, 0, 0, 0] }, (pass) => pass.draw(tile.draw));
}

const SPINE_SAMPLES = 7;
const FLOATS_PER_VERTEX = 8;

/**
 * Blades bend sideways along +x (the comb direction) with random spread: each spine starts a
 * little off vertical and arcs over until it lies down, drifting in yaw and twisting along
 * its length. Blades that cross the tile edge are duplicated on the far side so it wraps.
 */
export function buildBladeMesh(size: number, count: number, seed: number): BladeMesh {
  const random = mulberry32(seed);
  const gaussian = () => (random() + random() + random() - 1.5) * 1.15;
  const vertices: number[] = [];
  const indices: number[] = [];
  const spine = new Float64Array(SPINE_SAMPLES * 3);
  const dirs = new Float64Array(SPINE_SAMPLES * 3);

  for (let b = 0; b < count; b++) {
    const bx = random() * size;
    const bz = random() * size;
    const yaw0 = gaussian() * 0.55;
    const curl = (random() - 0.5) * 0.7;
    const length = 0.3 + 0.3 * random() * random() + 0.1 * random();
    const width = 0.005 + 0.006 * random();
    const theta0 = 0.25 + 0.4 * random();         // from vertical at the base
    const theta1 = 1.25 + 0.5 * random();         // nearly flat at the tip
    const twist = (random() - 0.5) * 1.4;
    const bladeSeed = random();

    // Integrate the spine.
    let x = bx, y = 0, z = bz;
    let minX = bx, maxX = bx, minZ = bz, maxZ = bz;
    const step = length / (SPINE_SAMPLES - 1);
    for (let i = 0; i < SPINE_SAMPLES; i++) {
      const s = i / (SPINE_SAMPLES - 1);
      const theta = theta0 + (theta1 - theta0) * s;
      const yaw = yaw0 + curl * s;
      const dx = Math.sin(theta) * Math.cos(yaw);
      const dy = Math.cos(theta);
      const dz = Math.sin(theta) * Math.sin(yaw);
      spine[i * 3] = x; spine[i * 3 + 1] = y; spine[i * 3 + 2] = z;
      dirs[i * 3] = dx; dirs[i * 3 + 1] = dy; dirs[i * 3 + 2] = dz;
      x += dx * step; y += dy * step; z += dz * step;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
    }

    const offsets: [number, number][] = [[0, 0]];
    const ox = minX < 0 ? size : maxX > size ? -size : 0;
    const oz = minZ < 0 ? size : maxZ > size ? -size : 0;
    if (ox) offsets.push([ox, 0]);
    if (oz) offsets.push([0, oz]);
    if (ox && oz) offsets.push([ox, oz]);

    for (const [offX, offZ] of offsets) {
      const first = vertices.length / FLOATS_PER_VERTEX;
      for (let i = 0; i < SPINE_SAMPLES; i++) {
        const s = i / (SPINE_SAMPLES - 1);
        const px = spine[i * 3]! + offX, py = spine[i * 3 + 1]!, pz = spine[i * 3 + 2]! + offZ;
        const tx = dirs[i * 3]!, ty = dirs[i * 3 + 1]!, tz = dirs[i * 3 + 2]!;
        // Side vector: horizontal, perpendicular to the tangent, then twisted around it.
        let sx = tz, sy = 0, sz = -tx;
        const sl = Math.hypot(sx, sz) || 1;
        sx /= sl; sz /= sl;
        const tau = twist * s;
        const cx = ty * sz - tz * sy, cy = tz * sx - tx * sz, cz = tx * sy - ty * sx;   // tangent x side
        const c = Math.cos(tau), sn = Math.sin(tau);
        const wx = sx * c + cx * sn, wy = sy * c + cy * sn, wz = sz * c + cz * sn;
        const w = width * (1 - Math.pow(s, 1.4) * 0.93);
        vertices.push(px - wx * w, py - wy * w, pz - wz * w, tx, ty, tz, s, bladeSeed);
        vertices.push(px + wx * w, py + wy * w, pz + wz * w, tx, ty, tz, s, bladeSeed);
      }
      for (let i = 0; i < SPINE_SAMPLES - 1; i++) {
        const a = first + i * 2;
        indices.push(a, a + 1, a + 3, a, a + 3, a + 2);
      }
    }
  }
  return { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
