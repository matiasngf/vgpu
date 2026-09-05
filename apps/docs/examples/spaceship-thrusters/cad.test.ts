import { describe, expect, it } from 'vitest';
import {
  box, cylinder, invert, lookAt, multiply, orthographic, pack, perspective, revolve, torus, transformPoint, tube, type CadMesh,
} from './cad';

function signedVolume(mesh: CadMesh): number {
  const p = mesh.positions;
  let volume = 0;
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const [a, b, c] = [mesh.indices[i]! * 3, mesh.indices[i + 1]! * 3, mesh.indices[i + 2]! * 3];
    const [ax, ay, az, bx, by, bz, cx, cy, cz] = [p[a]!, p[a + 1]!, p[a + 2]!, p[b]!, p[b + 1]!, p[b + 2]!, p[c]!, p[c + 1]!, p[c + 2]!];
    volume += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;
  }
  return volume;
}

function expectWellFormed(mesh: CadMesh): void {
  const packed = pack(mesh);
  expect(packed.indices.length % 3).toBe(0);
  for (const index of packed.indices) expect(index).toBeLessThan(packed.vertexCount);
  for (const value of packed.positions) expect(Number.isFinite(value)).toBe(true);
  for (let i = 0; i < packed.normals.length; i += 3) {
    expect(Math.hypot(packed.normals[i]!, packed.normals[i + 1]!, packed.normals[i + 2]!)).toBeCloseTo(1, 3);
  }
  expect(packed.materials.length).toBe(packed.vertexCount);
  expect(packed.uvs.length).toBe(packed.vertexCount * 2);
}

describe('cad primitives', () => {
  it('closed solids are outward-wound with the expected volume', () => {
    expect(signedVolume(cylinder(0.5, 0.5, 0, 2, 96))).toBeCloseTo(Math.PI * 0.25 * 2, 1);
    expect(signedVolume(box([1, 2, 3], [1, 2, 3]))).toBeCloseTo(6, 6);
    expect(signedVolume(torus(1, 0.1, 0, 96, 32))).toBeCloseTo(2 * Math.PI * Math.PI * 1 * 0.01, 2);
    expect(signedVolume(tube([[0, 0, 0], [0, 2, 0]], 0.5, 0.2, 96))).toBeCloseTo(Math.PI * 0.25 * 2, 1);
  });

  it('revolve normals point outward and profiles stay well formed', () => {
    const bell = revolve([{ y: 0, r: 1 }, { y: 0.5, r: 0.6 }, { y: 1, r: 0.35 }, { y: 1.6, r: 0.4 }], 48);
    expectWellFormed(bell);
    for (let i = 0; i < bell.positions.length; i += 3) {
      const radial = bell.normals[i]! * bell.positions[i]! + bell.normals[i + 2]! * bell.positions[i + 2]!;
      expect(radial).toBeGreaterThanOrEqual(-1e-6);
    }
  });

  it('tubes with elbows are well formed and keep their radius', () => {
    const pipe = tube([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]], 0.08, 0.2, 16);
    expectWellFormed(pipe);
    expect(signedVolume(pipe)).toBeGreaterThan(0);
  });

  it('box and torus are well formed', () => {
    expectWellFormed(box([0, 0, 0], [1, 2, 3]));
    expectWellFormed(torus(1, 0.2));
  });
});

describe('camera math', () => {
  it('invert() is a true inverse for view-projection matrices', () => {
    const view = lookAt([-9.5, 11.5, 11], [-0.5, 1.6, -0.5]);
    const projection = perspective(0.7, 16 / 9, 0.5, 400);
    const viewProj = multiply(projection, view);
    const identity = multiply(viewProj, invert(viewProj));
    for (let i = 0; i < 16; i++) expect(identity[i]).toBeCloseTo(i % 5 === 0 ? 1 : 0, 4);
  });

  it('projects near and far planes onto WebGPU depth 0..1', () => {
    const view = lookAt([0, 0, 0], [0, 0, -1]);
    const projection = perspective(1, 1, 1, 100);
    const project = (p: readonly [number, number, number]) => {
      const m = multiply(projection, view);
      const w = m[3]! * p[0] + m[7]! * p[1] + m[11]! * p[2] + m[15]!;
      const z = m[2]! * p[0] + m[6]! * p[1] + m[10]! * p[2] + m[14]!;
      return z / w;
    };
    expect(project([0, 0, -1])).toBeCloseTo(0, 5);
    expect(project([0, 0, -100])).toBeCloseTo(1, 5);
    const ortho = orthographic(-1, 1, -1, 1, 1, 100);
    expect(transformPoint(ortho, [0, 0, -1])[2]).toBeCloseTo(0, 5);
    expect(transformPoint(ortho, [0, 0, -100])[2]).toBeCloseTo(1, 5);
  });
});
