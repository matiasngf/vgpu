import { describe, expect, it } from 'vitest';
import { pack } from './cad';
import { buildEngine, buildGround, buildStand, DEFAULT_ENGINE, engineToStand, MAT_DECAL } from './engine';

function bounds(positions: Float32Array) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a]!, positions[i + a]!);
      max[a] = Math.max(max[a]!, positions[i + a]!);
    }
  }
  return { min, max };
}

describe('parametric engine', () => {
  it('has Merlin-like proportions in exit-radius units', () => {
    const engine = pack(buildEngine(DEFAULT_ENGINE));
    const { min, max } = bounds(engine.positions);
    expect(min[1]).toBeCloseTo(0, 5);           // exit plane at y = 0
    expect(max[1]).toBeGreaterThan(4.5);         // head stack well above the chamber
    expect(max[1]).toBeLessThan(6);
    expect(Math.max(-min[0]!, max[0]!)).toBeLessThan(1.5); // no part wider than 1.5 exit radii
    expect(engine.indices.length / 3).toBeGreaterThan(10_000);
    for (const index of engine.indices) expect(index).toBeLessThan(engine.vertexCount);
    for (const m of engine.materials) expect(m).toBeGreaterThanOrEqual(0);
    for (const m of engine.materials) expect(m).toBeLessThanOrEqual(MAT_DECAL);
  });

  it('lies on its side on the stand with the exit at the origin', () => {
    const axisHeight = 1.7;
    const engine = pack(engineToStand(buildEngine(DEFAULT_ENGINE), axisHeight));
    const { min, max } = bounds(engine.positions);
    expect(max[0]).toBeCloseTo(0.1, 0);          // exit lip near x = 0
    expect(min[0]).toBeLessThan(-4.5);           // head toward -X
    expect(min[1]).toBeGreaterThan(0.2);         // clears the pad
    expect((min[1]! + max[1]!) / 2).toBeCloseTo(axisHeight, 0);
  });

  it('builds a stand and ground that sit on the pad', () => {
    const stand = pack(buildStand(DEFAULT_ENGINE, 1.7));
    const ground = pack(buildGround());
    expect(bounds(stand.positions).min[1]).toBeCloseTo(0, 5);
    expect(bounds(ground.positions).max[1]).toBeCloseTo(0, 5);
    for (const index of stand.indices) expect(index).toBeLessThan(stand.vertexCount);
  });
});
