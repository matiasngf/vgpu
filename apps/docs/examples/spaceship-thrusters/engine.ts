// Parametric rocket engine (Merlin-1D style) built from the CAD kit.
//
// Units: nozzle EXIT RADIUS = 1. The engine is modelled upright along +Y with
// the nozzle exit plane at y = 0 and the turbopump at the top, matching the
// reference photo; the scene rotates it onto its side for the test stand.
//
// Material ids (see scene.wgsl):
//   0 matte black (nozzle, chamber, insulated lines)
//   1 dark steel (turbopump housings, flanges)
//   2 stainless (small lines, valves)
//   3 concrete pad   4 gravel   5 painted steel (stand)   6 white decal

import {
  add, box, compose, cylinder, disc, emptyMesh, merge, revolve, rotationX, rotationY, rotationZ, scale as scaleVec, torus, transform,
  translation, tube, type CadMesh, type ProfilePoint, type Vec3,
} from './cad';

export const MAT_BLACK = 0;
export const MAT_STEEL = 1;
export const MAT_STAINLESS = 2;
export const MAT_CONCRETE = 3;
export const MAT_GRAVEL = 4;
export const MAT_PAINT = 5;
export const MAT_DECAL = 6;

export interface EngineParams {
  /** Radius at the throat (narrowest point). */
  throatRadius: number;
  /** Axial distance from the exit plane to the throat. */
  nozzleLength: number;
  /** Regenerative chamber radius and its top. */
  chamberRadius: number;
  chamberTop: number;
  /** Nozzle wall thickness (a thin shell, visible at the lip). */
  wallThickness: number;
  segments: number;
}

export const DEFAULT_ENGINE: EngineParams = {
  throatRadius: 0.29,
  nozzleLength: 1.55,
  chamberRadius: 0.46,
  chamberTop: 2.75,
  wallThickness: 0.035,
  segments: 96,
};

/** Parabolic bell contour from the exit (y = 0) up to the throat. */
function bellProfile(p: EngineParams, offset = 0): ProfilePoint[] {
  const points: ProfilePoint[] = [];
  const n = 14;
  for (let i = 0; i <= n; i++) {
    const t = i / n; // 0 at exit, 1 at throat
    // Radius shrinks quickly near the exit and slowly near the throat:
    // r(t) = throat + (1 - throat) * (1 - t)^1.7 keeps the flare of a bell.
    const r = p.throatRadius + (1 - p.throatRadius) * Math.pow(1 - t, 1.7);
    points.push({ y: t * p.nozzleLength, r: r + offset });
  }
  return points;
}

/** Converging section + cylindrical chamber above the throat. */
function chamberProfile(p: EngineParams, offset = 0): ProfilePoint[] {
  const throatY = p.nozzleLength;
  const conv = throatY + 0.32;
  return [
    { y: throatY, r: p.throatRadius + offset },
    { y: throatY + 0.12, r: p.throatRadius * 1.15 + offset },
    { y: conv, r: p.chamberRadius + offset },
    { y: p.chamberTop, r: p.chamberRadius + offset },
  ];
}

export function buildEngine(p: EngineParams = DEFAULT_ENGINE): CadMesh {
  const mesh = emptyMesh();
  const seg = p.segments;

  // Nozzle: outer shell, inner shell (flipped) and the lip ring joining them.
  const outer = revolve([...bellProfile(p), ...chamberProfile(p).slice(1)], seg, MAT_BLACK);
  const innerProfile = [...bellProfile(p, -p.wallThickness), ...chamberProfile(p, -p.wallThickness).slice(1)];
  const inner = flip(revolve(innerProfile, seg, MAT_BLACK));
  merge(mesh, outer, inner, annulus(1 - p.wallThickness, 1, 0, seg, MAT_BLACK, false));

  // Chamber top flange and injector dome.
  merge(mesh, cylinder(p.chamberRadius + 0.08, p.chamberRadius + 0.08, p.chamberTop, p.chamberTop + 0.12, seg, MAT_STEEL));
  merge(mesh, revolve([
    { y: p.chamberTop + 0.12, r: p.chamberRadius + 0.02 },
    { y: p.chamberTop + 0.3, r: p.chamberRadius * 0.85 },
    { y: p.chamberTop + 0.42, r: p.chamberRadius * 0.55 },
    { y: p.chamberTop + 0.46, r: 0.0 },
  ], seg, MAT_STEEL));

  // Manifold rings: a thick one just below the throat, a thinner one at the
  // top of the chamber (regen inlet/outlet manifolds).
  merge(mesh, torus(p.throatRadius + 0.24, 0.13, p.nozzleLength + 0.05, seg, 20, MAT_BLACK));
  merge(mesh, torus(p.chamberRadius + 0.12, 0.09, p.chamberTop - 0.15, seg, 20, MAT_BLACK));

  // Main propellant lines: two thick insulated pipes rising from the throat
  // manifold, hugging the chamber, into the pump housing. A thinner third
  // line runs up the back.
  const lineR = 0.1;
  const hug = p.chamberRadius + lineR + 0.04;
  const y0 = p.nozzleLength + 0.05;
  const pumpY = p.chamberTop + 0.46 + 0.72;
  merge(mesh, tube(helixPath(p.throatRadius + 0.24 + lineR + 0.02, y0, y0 + 0.7, Math.PI * 0.1, Math.PI * 0.62, 8).concat([
    [Math.cos(Math.PI * 0.62) * hug, y0 + 1.0, Math.sin(Math.PI * 0.62) * hug],
    [Math.cos(Math.PI * 0.62) * hug, pumpY - 0.1, Math.sin(Math.PI * 0.62) * hug],
    [Math.cos(Math.PI * 0.62) * 0.3, pumpY - 0.1, Math.sin(Math.PI * 0.62) * 0.3],
  ]), lineR, 0.28, 20, MAT_BLACK));
  merge(mesh, tube(helixPath(p.throatRadius + 0.24 + lineR + 0.02, y0 + 0.08, y0 + 0.75, Math.PI * 1.15, Math.PI * 1.7, 8).concat([
    [Math.cos(Math.PI * 1.7) * hug, y0 + 1.1, Math.sin(Math.PI * 1.7) * hug],
    [Math.cos(Math.PI * 1.7) * hug, pumpY + 0.05, Math.sin(Math.PI * 1.7) * hug],
    [Math.cos(Math.PI * 1.7) * 0.3, pumpY + 0.05, Math.sin(Math.PI * 1.7) * 0.3],
  ]), lineR, 0.28, 20, MAT_BLACK));
  merge(mesh, tube([
    [Math.cos(Math.PI * 1.0) * (hug - 0.03), p.chamberTop - 0.15, Math.sin(Math.PI * 1.0) * (hug - 0.03)],
    [Math.cos(Math.PI * 1.0) * (hug - 0.03), pumpY + 0.9, Math.sin(Math.PI * 1.0) * (hug - 0.03)],
    [Math.cos(Math.PI * 1.0) * 0.25, pumpY + 0.9, Math.sin(Math.PI * 1.0) * 0.25],
  ], 0.06, 0.15, 14, MAT_BLACK));

  // Turbopump assembly on top of the dome.
  const pumpBase = p.chamberTop + 0.46;
  // Gearbox / pump body: a squat vertical cylinder.
  merge(mesh, cylinder(0.42, 0.42, pumpBase, pumpBase + 0.4, seg, MAT_STEEL));
  merge(mesh, torus(0.44, 0.05, pumpBase + 0.02, seg, 12, MAT_STEEL));
  merge(mesh, torus(0.44, 0.05, pumpBase + 0.38, seg, 12, MAT_STEEL));
  // Horizontal pump housing (LOX / fuel impellers) across the top, with the
  // turbine housing at +X and the fuel pump volute and inducer at -X.
  merge(mesh, transform(cylinder(0.34, 0.34, -0.72, 0.72, 48, MAT_STEEL), compose(translation([0, pumpBase + 0.72, 0]), rotationZ(Math.PI / 2))));
  merge(mesh, transform(cylinder(0.42, 0.42, -0.85, -0.6, 48, MAT_STEEL), compose(translation([0, pumpBase + 0.72, 0]), rotationZ(Math.PI / 2))));
  merge(mesh, transform(cylinder(0.3, 0.16, -1.1, -0.85, 40, MAT_STEEL), compose(translation([0, pumpBase + 0.72, 0]), rotationZ(Math.PI / 2))));
  merge(mesh, transform(cylinder(0.44, 0.44, 0.62, 0.9, 48, MAT_STEEL), compose(translation([0, pumpBase + 0.72, 0]), rotationZ(Math.PI / 2))));
  merge(mesh, transform(torus(0.46, 0.04, 0.9, 48, 10, MAT_STEEL), compose(translation([0, pumpBase + 0.72, 0]), rotationZ(Math.PI / 2))));
  // Gearbox in front (+Z) of the pump body.
  merge(mesh, transform(cylinder(0.3, 0.3, 0, 0.42, 40, MAT_STEEL), compose(translation([0.05, pumpBase + 0.72, 0.28]), rotationX(Math.PI / 2))));
  // Upper stack: gas generator + valves.
  merge(mesh, cylinder(0.24, 0.24, pumpBase + 1.0, pumpBase + 1.35, 48, MAT_STEEL));
  merge(mesh, cylinder(0.3, 0.3, pumpBase + 1.35, pumpBase + 1.45, 48, MAT_STEEL));
  merge(mesh, cylinder(0.2, 0.2, pumpBase + 1.45, pumpBase + 1.8, 48, MAT_STAINLESS));
  merge(mesh, cylinder(0.24, 0.24, pumpBase + 1.8, pumpBase + 1.9, 48, MAT_STEEL));
  // Small valves and actuators around the pump.
  const valve = (x: number, z: number, y: number, r: number, h: number, mat: number) =>
    merge(mesh, transform(cylinder(r, r, 0, h, 24, mat), translation([x, y, z])));
  valve(-0.25, 0.45, pumpBase + 0.4, 0.11, 0.5, MAT_STAINLESS);
  valve(0.2, -0.5, pumpBase + 0.4, 0.13, 0.42, MAT_STEEL);
  valve(-0.45, -0.3, pumpBase + 1.0, 0.09, 0.35, MAT_STAINLESS);
  valve(0.4, 0.3, pumpBase + 1.0, 0.1, 0.3, MAT_STEEL);
  // Gas-generator exhaust duct: a big elbow leaving the pump sideways and down.
  merge(mesh, tube([
    [0.45, pumpBase + 1.15, 0.2],
    [0.95, pumpBase + 1.15, 0.2],
    [0.95, pumpBase + 0.2, 0.45],
    [0.8, p.chamberTop - 0.4, 0.8],
  ], 0.12, 0.25, 20, MAT_BLACK));
  // Thin stainless lines.
  merge(mesh, tube([[-0.2, pumpBase + 0.9, 0.4], [-0.55, pumpBase + 0.9, 0.4], [-0.55, pumpBase + 1.7, 0.1], [-0.15, pumpBase + 1.7, 0.1]], 0.035, 0.1, 12, MAT_STAINLESS));
  merge(mesh, tube([[0.3, pumpBase + 1.5, -0.2], [0.3, pumpBase + 1.5, -0.55], [-0.3, pumpBase + 0.5, -0.55]], 0.035, 0.1, 12, MAT_STAINLESS));

  // White "1" decal: a flat strip on the nozzle side.
  merge(mesh, decalOne(p));
  return mesh;
}

/** Points on a helix around +Y between two angles (radians). */
function helixPath(radius: number, y0: number, y1: number, a0: number, a1: number, steps: number): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const a = a0 + (a1 - a0) * t;
    out.push([Math.cos(a) * radius, y0 + (y1 - y0) * t, Math.sin(a) * radius]);
  }
  return out;
}

/** Flat ring between two radii at height y. */
function annulus(r0: number, r1: number, y: number, segments: number, material: number, up: boolean): CadMesh {
  const mesh = emptyMesh();
  const ny = up ? 1 : -1;
  for (let j = 0; j <= segments; j++) {
    const a = (j / segments) * Math.PI * 2;
    const c = Math.cos(a), s = Math.sin(a);
    mesh.positions.push(r0 * c, y, r0 * s, r1 * c, y, r1 * s);
    mesh.normals.push(0, ny, 0, 0, ny, 0);
    mesh.uvs.push(j / segments, 0, j / segments, 1);
    mesh.materials.push(material, material);
  }
  for (let j = 0; j < segments; j++) {
    const a = j * 2;
    if (up) mesh.indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    else mesh.indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  return mesh;
}

/** Reverses winding and normals (for inner surfaces). */
function flip(mesh: CadMesh): CadMesh {
  const out: CadMesh = { ...mesh, indices: [], normals: mesh.normals.map((n) => -n) };
  for (let i = 0; i < mesh.indices.length; i += 3) out.indices.push(mesh.indices[i]!, mesh.indices[i + 2]!, mesh.indices[i + 1]!);
  return out;
}

/** A "1" painted on the bell: thin quads floating just off the surface. */
function decalOne(p: EngineParams): CadMesh {
  const mesh = emptyMesh();
  const profile = bellProfile(p, 0.012);
  const yCenter = 0.55, height = 0.5, width = 0.08;
  const angle = 0; // faces +X
  const strip = (yA: number, yB: number, halfW: number, shift: number) => {
    const rAt = (y: number) => {
      for (let i = 1; i < profile.length; i++) {
        if (profile[i]!.y >= y) {
          const a = profile[i - 1]!, b = profile[i]!;
          return a.r + (b.r - a.r) * ((y - a.y) / (b.y - a.y));
        }
      }
      return profile[profile.length - 1]!.r;
    };
    const base = mesh.positions.length / 3;
    for (const [y, sz] of [[yA, -1], [yA, 1], [yB, 1], [yB, -1]] as const) {
      const r = rAt(y);
      const z = sz * halfW + shift;
      mesh.positions.push(r * Math.cos(angle), y, z);
      mesh.normals.push(1, 0, 0);
      mesh.uvs.push(0, 0);
      mesh.materials.push(MAT_DECAL);
    }
    mesh.indices.push(base, base + 2, base + 1, base, base + 3, base + 2);
  };
  strip(yCenter - height / 2, yCenter + height / 2, width / 2, 0);        // stem
  strip(yCenter + height / 2 - 0.09, yCenter + height / 2, width / 2, -0.09); // serif flag
  strip(yCenter - height / 2, yCenter - height / 2 + 0.07, 0.14, 0);       // foot
  return mesh;
}

export interface StandParams {
  /** Height of the engine axis above the pad. */
  axisHeight: number;
  /** Engine length from exit plane to the top of the pump stack. */
  engineLength: number;
}

/**
 * Test stand: the engine lies on its side (axis +X, nozzle pointing +X, exit
 * at the origin) on a steel base frame with a thrust plate at the head end.
 */
export function buildStand(engineLength: number, axisHeight: number): CadMesh {
  const mesh = emptyMesh();
  const headX = -engineLength;
  const frameLength = engineLength + 1.8;
  const frameCenter = headX - 0.6 + frameLength / 2;
  // Longitudinal I-beams and cross members.
  for (const z of [-1.2, 1.2]) merge(mesh, iBeam([frameCenter, 0, z], frameLength, 0.36, 0.3, 'x'));
  for (const x of [headX - 1.2, headX + 1.4, 0.4]) merge(mesh, iBeam([x, 0, 0], 2.7, 0.36, 0.3, 'z'));
  // Thrust plate (vertical) with a stiffening frame and hold-down bolts.
  const plateHeight = axisHeight * 2 + 0.4;
  merge(mesh, box([headX - 0.9, plateHeight / 2, 0], [0.22, plateHeight, 3.0], MAT_PAINT));
  for (const z of [-1.35, 1.35]) merge(mesh, box([headX - 0.9, plateHeight / 2, z], [0.34, plateHeight, 0.28], MAT_PAINT));
  merge(mesh, box([headX - 0.9, plateHeight + 0.14, 0], [0.34, 0.28, 3.0], MAT_PAINT));
  // Rear anchor block and two gusset webs bracing the plate.
  merge(mesh, box([headX - 1.6, 0.25, 0], [1.1, 0.5, 3.2], MAT_PAINT));
  for (const z of [-1.25, 1.25]) merge(mesh, box([headX - 1.35, axisHeight * 0.6, z], [0.6, axisHeight * 1.2, 0.1], MAT_PAINT));
  for (const z of [-1.0, -0.4, 0.4, 1.0]) merge(mesh, transform(cylinder(0.08, 0.08, 0, 0.16, 12, MAT_STAINLESS), translation([headX - 1.6, 0.5, z])));
  // Gimbal block joining the plate to the engine head.
  merge(mesh, transform(cylinder(0.55, 0.55, 0, 0.5, 48, MAT_STEEL), compose(translation([headX - 0.75, axisHeight, 0]), rotationZ(-Math.PI / 2))));
  merge(mesh, transform(cylinder(0.32, 0.32, 0, 0.45, 32, MAT_STEEL), compose(translation([headX - 0.3, axisHeight, 0]), rotationZ(-Math.PI / 2))));
  // Cradle under the chamber: two saddle posts and a cross bar.
  for (const z of [-0.7, 0.7]) merge(mesh, box([headX + 2.2, axisHeight * 0.45, z], [0.22, axisHeight * 0.9, 0.22], MAT_PAINT));
  merge(mesh, box([headX + 2.2, axisHeight * 0.9 + 0.05, 0], [0.26, 0.12, 1.7], MAT_PAINT));
  // Feed lines: run low along both sides of the frame from the plate to the
  // pump inlets, staying below the engine so the engine stays visible.
  const pumpX = headX + 1.35;
  merge(mesh, tube([
    [headX - 1.0, 0.55, 1.55], [pumpX - 0.4, 0.55, 1.55], [pumpX - 0.4, axisHeight + 0.3, 1.55], [pumpX, axisHeight + 0.3, 0.95],
  ], 0.09, 0.25, 16, MAT_STAINLESS));
  merge(mesh, tube([
    [headX - 1.0, 0.55, -1.55], [pumpX + 0.2, 0.55, -1.55], [pumpX + 0.2, axisHeight - 0.2, -1.55], [pumpX + 0.5, axisHeight - 0.2, -0.95],
  ], 0.09, 0.25, 16, MAT_STAINLESS));
  // Cable tray along the near beam.
  merge(mesh, box([frameCenter, 0.45, 1.5], [frameLength - 0.6, 0.12, 0.35], MAT_STEEL));
  return mesh;
}

/** I-beam (two flanges + web) centred on `center`, running along `axis`. */
function iBeam(center: Vec3, length: number, height: number, width: number, axis: 'x' | 'z'): CadMesh {
  const mesh = emptyMesh();
  const flange = 0.05, web = 0.06;
  const size = (l: number, h: number, w: number): Vec3 => (axis === 'x' ? [l, h, w] : [w, h, l]);
  merge(mesh, box([center[0], center[1] + flange / 2, center[2]], size(length, flange, width), MAT_PAINT));
  merge(mesh, box([center[0], center[1] + height - flange / 2, center[2]], size(length, flange, width), MAT_PAINT));
  merge(mesh, box([center[0], center[1] + height / 2, center[2]], size(length, height - 2 * flange, web), MAT_PAINT));
  return mesh;
}

/** Concrete pad plus a wide gravel apron; the shader adds the grain. */
export function buildGround(): CadMesh {
  const mesh = emptyMesh();
  merge(mesh, box([6, -0.05, 0], [30, 0.1, 18], MAT_CONCRETE));
  merge(mesh, box([6, -0.12, 0], [140, 0.1, 140], MAT_GRAVEL));
  return mesh;
}

/** Places the upright engine on its side: +Y (head) -> -X, exit at the origin, lifted to axisHeight. */
export function engineToStand(engine: CadMesh, axisHeight: number): CadMesh {
  return transform(engine, compose(translation([0, axisHeight, 0]), rotationZ(Math.PI / 2)));
}

export { add, scaleVec, disc, rotationX, rotationY };
