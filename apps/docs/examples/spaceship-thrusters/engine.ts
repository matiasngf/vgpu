// Parametric rocket engine (Merlin-1D style), test stand and pad, built from
// the CAD kit with the text-to-cad discipline: named constants first, pure
// factories, closed outward-wound solids, one assembly function per part.
//
// Units: nozzle EXIT RADIUS = 1. The engine is modelled upright along +Y with
// the nozzle exit plane at y = 0 and the turbopump at the top, matching the
// reference photo; `engineToStand` lays it on its side for the test stand
// (local +Y -> world -X, local +Z stays +Z, which is the camera side).
//
// Material ids (see scene.wgsl):
//   0 matte black (nozzle, chamber, insulated lines)   1 dark steel (housings, flanges)
//   2 stainless (small lines, valves)                  3 concrete pad   4 gravel
//   5 painted steel (stand)   6 white decal   7 safety yellow (gantry)

import {
  box, compose, cylinder, emptyMesh, merge, normalize, revolve, rotationX, rotationZ, scale as scaleVec, sub, torus, transform,
  translation, tube, type CadMesh, type ProfilePoint, type Vec3,
} from './cad';

export const MAT_BLACK = 0;
export const MAT_STEEL = 1;
export const MAT_STAINLESS = 2;
export const MAT_CONCRETE = 3;
export const MAT_GRAVEL = 4;
export const MAT_PAINT = 5;
export const MAT_DECAL = 6;
export const MAT_YELLOW = 7;

export interface EngineParams {
  /** Radius at the throat (narrowest point). 0.25 gives Merlin's expansion ratio of 16. */
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
  throatRadius: 0.25,
  nozzleLength: 1.55,
  chamberRadius: 0.46,
  chamberTop: 2.75,
  wallThickness: 0.035,
  segments: 96,
};

/** Turbopump placement in engine-local space; the stand routes feed lines to it. */
export const PUMP = {
  /** Lateral offset of the pump axis (the pump runs along local Z). */
  x: 0.15,
  z: 0.25,
  /** Height of the pump axis above the injector dome base. */
  lift: 0.72,
  halfLength: 0.72,
  /** Where the fuel inlet (+Z, turbine end) and LOX inducer (-Z) end. */
  inletPlusZ: 1.5,
  inletMinusZ: -1.25,
  radius: 0.34,
} as const;

/** Height of the dome base (top of the chamber flange + injector cap). */
export function pumpBaseY(p: EngineParams): number { return p.chamberTop + 0.49; }
export function pumpAxisY(p: EngineParams): number { return pumpBaseY(p) + PUMP.lift; }
/** Exit plane to the top of the gimbal hub. */
export function engineLength(p: EngineParams): number { return pumpBaseY(p) + 1.5; }

/**
 * Rao/parabolic bell: leaves the throat steeply (~33 deg) and flattens toward
 * the lip (~9 deg). A quadratic Bezier from the exit through a control point
 * to the throat gives that with monotonic y, so profile lookups stay simple.
 */
function bellProfile(p: EngineParams, offset = 0): ProfilePoint[] {
  const n = 20;
  const P0 = { y: 0, r: 1 }, P1 = { y: 0.55, r: 0.91 }, P2 = { y: p.nozzleLength, r: p.throatRadius };
  const points: ProfilePoint[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, u = 1 - t;
    points.push({ y: u * u * P0.y + 2 * u * t * P1.y + t * t * P2.y, r: u * u * P0.r + 2 * u * t * P1.r + t * t * P2.r + offset });
  }
  return points;
}

/** Converging section + cylindrical chamber above the throat. */
function chamberProfile(p: EngineParams, offset = 0): ProfilePoint[] {
  const throatY = p.nozzleLength;
  return [
    { y: throatY, r: p.throatRadius + offset },
    { y: throatY + 0.12, r: p.throatRadius * 1.15 + offset },
    { y: throatY + 0.32, r: p.chamberRadius + offset },
    { y: p.chamberTop, r: p.chamberRadius + offset },
  ];
}

export function buildEngine(p: EngineParams = DEFAULT_ENGINE): CadMesh {
  const mesh = emptyMesh();
  const seg = p.segments;
  const throatY = p.nozzleLength;
  const domeBase = pumpBaseY(p);
  const pumpY = pumpAxisY(p);

  // Nozzle + chamber: outer shell, inner shell (flipped) and the lip ring.
  const outer = revolve([...bellProfile(p), ...chamberProfile(p).slice(1)], seg, MAT_BLACK);
  const inner = flip(revolve([...bellProfile(p, -p.wallThickness), ...chamberProfile(p, -p.wallThickness).slice(1)], seg, MAT_BLACK));
  merge(mesh, outer, inner, annulus(1 - p.wallThickness, 1, 0, seg, MAT_BLACK, false));

  // Bolted chamber flange and a black insulated injector dome.
  merge(mesh, cylinder(p.chamberRadius + 0.08, p.chamberRadius + 0.08, p.chamberTop, p.chamberTop + 0.07, seg, MAT_STEEL));
  merge(mesh, torus(p.chamberRadius + 0.08, 0.02, p.chamberTop + 0.035, seg, 8, MAT_STEEL));
  merge(mesh, revolve([
    { y: p.chamberTop + 0.07, r: p.chamberRadius + 0.02 },
    { y: p.chamberTop + 0.2, r: p.chamberRadius },
    { y: p.chamberTop + 0.32, r: p.chamberRadius * 0.8 },
    { y: p.chamberTop + 0.42, r: p.chamberRadius * 0.45 },
    { y: domeBase, r: 0.0 },
  ], seg, MAT_BLACK));

  // Manifold rings: a slim throat ring, the chamber/nozzle joint flange and
  // the regen outlet ring at the chamber top.
  merge(mesh, torus(p.throatRadius + 0.1, 0.05, throatY + 0.04, seg, 16, MAT_BLACK));
  merge(mesh, torus(p.chamberRadius + 0.03, 0.035, throatY + 0.34, seg, 12, MAT_STEEL));
  merge(mesh, torus(p.chamberRadius + 0.06, 0.05, p.chamberTop - 0.15, seg, 16, MAT_BLACK));

  // Main propellant lines: insulated pipes leaving bosses on the throat ring,
  // running axially up the chamber into the pump body.
  const lineR = 0.1;
  const hug = p.chamberRadius + lineR + 0.04;
  const lineAngles = [Math.PI * 0.62, Math.PI * 1.7];
  const lineTops = [pumpY - 0.06, pumpY + 0.06];
  lineAngles.forEach((a, i) => {
    const radial: Vec3 = [Math.cos(a), 0, Math.sin(a)];
    const bossR = p.throatRadius + 0.1;
    merge(mesh, tube([scaleVec(radial, bossR - 0.05), scaleVec(radial, bossR + lineR + 0.05)].map((v) => [v[0], throatY + 0.04, v[2]] as Vec3), 0.13, 0, 16, MAT_BLACK));
    merge(mesh, tube([
      [radial[0] * (bossR + lineR), throatY + 0.04, radial[2] * (bossR + lineR)],
      [radial[0] * hug, throatY + 0.5, radial[2] * hug],
      [radial[0] * hug, lineTops[i]!, radial[2] * hug],
      [PUMP.x + radial[0] * 0.2, lineTops[i]!, PUMP.z + radial[2] * 0.2],
    ], lineR, 0.25, 20, MAT_BLACK));
    // Pneumatic actuator body across each line, with a small stainless can.
    const tangent: Vec3 = [-Math.sin(a), 0, Math.cos(a)];
    const centre: Vec3 = [radial[0] * hug, pumpY - 0.55, radial[2] * hug];
    merge(mesh, tube([sub(centre, scaleVec(tangent, 0.25)), [centre[0] + tangent[0] * 0.25, centre[1], centre[2] + tangent[2] * 0.25]], 0.17, 0, 24, MAT_STEEL));
    merge(mesh, transform(cylinder(0.07, 0.07, 0, 0.3, 16, MAT_STAINLESS), translation([centre[0] + tangent[0] * 0.3, centre[1] - 0.1, centre[2] + tangent[2] * 0.3])));
  });
  // Control tubing between the two actuators.
  merge(mesh, tube([
    [Math.cos(lineAngles[0]!) * hug, pumpY - 0.35, Math.sin(lineAngles[0]!) * hug],
    [0, pumpY - 0.35, 0.68],
    [Math.cos(lineAngles[1]!) * hug, pumpY - 0.35, Math.sin(lineAngles[1]!) * hug],
  ], 0.03, 0.06, 12, MAT_STAINLESS));
  // Thin back line from the regen ring into the pump.
  merge(mesh, tube([
    [-(hug - 0.03), p.chamberTop - 0.15, 0], [-(hug - 0.03), pumpY, 0], [PUMP.x - 0.2, pumpY, PUMP.z],
  ], 0.06, 0.15, 14, MAT_BLACK));

  // Turbopump: a chain of bolted housings along Z (turbine at +Z, LOX inducer
  // at -Z), offset from the engine axis like the real head.
  const pumpM = compose(translation([PUMP.x, pumpY, PUMP.z]), rotationX(Math.PI / 2)); // local Y -> +Z
  const pump = (m: CadMesh) => merge(mesh, transform(m, pumpM));
  pump(cylinder(PUMP.radius, PUMP.radius, -PUMP.halfLength, PUMP.halfLength, 48, MAT_STEEL));
  for (const s of [-0.55, -0.15, 0.2, 0.58]) pump(torus(PUMP.radius + 0.02, 0.035, s, 48, 10, MAT_STEEL));
  pump(cylinder(0.44, 0.44, 0.62, 0.9, 48, MAT_STEEL));                         // turbine housing
  pump(cylinder(0.18, 0.18, 0.9, PUMP.inletPlusZ, 32, MAT_STEEL));               // fuel inlet stub
  pump(cylinder(0.42, 0.42, -0.85, -0.6, 48, MAT_STEEL));                       // LOX pump volute
  pump(cylinder(0.3, 0.16, -1.05, -0.85, 40, MAT_STEEL));                       // inducer cone
  pump(cylinder(0.16, 0.16, PUMP.inletMinusZ, -1.05, 32, MAT_STEEL));           // LOX inlet stub
  // Discharge stubs where the main lines arrive.
  for (const s of [-0.62, 0.12]) pump(transform(cylinder(0.15, 0.15, 0, 0.45, 24, MAT_STEEL), compose(translation([0, s, 0]), rotationX(Math.PI / 2))));
  // Thrust frame: three struts from the dome flange up to the gimbal hub.
  const hubY = domeBase + 1.15;
  merge(mesh, cylinder(0.36, 0.36, hubY, hubY + 0.35, 48, MAT_STEEL));
  merge(mesh, torus(0.38, 0.03, hubY + 0.05, 48, 8, MAT_STEEL));
  for (const a of [Math.PI * 0.2, Math.PI * 0.95, Math.PI * 1.45]) {
    merge(mesh, tube([[Math.cos(a) * (p.chamberRadius + 0.05), p.chamberTop + 0.1, Math.sin(a) * (p.chamberRadius + 0.05)], [Math.cos(a) * 0.28, hubY + 0.1, Math.sin(a) * 0.28]], 0.05, 0, 12, MAT_STEEL));
  }
  // Gas-generator exhaust: the biggest pipe on the engine, leaving the
  // turbine housing and running down past the throat on the camera side.
  merge(mesh, tube([
    [PUMP.x + 0.1, pumpY, PUMP.z + 0.85],
    [PUMP.x + 0.1, pumpY, PUMP.z + 1.15],
    [0.25, throatY + 0.3, 1.2],
    [0.2, throatY - 0.45, 1.05],
  ], 0.15, 0.3, 20, MAT_BLACK));
  merge(mesh, transform(torus(0.16, 0.025, 0, 24, 8, MAT_STEEL), translation([0.2, throatY - 0.45, 1.05])));

  merge(mesh, decalOne(p));
  return mesh;
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

/** A white "1" wrapped onto the bell, facing the camera side (+Z). */
function decalOne(p: EngineParams): CadMesh {
  const mesh = emptyMesh();
  const profile = bellProfile(p, 0.012);
  const angle = Math.PI / 2;
  const rAt = (y: number) => {
    for (let i = 1; i < profile.length; i++) {
      if (profile[i]!.y >= y) {
        const a = profile[i - 1]!, b = profile[i]!;
        return a.r + (b.r - a.r) * ((y - a.y) / (b.y - a.y));
      }
    }
    return profile[profile.length - 1]!.r;
  };
  const strip = (yA: number, yB: number, halfW: number, shift: number) => {
    const base = mesh.positions.length / 3;
    for (const [y, sz] of [[yA, -1], [yA, 1], [yB, 1], [yB, -1]] as const) {
      const rr = rAt(y);
      const phi = angle + (sz * halfW + shift) / rr;
      mesh.positions.push(rr * Math.cos(phi), y, rr * Math.sin(phi));
      mesh.normals.push(Math.cos(phi), 0, Math.sin(phi));
      mesh.uvs.push(0, 0);
      mesh.materials.push(MAT_DECAL);
    }
    mesh.indices.push(base, base + 2, base + 1, base, base + 3, base + 2);
  };
  const yCenter = 0.6, height = 0.62, width = 0.09;
  strip(yCenter - height / 2, yCenter + height / 2, width / 2, 0);            // stem
  strip(yCenter + height / 2 - 0.2, yCenter + height / 2 - 0.05, 0.05, -0.1);  // flag
  strip(yCenter - height / 2, yCenter - height / 2 + 0.07, 0.15, 0);          // foot
  return mesh;
}

/** Places the upright engine on its side: local +Y (head) -> -X, exit at the origin, lifted to axisHeight. */
export function engineToStand(engine: CadMesh, axisHeight: number): CadMesh {
  return transform(engine, compose(translation([0, axisHeight, 0]), rotationZ(Math.PI / 2)));
}

/** Engine-local point -> world point on the stand (same transform as engineToStand). */
export function localToStand(local: Vec3, axisHeight: number): Vec3 {
  return [-local[1], local[0] + axisHeight, local[2]];
}

/**
 * Test stand: a low steel frame of I-beams with a thrust plate at the head
 * end, a cradle under the chamber, feed lines into the pump inlets, and a
 * cable tray.
 */
export function buildStand(p: EngineParams, axisHeight: number): CadMesh {
  const mesh = emptyMesh();
  const length = engineLength(p);
  const headX = -length;
  const frameLength = length + 1.8;
  const frameCenter = headX - 0.6 + frameLength / 2;
  // Longitudinal I-beams, cross members and foot plates.
  for (const z of [-1.2, 1.2]) merge(mesh, iBeam([frameCenter, 0, z], frameLength, 0.36, 0.3, 'x'));
  for (const x of [headX - 1.2, headX + 1.4, 0.4]) merge(mesh, iBeam([x, 0, 0], 2.7, 0.36, 0.3, 'z'));
  for (const x of [headX - 1.2, headX + 1.4, 0.4]) {
    for (const z of [-1.2, 1.2]) {
      merge(mesh, box([x, 0.02, z], [0.55, 0.04, 0.55], MAT_PAINT));
      for (const [dx, dz] of [[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]]) merge(mesh, transform(cylinder(0.045, 0.045, 0, 0.09, 10, MAT_STAINLESS), translation([x + dx!, 0.04, z + dz!])));
    }
  }
  // Thrust plate with side stiffeners, top cap and a bolt circle around the gimbal.
  const plateHeight = axisHeight * 2 - 0.2;
  const plateX = headX - 0.9;
  merge(mesh, box([plateX, plateHeight / 2, 0], [0.22, plateHeight, 2.8], MAT_PAINT));
  for (const z of [-1.25, 1.25]) merge(mesh, box([plateX, plateHeight / 2, z], [0.34, plateHeight, 0.28], MAT_PAINT));
  merge(mesh, box([plateX, plateHeight + 0.14, 0], [0.34, 0.28, 2.8], MAT_PAINT));
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    merge(mesh, transform(cylinder(0.06, 0.06, 0, 0.08, 12, MAT_STAINLESS), compose(translation([plateX + 0.11, axisHeight + Math.sin(a) * 0.7, Math.cos(a) * 0.7]), rotationZ(-Math.PI / 2))));
  }
  // Rear anchor block and four gusset webs.
  merge(mesh, box([headX - 1.6, 0.25, 0], [1.1, 0.5, 3.2], MAT_PAINT));
  for (const z of [-1.25, -0.45, 0.45, 1.25]) merge(mesh, box([headX - 1.35, axisHeight * 0.6, z], [0.6, axisHeight * 1.2, 0.1], MAT_PAINT));
  for (const z of [-1.0, -0.4, 0.4, 1.0]) merge(mesh, transform(cylinder(0.08, 0.08, 0, 0.16, 12, MAT_STAINLESS), translation([headX - 1.6, 0.5, z])));
  // Gimbal block joining the plate to the engine hub.
  merge(mesh, transform(cylinder(0.55, 0.55, 0, 0.5, 48, MAT_STEEL), compose(translation([plateX + 0.15, axisHeight, 0]), rotationZ(-Math.PI / 2))));
  merge(mesh, transform(cylinder(0.3, 0.3, 0, 0.45, 32, MAT_STEEL), compose(translation([plateX + 0.6, axisHeight, 0]), rotationZ(-Math.PI / 2))));
  // Cradle under the chamber: two saddle posts and a cross bar.
  const cradleX = -(p.nozzleLength + 0.8);
  for (const z of [-0.7, 0.7]) merge(mesh, box([cradleX, axisHeight * 0.45, z], [0.22, axisHeight * 0.9, 0.22], MAT_PAINT));
  merge(mesh, box([cradleX, axisHeight * 0.9 + 0.05, 0], [0.26, 0.12, 1.7], MAT_PAINT));
  // Feed lines: low along both sides of the frame, up into the pump inlets.
  const fuelInlet = localToStand([PUMP.x, pumpAxisY(p), PUMP.inletPlusZ], axisHeight);
  const loxInlet = localToStand([PUMP.x, pumpAxisY(p), PUMP.inletMinusZ], axisHeight);
  const feed = (inlet: Vec3, side: number, r: number) => {
    const runZ = 1.55 * side;
    merge(mesh, tube([
      [headX - 1.0, 0.55, runZ], [inlet[0], 0.55, runZ], [inlet[0], inlet[1], runZ], [inlet[0], inlet[1], inlet[2] - 0.12 * side],
    ], r, 0.25, 16, MAT_STAINLESS));
    // Flange and bellows just before the inlet.
    for (let i = 0; i < 3; i++) merge(mesh, transform(torus(r + 0.03, 0.025, 0, 24, 8, MAT_STEEL), compose(translation([inlet[0], inlet[1], inlet[2] + (0.14 + 0.08 * i) * side]), rotationX(Math.PI / 2))));
    merge(mesh, transform(cylinder(r + 0.07, r + 0.07, 0, 0.06, 24, MAT_STEEL), compose(translation([inlet[0], inlet[1], inlet[2] + 0.04 * side]), rotationX(Math.PI / 2))));
  };
  feed(fuelInlet, 1, 0.09);
  feed(loxInlet, -1, 0.11);
  // Cable tray (U-channel) along the near beam with a few cables.
  merge(mesh, box([frameCenter, 0.4, 1.62], [frameLength - 0.6, 0.02, 0.3], MAT_STEEL));
  for (const z of [1.48, 1.76]) merge(mesh, box([frameCenter, 0.46, z], [frameLength - 0.6, 0.12, 0.02], MAT_STEEL));
  for (const z of [1.54, 1.62, 1.7]) merge(mesh, tube([[headX - 0.8, 0.44, z], [frameCenter + frameLength / 2 - 0.4, 0.44, z]], 0.02, 0, 8, MAT_BLACK));
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

/** A yellow lifting gantry in the background for scale and a colour accent. */
export function buildGantry(): CadMesh {
  const mesh = emptyMesh();
  const base: Vec3 = [-9, 0, -17.5];
  for (const z of [-1.5, 1.5]) merge(mesh, box([base[0], 2.25, base[2] + z], [0.4, 4.5, 0.4], MAT_YELLOW));
  merge(mesh, box([base[0], 4.7, base[2]], [0.4, 0.4, 3.4], MAT_YELLOW));
  merge(mesh, box([base[0] + 2.5, 4.7, base[2]], [5.4, 0.3, 0.3], MAT_YELLOW));
  merge(mesh, tube([[base[0] + 4.5, 4.55, base[2]], [base[0] + 4.5, 2.6, base[2]], [base[0] + 4.2, 2.3, base[2]]], 0.05, 0.15, 10, MAT_STEEL));
  return mesh;
}

/** Concrete pad plus a wide gravel apron; the shader adds the grain. */
export function buildGround(): CadMesh {
  const mesh = emptyMesh();
  merge(mesh, box([6, -0.05, 0], [30, 0.1, 18], MAT_CONCRETE));
  merge(mesh, box([6, -0.12, 0], [140, 0.1, 140], MAT_GRAVEL));
  return mesh;
}

export { normalize };
