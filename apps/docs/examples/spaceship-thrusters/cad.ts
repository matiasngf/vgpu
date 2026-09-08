// Minimal parametric CAD kit for the spaceship-thrusters example.
//
// Every builder returns a `CadMesh` (positions, normals, uvs, indices) and can
// be transformed and merged, so the engine and the test stand are described
// as a handful of parameters instead of baked vertex blobs. The primitives are
// the ones a rocket engine needs: lathe/revolve profiles (nozzle, chamber),
// tubes swept along polylines with rounded elbows (feed lines, manifolds),
// tori (rings), cylinders, cones and boxes (stand, plates).

export interface CadMesh {
  positions: number[];
  normals: number[];
  uvs: number[];
  indices: number[];
  /** Per-vertex material id, used by the shader to pick a colour set. */
  materials: number[];
}

export type Vec3 = readonly [number, number, number];

export function emptyMesh(): CadMesh {
  return { positions: [], normals: [], uvs: [], indices: [], materials: [] };
}

export function merge(target: CadMesh, ...meshes: CadMesh[]): CadMesh {
  for (const mesh of meshes) {
    const base = target.positions.length / 3;
    target.positions.push(...mesh.positions);
    target.normals.push(...mesh.normals);
    target.uvs.push(...mesh.uvs);
    target.materials.push(...mesh.materials);
    for (const index of mesh.indices) target.indices.push(index + base);
  }
  return target;
}

// --- Small vector / matrix helpers (column-major 4x4) -------------------------

export function v3(x: number, y: number, z: number): Vec3 { return [x, y, z]; }
export function add(a: Vec3, b: Vec3): Vec3 { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function sub(a: Vec3, b: Vec3): Vec3 { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function scale(a: Vec3, s: number): Vec3 { return [a[0] * s, a[1] * s, a[2] * s]; }
export function dot(a: Vec3, b: Vec3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function length(a: Vec3): number { return Math.hypot(a[0], a[1], a[2]); }
export function normalize(a: Vec3): Vec3 {
  const l = length(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

export type Mat4 = number[]; // 16 numbers, column-major

export function identity(): Mat4 {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + r]! * b[c * 4 + k]!;
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

export function translation(t: Vec3): Mat4 {
  const m = identity();
  m[12] = t[0]; m[13] = t[1]; m[14] = t[2];
  return m;
}

export function scaling(s: Vec3): Mat4 {
  const m = identity();
  m[0] = s[0]; m[5] = s[1]; m[10] = s[2];
  return m;
}

/** Rotation about an arbitrary unit axis (radians). */
export function rotation(axis: Vec3, angle: number): Mat4 {
  const [x, y, z] = normalize(axis);
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [
    t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0,
    t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0,
    t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0,
    0, 0, 0, 1,
  ];
}

export function rotationX(a: number): Mat4 { return rotation([1, 0, 0], a); }
export function rotationY(a: number): Mat4 { return rotation([0, 1, 0], a); }
export function rotationZ(a: number): Mat4 { return rotation([0, 0, 1], a); }

/** Compose transforms right-to-left: compose(T, R, S) applies S first. */
export function compose(...mats: Mat4[]): Mat4 {
  return mats.reduce((acc, m) => multiply(acc, m), identity());
}

export function transformPoint(m: Mat4, p: Vec3): Vec3 {
  return [
    m[0]! * p[0] + m[4]! * p[1] + m[8]! * p[2] + m[12]!,
    m[1]! * p[0] + m[5]! * p[1] + m[9]! * p[2] + m[13]!,
    m[2]! * p[0] + m[6]! * p[1] + m[10]! * p[2] + m[14]!,
  ];
}

export function transformDirection(m: Mat4, d: Vec3): Vec3 {
  return normalize([
    m[0]! * d[0] + m[4]! * d[1] + m[8]! * d[2],
    m[1]! * d[0] + m[5]! * d[1] + m[9]! * d[2],
    m[2]! * d[0] + m[6]! * d[1] + m[10]! * d[2],
  ]);
}

/** Applies a rigid/uniform transform (normals are rotated, not inverse-transposed). */
export function transform(mesh: CadMesh, m: Mat4): CadMesh {
  const out = emptyMesh();
  for (let i = 0; i < mesh.positions.length; i += 3) {
    const p = transformPoint(m, [mesh.positions[i]!, mesh.positions[i + 1]!, mesh.positions[i + 2]!]);
    const n = transformDirection(m, [mesh.normals[i]!, mesh.normals[i + 1]!, mesh.normals[i + 2]!]);
    out.positions.push(...p);
    out.normals.push(...n);
  }
  out.uvs.push(...mesh.uvs);
  out.indices.push(...mesh.indices);
  out.materials.push(...mesh.materials);
  return out;
}

export function withMaterial(mesh: CadMesh, material: number): CadMesh {
  return { ...mesh, materials: mesh.materials.map(() => material) };
}

// --- Primitives ---------------------------------------------------------------

export interface ProfilePoint { readonly y: number; readonly r: number; }

/**
 * Revolves a profile (radius as a function of height along +Y) around the Y
 * axis. Profiles run from bottom to top; `closed` caps both ends with discs.
 * Normals come from the profile tangent, so smooth bells stay smooth.
 */
export function revolve(profile: readonly ProfilePoint[], segments = 64, material = 0, closed = false): CadMesh {
  const mesh = emptyMesh();
  const rows = profile.length;
  for (let i = 0; i < rows; i++) {
    const p = profile[i]!;
    const prev = profile[Math.max(0, i - 1)]!;
    const next = profile[Math.min(rows - 1, i + 1)]!;
    // Tangent along the profile; the normal is its perpendicular pointing outward.
    const tY = next.y - prev.y, tR = next.r - prev.r;
    const tLen = Math.hypot(tY, tR) || 1;
    const nR = tY / tLen, nY = -tR / tLen;
    for (let j = 0; j <= segments; j++) {
      const a = (j / segments) * Math.PI * 2;
      const c = Math.cos(a), s = Math.sin(a);
      mesh.positions.push(p.r * c, p.y, p.r * s);
      mesh.normals.push(nR * c, nY, nR * s);
      mesh.uvs.push(j / segments, i / (rows - 1));
      mesh.materials.push(material);
    }
  }
  for (let i = 0; i < rows - 1; i++) {
    for (let j = 0; j < segments; j++) {
      const a = i * (segments + 1) + j;
      const b = a + segments + 1;
      mesh.indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  if (closed) {
    merge(mesh, disc(profile[0]!.r, profile[0]!.y, segments, material, false), disc(profile[rows - 1]!.r, profile[rows - 1]!.y, segments, material, true));
  }
  return mesh;
}

/** A flat disc in the XZ plane at height y, facing +Y (up=true) or -Y. */
export function disc(radius: number, y: number, segments = 64, material = 0, up = true): CadMesh {
  const mesh = emptyMesh();
  const ny = up ? 1 : -1;
  mesh.positions.push(0, y, 0);
  mesh.normals.push(0, ny, 0);
  mesh.uvs.push(0.5, 0.5);
  mesh.materials.push(material);
  for (let j = 0; j <= segments; j++) {
    const a = (j / segments) * Math.PI * 2;
    const c = Math.cos(a), s = Math.sin(a);
    mesh.positions.push(radius * c, y, radius * s);
    mesh.normals.push(0, ny, 0);
    mesh.uvs.push(0.5 + 0.5 * c, 0.5 + 0.5 * s);
    mesh.materials.push(material);
  }
  for (let j = 1; j <= segments; j++) {
    if (up) mesh.indices.push(0, j + 1, j);
    else mesh.indices.push(0, j, j + 1);
  }
  return mesh;
}

/** Cylinder (or truncated cone) along +Y from y0 to y1 with flat caps. */
export function cylinder(r0: number, r1: number, y0: number, y1: number, segments = 48, material = 0): CadMesh {
  return revolve([{ y: y0, r: r0 }, { y: y1, r: r1 }], segments, material, true);
}

/** Torus around the Y axis: ring radius R, tube radius r, centred at height y. */
export function torus(R: number, r: number, y = 0, segments = 64, tubeSegments = 16, material = 0): CadMesh {
  const mesh = emptyMesh();
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    const ca = Math.cos(a), sa = Math.sin(a);
    for (let j = 0; j <= tubeSegments; j++) {
      const b = (j / tubeSegments) * Math.PI * 2;
      const cb = Math.cos(b), sb = Math.sin(b);
      mesh.positions.push((R + r * cb) * ca, y + r * sb, (R + r * cb) * sa);
      mesh.normals.push(cb * ca, sb, cb * sa);
      mesh.uvs.push(i / segments, j / tubeSegments);
      mesh.materials.push(material);
    }
  }
  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < tubeSegments; j++) {
      const a = i * (tubeSegments + 1) + j;
      const b = a + tubeSegments + 1;
      mesh.indices.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }
  return mesh;
}

/** Axis-aligned box centred at `center` with full size `size`. Flat-shaded. */
export function box(center: Vec3, size: Vec3, material = 0): CadMesh {
  const mesh = emptyMesh();
  const [cx, cy, cz] = center;
  const [hx, hy, hz] = [size[0] / 2, size[1] / 2, size[2] / 2];
  const faces: { n: Vec3; u: Vec3; v: Vec3 }[] = [
    { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] },
    { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
    { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] },
    { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
    { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
    { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] },
  ];
  for (const face of faces) {
    const base = mesh.positions.length / 3;
    const hn: Vec3 = [face.n[0] * hx, face.n[1] * hy, face.n[2] * hz];
    const hu: Vec3 = [face.u[0] * hx, face.u[1] * hy, face.u[2] * hz];
    const hv: Vec3 = [face.v[0] * hx, face.v[1] * hy, face.v[2] * hz];
    const corners: [number, number][] = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    for (const [su, sv] of corners) {
      mesh.positions.push(cx + hn[0] + su * hu[0] + sv * hv[0], cy + hn[1] + su * hu[1] + sv * hv[1], cz + hn[2] + su * hu[2] + sv * hv[2]);
      mesh.normals.push(...face.n);
      mesh.uvs.push(su * 0.5 + 0.5, sv * 0.5 + 0.5);
      mesh.materials.push(material);
    }
    mesh.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return mesh;
}

/**
 * Sweeps a circle of radius `r` along a polyline, rounding every corner with
 * an arc of radius `bend` (a pipe with elbows). Uses parallel-transport frames
 * so the tube never twists. Caps are closed.
 */
export function tube(path: readonly Vec3[], r: number, bend = r * 2, segments = 16, material = 0): CadMesh {
  const points = roundPolyline(path, bend);
  const mesh = emptyMesh();
  // Parallel transport frames.
  let tangent = normalize(sub(points[1]!, points[0]!));
  let normal = normalize(cross(tangent, Math.abs(tangent[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]));
  const frames: { p: Vec3; n: Vec3; b: Vec3 }[] = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i]!;
    const nextTangent = i < points.length - 1 ? normalize(sub(points[i + 1]!, p)) : tangent;
    // Rotate `normal` from the old tangent to the new one.
    const axis = cross(tangent, nextTangent);
    const sinA = length(axis);
    if (sinA > 1e-6) {
      const cosA = Math.max(-1, Math.min(1, dot(tangent, nextTangent)));
      const rot = rotation(axis, Math.atan2(sinA, cosA));
      normal = transformDirection(rot, normal);
    }
    tangent = nextTangent;
    normal = normalize(sub(normal, scale(tangent, dot(normal, tangent))));
    frames.push({ p, n: normal, b: normalize(cross(tangent, normal)) });
  }
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i]!;
    for (let j = 0; j <= segments; j++) {
      const a = (j / segments) * Math.PI * 2;
      const c = Math.cos(a), s = Math.sin(a);
      const n: Vec3 = [f.n[0] * c + f.b[0] * s, f.n[1] * c + f.b[1] * s, f.n[2] * c + f.b[2] * s];
      mesh.positions.push(f.p[0] + n[0] * r, f.p[1] + n[1] * r, f.p[2] + n[2] * r);
      mesh.normals.push(...n);
      mesh.uvs.push(j / segments, i / (frames.length - 1));
      mesh.materials.push(material);
    }
  }
  for (let i = 0; i < frames.length - 1; i++) {
    for (let j = 0; j < segments; j++) {
      const a = i * (segments + 1) + j;
      const b = a + segments + 1;
      mesh.indices.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }
  // End caps.
  merge(mesh, capAt(frames[0]!.p, scale(normalize(sub(points[0]!, points[1]!)), 1), r, segments, material));
  const last = frames.length - 1;
  merge(mesh, capAt(frames[last]!.p, normalize(sub(points[last]!, points[last - 1]!)), r, segments, material));
  return mesh;
}

function capAt(center: Vec3, dir: Vec3, r: number, segments: number, material: number): CadMesh {
  const n = normalize(dir);
  const u = normalize(cross(n, Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]));
  const v = cross(n, u);
  const mesh = emptyMesh();
  mesh.positions.push(...center);
  mesh.normals.push(...n);
  mesh.uvs.push(0.5, 0.5);
  mesh.materials.push(material);
  for (let j = 0; j <= segments; j++) {
    const a = (j / segments) * Math.PI * 2;
    const c = Math.cos(a), s = Math.sin(a);
    mesh.positions.push(center[0] + (u[0] * c + v[0] * s) * r, center[1] + (u[1] * c + v[1] * s) * r, center[2] + (u[2] * c + v[2] * s) * r);
    mesh.normals.push(...n);
    mesh.uvs.push(0.5 + 0.5 * c, 0.5 + 0.5 * s);
    mesh.materials.push(material);
  }
  for (let j = 1; j <= segments; j++) mesh.indices.push(0, j, j + 1);
  return mesh;
}

/** Replaces every interior corner of a polyline with a sampled circular arc. */
function roundPolyline(path: readonly Vec3[], bend: number): Vec3[] {
  if (path.length < 3 || bend <= 0) return [...path];
  const out: Vec3[] = [path[0]!];
  for (let i = 1; i < path.length - 1; i++) {
    const prev = path[i - 1]!, p = path[i]!, next = path[i + 1]!;
    const dIn = normalize(sub(p, prev)), dOut = normalize(sub(next, p));
    const cosTheta = Math.max(-1, Math.min(1, dot(dIn, dOut)));
    const theta = Math.acos(cosTheta); // turning angle
    if (theta < 1e-4) { out.push(p); continue; }
    // Distance from the corner to where the arc starts/ends.
    const maxTangent = Math.min(length(sub(p, prev)), length(sub(next, p))) * 0.49;
    const tangentLength = Math.min(bend * Math.tan(theta / 2), maxTangent);
    const radius = tangentLength / Math.tan(theta / 2);
    const start = sub(p, scale(dIn, tangentLength));
    const end = add(p, scale(dOut, tangentLength));
    // Arc centre lies along the bisector, inside the corner.
    const bisector = normalize(sub(dOut, dIn));
    const centre = add(p, scale(bisector, radius / Math.cos(theta / 2)));
    const steps = Math.max(2, Math.ceil(theta / (Math.PI / 12)));
    const a0 = sub(start, centre), a1 = sub(end, centre);
    for (let k = 0; k <= steps; k++) {
      // Spherical interpolation between the two radius vectors.
      const t = k / steps;
      const omega = Math.acos(Math.max(-1, Math.min(1, dot(normalize(a0), normalize(a1)))));
      const w0 = Math.sin((1 - t) * omega) / Math.sin(omega), w1 = Math.sin(t * omega) / Math.sin(omega);
      out.push(add(centre, add(scale(a0, w0), scale(a1, w1))));
    }
  }
  out.push(path[path.length - 1]!);
  return out;
}

/** Packs a CadMesh into typed arrays for upload. */
export function pack(mesh: CadMesh) {
  return {
    positions: new Float32Array(mesh.positions),
    normals: new Float32Array(mesh.normals),
    uvs: new Float32Array(mesh.uvs),
    materials: new Float32Array(mesh.materials),
    indices: new Uint32Array(mesh.indices),
    vertexCount: mesh.positions.length / 3,
  };
}

// --- Camera helpers (WebGPU conventions: right-handed, NDC z in [0, 1]) -------

export function perspective(fovYRadians: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovYRadians / 2);
  const m = new Array<number>(16).fill(0);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = far / (near - far);
  m[11] = -1;
  m[14] = (near * far) / (near - far);
  return m;
}

export function lookAt(eye: Vec3, target: Vec3, up: Vec3 = [0, 1, 0]): Mat4 {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ];
}

/** General 4x4 inverse (cofactor expansion). Returns identity when singular. */
export function invert(m: Mat4): Mat4 {
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) return identity();
  const inv = 1 / det;
  return [
    (a11 * b11 - a12 * b10 + a13 * b09) * inv, (a02 * b10 - a01 * b11 - a03 * b09) * inv, (a31 * b05 - a32 * b04 + a33 * b03) * inv, (a22 * b04 - a21 * b05 - a23 * b03) * inv,
    (a12 * b08 - a10 * b11 - a13 * b07) * inv, (a00 * b11 - a02 * b08 + a03 * b07) * inv, (a32 * b02 - a30 * b05 - a33 * b01) * inv, (a20 * b05 - a22 * b02 + a23 * b01) * inv,
    (a10 * b10 - a11 * b08 + a13 * b06) * inv, (a01 * b08 - a00 * b10 - a03 * b06) * inv, (a30 * b04 - a31 * b02 + a33 * b00) * inv, (a21 * b02 - a20 * b04 - a23 * b00) * inv,
    (a11 * b07 - a10 * b09 - a12 * b06) * inv, (a00 * b09 - a01 * b07 + a02 * b06) * inv, (a31 * b01 - a30 * b03 - a32 * b00) * inv, (a20 * b03 - a21 * b01 + a22 * b00) * inv,
  ];
}

export function orthographic(left: number, right: number, bottom: number, top: number, near: number, far: number): Mat4 {
  const m = new Array<number>(16).fill(0);
  m[0] = 2 / (right - left);
  m[5] = 2 / (top - bottom);
  m[10] = 1 / (near - far);
  m[12] = -(right + left) / (right - left);
  m[13] = -(top + bottom) / (top - bottom);
  m[14] = near / (near - far);
  m[15] = 1;
  return m;
}
