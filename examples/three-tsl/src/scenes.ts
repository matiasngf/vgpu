import * as THREE from "three/webgpu";
import type { Node } from "three/webgpu";
import { createLavaMaterial } from "./lava-material.ts";
import { applyNightEnvironment } from "./environment.ts";

export type DemoMeshKind = "knot" | "sphere" | "plane";

export interface DemoSceneOptions {
  readonly mesh?: DemoMeshKind;
  /** Fixed frame time for deterministic stills; defaults to the live clock. */
  readonly timeNode?: Node;
  /** Debug multiplier for the key light. */
  readonly lightScale?: number;
  /** Debug override for the lava glow intensity. */
  readonly glowIntensity?: number;
}

export interface DemoScene {
  readonly scene: THREE.Scene;
  readonly mesh: THREE.Mesh;
}

export function buildDemoMesh(kind: DemoMeshKind, material: THREE.Material): THREE.Mesh {
  if (kind === "sphere") return new THREE.Mesh(new THREE.IcosahedronGeometry(1.45, 96), material);
  if (kind === "plane") {
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(4.4, 4.4, 256, 256), material);
    plane.rotation.x = -1.05;
    return plane;
  }
  return new THREE.Mesh(new THREE.TorusKnotGeometry(1, 0.38, 400, 64), material);
}

/** The demo camera: slightly above, looking at the origin. */
export function createDemoCamera(aspect: number): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 50);
  camera.position.set(0, 1.2, 4.2);
  camera.lookAt(0, 0, 0);
  return camera;
}

/** Scene, lights, environment, and mesh for the lava demo. */
export async function createDemoScene(options: DemoSceneOptions = {}): Promise<DemoScene> {
  const scene = new THREE.Scene();

  // HDRI ambient (backdrop stays black) plus a soft warm-neutral key; the
  // warm floor bounce fakes the glow lighting the crust back.
  await applyNightEnvironment(scene);
  const key = new THREE.DirectionalLight(0xf2e4d2, 1.8 * (options.lightScale ?? 1));
  key.position.set(3, 2.2, 2);
  scene.add(key);
  scene.add(new THREE.HemisphereLight(0x3a3230, 0xb33a10, 0.25));

  const lava = createLavaMaterial({ timeNode: options.timeNode });
  if (options.glowIntensity !== undefined) lava.glowIntensity.value = options.glowIntensity;

  const mesh = buildDemoMesh(options.mesh ?? "knot", lava.material);
  scene.add(mesh);
  return { scene, mesh };
}
