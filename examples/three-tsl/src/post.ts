import * as THREE from "three/webgpu";
import { pass } from "three/tsl";
import { bloom } from "three/addons/tsl/display/BloomNode.js";

/**
 * Scene pass + HDR bloom. The threshold sits above anything the crust can
 * reflect, so only the incandescent melt blooms; tone mapping is applied by
 * PostProcessing at the end of the chain (renderer.toneMapping).
 */
export function createBloomPipeline(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  options: { readonly samples?: number } = {},
): THREE.PostProcessing {
  const postProcessing = new THREE.PostProcessing(renderer);
  const scenePass = pass(scene, camera, { samples: options.samples ?? 4 });
  const sceneColor = scenePass.getTextureNode();
  const bloomNode = bloom(sceneColor, 0.3, 0.15, 1.7);
  postProcessing.outputNode = sceneColor.add(bloomNode);
  return postProcessing;
}
