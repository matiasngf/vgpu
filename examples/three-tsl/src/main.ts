import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { createDemoCamera, createDemoScene } from "./scenes.ts";
import { createBloomPipeline } from "./post.ts";

async function main(): Promise<void> {
  if (navigator.gpu === undefined) {
    document.querySelector("#overlay")!.textContent = "WebGPU is not available in this browser.";
    return;
  }

  const renderer = new THREE.WebGPURenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  document.querySelector("#app")!.append(renderer.domElement);
  await renderer.init();

  const { scene, mesh } = await createDemoScene();
  const camera = createDemoCamera(window.innerWidth / window.innerHeight);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const postProcessing = createBloomPipeline(renderer, scene, camera);

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  renderer.setAnimationLoop(() => {
    mesh.rotation.y += 0.0012;
    controls.update();
    postProcessing.render();
  });
}

void main();
