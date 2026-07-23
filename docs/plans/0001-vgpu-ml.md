# Plan: vgpu ML adapters — run and visualize neural networks on the vgpu device

Status: draft plan (no code yet)

## Goal

Make vgpu the best place to *use* neural networks interactively — webcam → model →
shader in the same frame, and live visualization of networks while they train — without building
our own inference runtime. Mature runtimes (TensorFlow.js, ONNX Runtime Web) already have
optimized WebGPU kernels, graph fusion, and full op coverage; reimplementing them would produce a
permanently half-finished library. Instead, vgpu provides the layer those runtimes lack:

1. **Device sharing** — the runtime and vgpu operate on the same `GPUDevice`, so tensors and
   shader resources are zero-copy neighbors.
2. **Tensor ↔ shader bridging** — wrap a runtime's output `GPUBuffer` as a vgpu-bindable resource
   with shape/dtype metadata, and feed vgpu textures (webcam, render targets) in as model inputs.
3. **Ergonomics** — `effect.set({ mask: out.tensor })` just works, with the same ownership and
   identity rules as every other vgpu resource.

Explicit non-goal: our own kernel library, graph parser, or autodiff. If a model doesn't run well
on the chosen runtime, the fix belongs upstream, not in a parallel half-runtime here.

## Target developer experience

Inference (interactive installation: webcam → segmentation → shader):

```ts
import { init } from "vgpu";
import { createOrtSession } from "@vgpu/ml-ort";

const gpu = await init();
const session = await createOrtSession(gpu, "/models/segmentation.onnx");

const camera = gpu.videoTexture(videoEl);                  // new core capability
const composite = gpu.effect(COMPOSITE_WGSL);

gpu.frame.loop(async () => {
  const input = session.input("image", { from: camera, size: [256, 256], normalize: [0, 1] });
  const out = await session.run({ image: input });         // outputs stay on-GPU
  composite.set({ mask: out.mask.texture, frame: camera });
  composite.draw();
});
```

Training visualization (tfjs owns training, vgpu owns the pixels):

```ts
import { init } from "vgpu";
import { shareDevice, wrapTensor } from "@vgpu/ml-tfjs";
import * as tf from "@tensorflow/tfjs";

const gpu = await init();
await shareDevice(gpu);                                    // tfjs webgpu backend on gpu's device

const model = buildKerasModel();
const view = gpu.effect(WEIGHTS_VIS_WGSL);
model.fit(xs, ys, {
  callbacks: { onBatchEnd: () => {
    const w = wrapTensor(gpu, model.layers[0].getWeights()[0]);  // zero-copy view
    gpu.frame((f) => { view.set({ weights: w }); f.pass(surfaceTarget, view); });
  } },
});
```

## Package layout

| Package | Contents |
| --- | --- |
| `@vgpu/ml` | Runtime-agnostic interop layer: `Tensor` facade (external `GPUBuffer` + shape/dtype, bindable via `set()`), preprocessing kernels (texture→tensor: resize/letterbox/normalize/NHWC-NCHW), postprocessing (tensor→texture blit for sampling), readback helpers. No runtime dependency. |
| `@vgpu/ml-ort` | ONNX Runtime Web adapter — the **optimal inference path** for pretrained networks. Uses ORT's WebGPU EP with I/O binding (`Tensor.fromGpuBuffer()` in, `preferredOutputLocation: "gpu-buffer"` out). ONNX is the format everything exports to (Keras, PyTorch, and TFLite/MediaPipe models via conversion). |
| `@vgpu/ml-tfjs` | TensorFlow.js adapter — **training + visualization**, and tfjs's pretrained model zoo (`@tensorflow-models/*`). Registers the tfjs WebGPU backend on vgpu's device; `tensor.dataToGPU()` out, `tf.tensor({ buffer })` in. |

Both runtime adapters are thin: device wiring + tensor conversion to/from the `@vgpu/ml`
facade. All shader-facing behavior lives in `@vgpu/ml` so the two runtimes are interchangeable
from the app's point of view.

## Why two runtimes

- **ORT Web (WebGPU EP)** is the fastest, most actively optimized web inference runtime: graph
  optimization/fusion at session creation, tuned kernels, GPU-resident I/O. It is the answer to
  "load a pretrained network and run it as fast as the browser allows" — the interactive
  installation case.
- **tfjs** is the only one with in-browser training (autodiff + `model.fit()` on WebGPU) and a
  curated model zoo. It is the answer to "watch a network learn, live, in a shader".

Recommending one runtime for both jobs would compromise one of them; two thin adapters over one
shared interop layer costs little.

## Device sharing: the one hard integration point

Zero-copy requires that the runtime and vgpu use the *same* `GPUDevice` — WebGPU resources cannot
cross devices. Two mechanisms, and vgpu should support both:

1. **Inject vgpu's device into the runtime.** tfjs supports this cleanly
   (`new WebGPUBackend(device)` + `tf.registerBackend`). For ORT, device injection must be
   verified against the current release during the milestone-0 spike (the `env.webgpu` surface
   has changed across versions).
2. **vgpu adopts an external device: `init({ device })`.** If a runtime insists on creating its
   own device, vgpu wraps that one instead. This is a small, generally useful core addition
   (embedding vgpu into any existing WebGPU app) and makes the integration robust against
   runtime API churn. This is the fallback that guarantees the plan works regardless of what the
   ORT spike finds.

## Required additions to existing vgpu packages

1. **`init({ device })`** — construct the vgpu context over an externally created `GPUDevice`
   (mechanism 2 above).
2. **External buffer wrapping** — `packages/vgpu-api/src/storage.ts` wraps core `Buffer`s;
   the `Tensor` facade needs a sibling that wraps a foreign `GPUBuffer` with a stable resource
   identity for the binding cache, plus explicit lifetime rules (the runtime owns the buffer;
   vgpu must invalidate bindings on release — ORT and tfjs both recycle output buffers, so the
   facade API must make "valid until next run/dispose" explicit).
3. **External image import** — `gpu.videoTexture(source)` for
   `HTMLVideoElement`/`VideoFrame`/`ImageBitmap`/canvas via `copyExternalImageToTexture`; no such
   path exists in `@vgpu/core` today.
4. **Feature helpers** — `gpu.features.has("shader-f16")` etc., so adapters can report what the
   shared device enables for the runtime (ORT WebGPU benefits from f16).

Notably *not* required anymore: batched compute encoding and sub-range storage bindings. The
runtimes own their own encoders; vgpu's pre/post kernels are 1–2 dispatches per frame, fine
through the existing `gpu.compute()` path. Both remain nice-to-haves for general compute users
but are off this plan's critical path.

## What "optimal" means here, concretely

- **Session-level optimization is the runtime's job**: ORT fuses and plans at
  `createSession()`; we pass the right session options (WebGPU EP, GPU output location,
  free-dimension overrides for fixed input sizes) rather than owning kernels.
- **The adapter's job is to make the seams free**: no CPU roundtrip for image-sized tensors
  (masks, feature maps) in either direction; preprocessing on-GPU; `await session.run()` as the
  only sync point per frame.
- **Small outputs don't need heroics**: landmark sets (hands/face/pose) are a few KB — reading
  them back and uploading as uniforms is negligible. Zero-copy matters for image-sized tensors;
  the API supports both paths and the docs say when each is appropriate.
- **Measure, don't assume**: the example apps double as benchmarks (frame time with/without I/O
  binding, tfjs vs ORT on the same model) using vgpu's measuring docs conventions.

## Testing strategy

- **Mock adapter**: interop-layer unit tests (facade identity/binding behavior, pre/post kernel
  encoding shapes, lifetime invalidation) run deterministically on `vgpu/mock`.
- **Node/Dawn**: pre/post kernels verified numerically (golden fixtures); adapter smoke tests run
  a tiny ONNX/tfjs model end-to-end where the runtime supports Node WebGPU, otherwise
  browser-lane tests per `docs/topics/browser-testing.docs.md`.
- **Examples as acceptance tests**: `examples/ml-segmentation` (webcam → ORT segmentation →
  shader composite) and `examples/ml-training-viz` (tfjs `model.fit()` visualized live),
  following the `by-example-*` pattern.

## Milestones

| # | Deliverable | Depends on |
| --- | --- | --- |
| 0 | Spike: verify ORT WebGPU device injection + GPU I/O binding against current release; verify tfjs shared-device registration; decide inject vs adopt per runtime | — |
| 1 | Core additions: `init({ device })`, external buffer wrapping with lifetime rules, `gpu.videoTexture()` | 0 |
| 2 | `@vgpu/ml` interop layer: `Tensor` facade + pre/post kernels, mock + Dawn tests | 1 |
| 3 | `@vgpu/ml-tfjs`: shared device, zero-copy wrap, training-visualization example | 2 |
| 4 | `@vgpu/ml-ort`: session wrapper with I/O binding, webcam→segmentation→shader example, benchmark vs tfjs on the same model | 2 |
| 5 | Polish: docs topic ("ML on vgpu"), model-format guidance (export-to-ONNX recipes for Keras/PyTorch/MediaPipe), profiling notes, examples gallery | 3, 4 |

## Risks and open questions

- **ORT device injection** may not be supported in the current release — mitigated by
  `init({ device })` adoption (milestone 1), which works regardless.
- **Output buffer lifetimes**: both runtimes recycle GPU buffers between runs; the facade's
  "valid until next run" contract must be enforced (invalidate-on-reuse), or apps get silent
  stale reads. This is the main correctness risk of the whole plan and gets dedicated tests.
- **MediaPipe models** ship as TFLite; for the ORT path they need one-time conversion to ONNX
  (documented recipe), or use the tfjs-converted versions via the tfjs adapter. No runtime work,
  but real DX friction to document honestly.
- **Bundle size**: adapters keep runtimes as peer dependencies so `vgpu` itself stays slim
  (CI bundle-size check unaffected); apps opt into the runtime they use.
- **WASM/threads headers**: ORT's WASM fallback and threading want COOP/COEP headers; examples
  must document server config so the WebGPU EP is actually used.
