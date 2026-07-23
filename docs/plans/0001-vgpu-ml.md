# Plan: `@vgpu/ml` — TensorFlow.js/Keras model inference on vgpu

Status: draft plan (no code yet)

## Goal

Run neural networks trained/exported in TensorFlow.js format (GraphModel `model.json` + weight
shards, Keras LayersModel) directly on the vgpu compute surface, so model outputs stay on the GPU
and feed straight into vgpu effects/draws. The headline use case is interactive
"MediaPipe → shader" pipelines: webcam frame in, landmarks/segmentation mask out, bound into a
fragment shader in the same frame with zero CPU roundtrip.

Non-goal for v1: training, and large-LLM-scale text generation. The op set and memory model are
designed so transformer ops (matmul, layernorm, attention, KV cache) fit later, but the first
target models are real-time vision models (`@tensorflow-models/*`: face landmarks, hand pose,
selfie segmentation, body pose) because they are small, latency-sensitive, and their outputs are
exactly what shaders want to consume.

## Target developer experience

```ts
import { init } from "vgpu";
import { loadGraphModel } from "@vgpu/ml";

const gpu = await init({ requiredFeatures: ["shader-f16"] });
const model = await loadGraphModel(gpu, "/models/selfie_segmentation/model.json");

const camera = gpu.videoTexture(videoEl);                    // new core capability
const input = model.input({ from: camera, size: [256, 256], normalize: [0, 1] });

const composite = gpu.effect(COMPOSITE_WGSL);
gpu.frame.loop(() => {
  const out = model.run(input);                              // one submit, all layers batched
  composite.set({ mask: out.segmentation.texture, frame: camera });
  composite.draw();
});
```

Key properties:

- `model.run()` encodes every layer as dispatches in one command encoder — never one submit per
  layer.
- Outputs are `Tensor` objects backed by vgpu storage buffers (or textures on request), directly
  accepted by `effect.set()` / `draw.set()` / `compute.set()` by identity, same ownership rules as
  today.
- `tensor.read()` exists for debugging/CPU consumers but is never required in the hot path.

## Architecture decision: bridge first, native second

There are two viable strategies, and the plan is to ship both, in order:

### Phase A — bridge mode (`@vgpu/ml-tfjs`)

tfjs already has a production WebGPU backend. Its backend can be constructed on an externally
provided `GPUDevice`, `tensor.dataToGPU()` exposes the output `GPUBuffer`, and `tf.tensor()`
accepts a `WebGPUData` buffer for zero-copy input. That means we can get the full
"MediaPipe → shader" experience quickly:

- Register the tfjs WebGPU backend on the same `GPUDevice` vgpu owns.
- Wrap tfjs output buffers as vgpu `StorageBuffer` facades (`wrapStorageBuffer` in
  `packages/vgpu-api/src/storage.ts` already wraps core buffers; this needs a sibling that wraps a
  raw external `GPUBuffer` with an identity vgpu's binding cache understands).
- Feed vgpu textures into tfjs via `tf.tensor({ buffer })` after a small
  texture→buffer preprocessing kernel.

Bridge mode validates the I/O design, the demos, and the public API shape with weeks of work
instead of months, and it inherits tfjs's full op coverage. Its costs — tfjs bundle size, no
control over fusion/memory, per-op submit granularity inside tfjs — are what native mode then
removes.

### Phase B — native mode (`@vgpu/ml`)

Our own runtime: parse the TFJS GraphModel topology directly (it is JSON of TF ops:
`Conv2D`, `FusedConv2D`/`_FusedConv2D`, `DepthwiseConv2dNative`, `Relu6`, `Add`, `Reshape`,
`Softmax`, …), plan it, and execute it with WGSL kernels authored as `@vgpu/wgsl` modules. Same
public API as bridge mode, selected per model load, so apps migrate model-by-model.

The rest of this plan details native mode; bridge mode reuses its tensor/I-O layers.

## Package layout

| Package | Contents |
| --- | --- |
| `@vgpu/ml` | Public API: `Tensor`, `loadGraphModel`, `loadLayersModel`, planner, executor. |
| `@vgpu/ml-kernels` | WGSL kernels as `@vgpu/wgsl` modules + per-op TS descriptors (dispatch geometry, bindings, uniforms). Usable standalone for hand-rolled compute. |
| `@vgpu/ml-tfjs` | Bridge mode: tfjs-webgpu interop on a shared device. Optional peer dep on `@tensorflow/tfjs`. |

`vgpu` (main package) gains no hard dependency on any of these; `@vgpu/ml` depends on `vgpu` +
`@vgpu/core`.

## Layers of the native runtime

### 1. Tensor layer

- `Tensor { shape, dtype, buffer, read(), toTexture(opts) }`, dtypes `f32`, `f16`, `i32`, `u32`
  (+ quantized `u8`/`u16` weights dequantized on upload).
- Backed by core `Buffer` via the same facade pattern as `RingStorageBuffer`, so `set()` binding
  by identity works unchanged.
- Reshape/transpose are metadata-only views whenever strides allow; kernels take layout uniforms.
- `toTexture()` for outputs shaders want to *sample* (segmentation masks) — one blit kernel,
  cached.

### 2. Kernel registry (`@vgpu/ml-kernels`)

Priority op set, chosen from what the target MediaPipe-family graphs actually use:

1. `matmul` (tiled, subgroup-friendly workgroup sizes, f32 + f16 variants)
2. `conv2d` and `depthwise_conv2d` (direct kernels; im2col+matmul fallback for odd shapes)
3. fused bias + activation epilogues (relu, relu6, sigmoid, tanh, hard-swish) on 1–2
4. elementwise binary/unary (add, mul, sub, clip, exp, …) with broadcast
5. reductions (max, sum, mean), `softmax`, `argmax`
6. `resize_bilinear`/`nearest`, `pad`, `concat`, `slice`, `transpose`
7. pooling (max/avg)

Each kernel ships as a WGSL module plus a TS descriptor: bindings, uniform layout, workgroup
size, dispatch-count function of shapes. Kernels are testable one-by-one against golden CPU
references. Later (LLM track): `layernorm`, `gelu`, fused attention, KV-cache-aware matmul.

### 3. Graph layer

- **Parser**: GraphModel `model.json` → internal graph IR; weight shard fetch + decode
  (including tfjs quantization headers). LayersModel support via a thin Keras→IR lowering
  (Dense/Conv/BN/Activation cover most exported Keras models).
- **Planner** (runs once at load):
  - topological order, dead-node elimination, constant folding
  - fusion: conv/matmul + bias + activation into one dispatch (mirrors `_FusedConv2D`)
  - memory plan: liveness analysis → arena of reusable storage buffers instead of
    one buffer per intermediate tensor
  - fail-fast on unsupported ops with a single error listing every missing op for that model
    (`VGPU-ML-OPS-UNSUPPORTED`), so coverage gaps are diagnosable in one shot
- **Executor**: replays the plan into one encoder per `run()`; uniforms for dynamic shapes via
  dynamic offsets (the performance-playbook pattern).

### 4. I/O bridges

- **In**: `gpu.videoTexture(video)` (core gap, see below) → preprocessing kernel
  (letterbox/resize/normalize/NCHW-vs-NHWC) → input tensor. Also `fromTexture(target)` so a vgpu
  render target can feed a model (shader → model → shader loops).
- **Out**: tensors bind directly into effects/draws; `toTexture()` for sampling;
  `read()` for CPU (async, never blocks the frame loop).

## Required additions to existing vgpu packages

These are prerequisites, worth landing as standalone PRs because they help all compute users:

1. **Batched compute encoding.** `Compute.dispatch()` currently creates and submits one encoder
   per call (`packages/vgpu-api/src/compute.ts`). Needed: compute passes on `Frame`
   (`frame.compute(cb)` encoding many dispatches into the frame's encoder) or an equivalent
   multi-dispatch recorder. A 100-layer model must be one submit.
2. **External image import.** No `copyExternalImageToTexture` path exists in
   `@vgpu/core`/`vgpu` today. Needed: `gpu.videoTexture(source)` /
   `texture.writeExternal(source)` for `HTMLVideoElement`/`VideoFrame`/`ImageBitmap`/canvas.
3. **Sub-range buffer bindings.** `set-resources.ts` always binds `{ offset: 0, size: full }`.
   The memory arena needs `{ buffer, offset, size }` binding views (offset-aligned to
   `minStorageBufferOffsetAlignment`).
4. **Feature/limit ergonomics.** `requiredFeatures` passthrough already exists; add detection
   helpers (`gpu.features.has("shader-f16")`) so kernels can select f16 variants, and optional
   `timestamp-query` plumbing for the profiler.
5. **External buffer wrapping** (bridge mode): wrap a foreign `GPUBuffer` as a bindable storage
   facade with a stable resource identity.

## Testing strategy

- **Mock adapter**: parser, planner, memory-arena, and encoding-shape tests run deterministically
  on `vgpu/mock` (dispatch counts, binding layouts, buffer reuse assertions).
- **Node/Dawn adapter**: numeric correctness per kernel and per model against golden outputs
  generated by tfjs-cpu in a script (checked-in `.npy`-style fixtures, tolerance-based compare,
  looser tolerances for f16). Runs in the existing `infra/test-docker` lanes.
- **Examples as acceptance tests**: an `examples/ml-segmentation` app (webcam → selfie
  segmentation → shader composite) and a headless Node render producing a snapshot, following the
  existing `by-example-*` pattern.

## Milestones

| # | Deliverable | Depends on |
| --- | --- | --- |
| 0 | Core prerequisites: batched compute pass, external image import, sub-range bindings | — |
| 1 | `@vgpu/ml-tfjs` bridge: shared-device tfjs backend, zero-copy in/out, webcam→segmentation→shader demo | 0 (partial) |
| 2 | Tensor layer + `@vgpu/ml-kernels` MVP (matmul, conv, depthwise, elementwise, softmax, resize) with golden tests | 0 |
| 3 | GraphModel parser + planner + executor; first real model (selfie segmentation) running natively, output-parity vs bridge mode | 2 |
| 4 | Second model class (face/hand landmarks), LayersModel lowering, quantized weights | 3 |
| 5 | Perf pass: fusion coverage, f16 kernels, arena tuning, timestamp profiler, workgroup autotune | 3 |
| 6 | Exploratory LLM track: layernorm/gelu/attention kernels, KV cache, tiny transformer (e.g. char-level or TinyStories-class) generating on-GPU | 2, 5 |

Milestone 1 is the demo-able moment ("media pipe → shader" works end to end); milestones 2–5
replace its internals without changing the app-facing API.

## Risks and open questions

- **Op coverage long tail.** TF graphs use hundreds of ops. Mitigation: fail-fast diagnostics,
  prioritize by target models, keep bridge mode as the escape hatch for uncovered models.
- **MediaPipe format reality.** Current MediaPipe Solutions ship TFLite models; the
  tfjs-converted versions live in `@tensorflow-models/*` GraphModel form — v1 targets those.
  A TFLite flatbuffer front-end is a possible later parser reusing the same IR.
- **Precision.** f16 doubles matmul throughput but some landmark models are f16-sensitive;
  per-model precision policy with f32 fallback, decided by the golden-test tolerances.
- **"LLM" scope.** True in-browser LLMs in tfjs format are effectively nonexistent; serious LLM
  inference would mean a GGUF/safetensors front-end on the same kernel/executor layers. Kept
  explicitly as the milestone-6 exploration rather than a v1 promise.
- **Naming.** `@vgpu/ml` vs a `vgpu/ml` subpath of the main package: proposed as a separate
  package so the main bundle stays slim (there is a bundle-size check in CI).
