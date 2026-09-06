# Docs app workflows

## Example thumbnail pipeline

- **Bake thumbnails in Docker:** Run `pnpm thumbs:docker [-- --only <slug>]` from the repo root. This script (`infra/test-docker/thumbs.sh`) builds the Dawn + Xvfb image and executes `pnpm --filter docs thumbs`, which in turn calls `apps/docs/scripts/render-example-thumbs.mjs --update` to write PNGs.
- **Smoke-check without writing:** Run `pnpm --filter docs thumbs:check [-- --only <slug>]` to run the same renderer with `--check`. The command exits non-zero if a thumbnail is missing or divergent.
- **Output:** The renderer writes two files per example inside `apps/docs/public/examples/`: `<slug>.card.png` (1280×720) and `<slug>.hero.png` (1600×900).

### Adding or updating an example

1. Create `apps/docs/examples/<slug>/` with:
   - `meta.ts` describing the example. Include a `thumb` object so the renderer knows how to drive the scene (see below).
   - `example.ts` exporting `run(canvas)` as a self-contained example. Import only from `vgpu` packages and files in the example's own folder.
   - Any WGSL or helper files referenced by the example.
2. Fragment-only examples should use the inline `run()` pattern: initialize vgpu, create a surface, create an effect, update `time` and `resolution` uniforms in `gpu.frame.loop`, draw the effect, and return cleanup that stops the loop and disposes the GPU. Export `thumb: { time: number }` in `meta.ts` so the script-local fragment thumbnail renderer samples a deterministic moment.
3. Compute-heavy or multi-pass examples should expose a custom `renderThumb(gpu, target, options)` that advances a fixed number of frames:
   - Set `thumb: { warmupFrames: number, dt: number }` so the Docker runner can pass `{ frames, dt }` into your renderer.
   - Keep the warm-up deterministic: fixed timesteps and seeded randomness only.
4. Run `pnpm thumbs:docker -- --only <slug>` until both PNGs look right, then commit them.
5. `apps/docs/scripts/ingest-examples.mjs` (wired to `predev`/`prebuild`) regenerates:
   - `lib/examples-source.generated.ts` for all example files.
   - `lib/example-thumbs.generated.ts`, which records which slugs have both PNGs. Missing PNGs fall back to the gradient placeholder in `<ExampleCard>`.

### Debugging internal render targets headlessly

- `node scripts/render-example-intermediates.mjs --slug <slug> --size 960x540 --time 6.2 --out ../../artifacts/<slug>` renders one example's `renderThumb` with `vgpu/node` and writes `final.png` plus one PNG per target the example reports through `onIntermediateRendered(kind, pixels, size)`.
- It needs a healthy `vgpu doctor`; a software Vulkan driver is enough (`apt-get install mesa-vulkan-drivers`, then export `VK_ICD_FILENAMES`/`VK_DRIVER_FILES` pointing at `lvp_icd.json`). A 960×540 frame of `spaceship-thrusters` renders in about a second on lavapipe, so you can iterate on shader detail without a browser.
- Readback only supports 8-bit formats, so examples preview HDR targets through a tonemapping pass first (`spaceship-thrusters/debug-preview.wgsl` is a reusable template). `spaceship-thrusters` reports `noise-atlas`, `detail`, `concrete`, `concrete-normal`, `gravel`, `gravel-normal` (baked ground material atlases: four box-filtered levels with periodic borders, since targets have no mip chain), `shadow-map`, `scene-color`, `scene-depth`, `scene-normal`, `plume-grid`, `fire-hdr`, and `bloom`, plus `ao`, `scene-lit` and `composite` in the social pipeline.
- `--quality social` on `render-example-intermediates.mjs` and `profile-example.mjs` selects `spaceship-thrusters`' heavier offline variant (screen-space ambient occlusion, higher-resolution bloom, and a lens pass with edge softness, vignette and film grain). The interactive page uses `RENDER_QUALITY` in `example.ts`; flip it to `'social'` to run the same pipeline in the browser. The plume passes are identical in both.

- `node scripts/render-example-sequence.mjs --slug <slug> --frames 8 --dt 0.01667 --out <dir>` renders consecutive animated frames of an example that exports `renderSequence(gpu, target, frames, dt, onFrame)`, one PNG per frame, to inspect temporal techniques (interleaved rendering, history blending) on moving content.
- `--camera "px,py,pz/tx,ty,tz[/fov]"` on `render-example-intermediates.mjs` overrides the example's camera (examples opt in through `ThrusterThumbOptions.camera`), which is how the plume grid was checked from six angles against the direct march.
- `node scripts/profile-example.mjs --slug <slug> --size 1280x720 --frames 20` times each pass of an example that exports `profile(gpu, target, frames)` (`spaceship-thrusters` reports scene, ao, grid, fire, bloom, composite and post; ao and post are zero in the fast pipeline). On lavapipe the numbers are CPU rasterization times, only meaningful relative to each other.

### Gotchas and safeguards

- **Determinism:** `render-example-thumbs.mjs` enforces deterministic inputs. Do not sample wall-clock time or rely on JS `Date` inside `renderThumb`; use the `time`, `frames`, and `dt` provided via `meta.thumb`.
- **Variance guard:** Renders with luma variance `< 6` fail the run (thumbs that stay black are treated as non-converged). Ensure the frame you capture has real contrast.
- **Resource lifetime:** The renderer calls `gpu.dispose()` for each example. Allocate one `init()` per example and clean up inside your `renderThumb`/`run` helpers to avoid Dawn segfaults.
- **Fixed dimensions:** Always render both `card` (1280×720) and `hero` (1600×900) targets. Hard-code size-aware uniforms if needed.
- **Headless opt-out:** If an example truly cannot run headlessly, set `thumb: { headless: false }` in `meta.ts` to skip it. This should be rare; most examples should implement a deterministic `renderThumb`.
