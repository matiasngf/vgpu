// Headless keyframe renderer for the Dreamcore example.
//
//   node apps/docs/scripts/render-dreamcore.mjs [--out <dir>] [--width 1080 --height 1920] [--phases 0,0.3,1]
//   node apps/docs/scripts/render-dreamcore.mjs --cycle 40 --width 405 --height 720
//
// Writes one PNG per phase (0 = night, 1 = day) plus a side-by-side triptych, or with
// --cycle N one numbered frame per step of the example's day/night cycle. Needs a
// working `vgpu/node` adapter; on Linux without a GPU install Mesa lavapipe and export
// VK_ICD_FILENAMES (see `npx --package @vgpu/cli vgpu doctor`).
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { init } from 'vgpu/node';
import { writePng } from '@vgpu/cli/lib/snapshot/png.js';
import { transformWgsl } from '@vgpu/wgsl/loader-vite';

const args = parseArgs(process.argv.slice(2));
const docsDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.resolve(args.out ?? path.join(docsDir, '..', '..', 'artifacts', 'dreamcore'));
const width = Number(args.width ?? 1080);
const height = Number(args.height ?? 1920);
const cycleFrames = args.cycle ? Number(args.cycle) : 0;
const cacheDir = path.join(docsDir, '.dreamcore-cache');
const DEBUG_MODES = { world: 1, map: 2, main: 3, 'main-clean': 4 };
// --look '{"camera":{"height":1.7},"door":{"z":9}}' overrides the framing; --name prefixes the files.
const look = args.look ? JSON.parse(args.look) : undefined;
const prefix = args.name ? `${args.name}.` : '';

await mkdir(outDir, { recursive: true });
const { renderStill, phaseAt, CYCLE_SECONDS } = await loadExample();
const steps = args.debug
  ? debugSteps(args.debug, args['debug-cam'])
  : cycleFrames > 0
    ? Array.from({ length: cycleFrames }, (_, i) => {
      const time = (i / cycleFrames) * CYCLE_SECONDS;
      return { phase: phaseAt(time), time, name: `cycle-${String(i).padStart(3, '0')}` };
    })
    : String(args.phases ?? '0,0.3,1').split(',').map(Number).map((phase) => ({ phase, time: 0, name: phaseName(phase) }));
const frames = [];
for (const { phase, time, name, debug } of steps) {
  const gpu = await init();
  try {
    const target = gpu.target({ size: [width, height], format: 'rgba8unorm', label: `dreamcore-${name}` });
    const started = Date.now();
    await renderStill(gpu, target, { phase, time, samples: 4, debug, look });
    const pixels = await target.read();
    const file = path.join(outDir, `dreamcore.${prefix}${name}.png`);
    await writePng(file, pixels, width, height);
    frames.push(pixels);
    console.log(`- ${path.relative(process.cwd(), file)} (${width}x${height}, phase ${phase}, ${Date.now() - started}ms)`);
  } finally {
    gpu.dispose();
  }
}
if (frames.length > 1 && cycleFrames === 0 && !args.debug) {
  const gap = 8;
  const stripWidth = frames.length * width + (frames.length - 1) * gap;
  const strip = new Uint8Array(stripWidth * height * 4).fill(24);
  for (let y = 0; y < height; y++) {
    for (let i = 0; i < frames.length; i++) {
      const src = frames[i].subarray(y * width * 4, (y + 1) * width * 4);
      strip.set(src, (y * stripWidth + i * (width + gap)) * 4);
    }
  }
  for (let i = 3; i < strip.length; i += 4) strip[i] = 255;
  const file = path.join(outDir, `dreamcore.${prefix}triptych.png`);
  await writePng(file, strip, stripWidth, height);
  console.log(`- ${path.relative(process.cwd(), file)} (${stripWidth}x${height})`);
}
await rm(cacheDir, { recursive: true, force: true });

function phaseName(phase) {
  if (phase <= 0) return 'night';
  if (phase >= 1) return 'day';
  return `phase-${String(phase).replace('.', '_')}`;
}

async function loadExample() {
  await mkdir(cacheDir, { recursive: true });
  const entry = path.join(cacheDir, 'entry.ts');
  const bundle = path.join(cacheDir, 'example.mjs');
  await writeFile(entry, "export { renderStill, phaseAt, CYCLE_SECONDS } from '../examples/dreamcore/example.ts';\n");
  await build({
    entryPoints: [entry],
    outfile: bundle,
    bundle: true,
    platform: 'node',
    format: 'esm',
    sourcemap: false,
    external: ['vgpu', 'vgpu/node'],
    plugins: [{
      name: 'wgsl',
      setup(build) {
        build.onLoad({ filter: /\.wgsl$/ }, async (file) => {
          const source = await readFile(file.path, 'utf8');
          const result = await transformWgsl({ source, id: file.path });
          return { contents: result.code, loader: 'js', resolveDir: path.dirname(file.path) };
        });
      },
    }],
    logLevel: 'silent',
  });
  return import(pathToFileURL(bundle).href);
}

// --debug world | map | main | main-clean | all [--debug-cam x,y,z]: views of the sand world.
function debugSteps(spec, cam) {
  const modes = spec === 'all' ? Object.keys(DEBUG_MODES) : [spec];
  const camera = cam ? cam.split(',').map(Number) : undefined;
  return modes.map((name) => {
    if (!(name in DEBUG_MODES)) throw new Error(`Unknown debug view '${name}' (${Object.keys(DEBUG_MODES).join(', ')} or all).`);
    return { phase: 0, time: 0, name: `debug-${name}`, debug: { mode: DEBUG_MODES[name], camera } };
  });
}

function parseArgs(argv) {
  const parsed = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--') continue;
    if (!arg.startsWith('--')) throw new Error(`Unknown argument '${arg}'.`);
    parsed[arg.slice(2)] = argv[++i];
  }
  return parsed;
}
