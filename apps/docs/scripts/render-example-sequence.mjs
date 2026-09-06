// Headless animation check: renders consecutive frames of an example that
// exports `renderSequence(gpu, target, frames, dt, onFrame)` and writes one
// PNG per frame, for inspecting temporal techniques on moving content.
//
//   node scripts/render-example-sequence.mjs --slug spaceship-thrusters --frames 8 --dt 0.01667 --out ../../artifacts/seq

import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { init } from 'vgpu/node';
import { writePng } from '@vgpu/cli/lib/snapshot/png.js';
import { transformWgsl } from '@vgpu/wgsl/loader-vite';

const args = parseArgs(process.argv.slice(2));
const docsDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const cacheDir = path.join(docsDir, '.sequence-cache');
const outDir = path.resolve(args.out);
await mkdir(outDir, { recursive: true });
const renderSequence = await loadRenderer(args.slug);
const gpu = await init();
try {
  const target = gpu.target({ size: args.size, format: 'rgba8unorm', label: `sequence-${args.slug}` });
  const started = performance.now();
  await renderSequence(gpu, target, args.frames, args.dt, async (index, pixels, size) => {
    await writePng(path.join(outDir, `frame-${String(index).padStart(3, '0')}.png`), pixels, size[0], size[1]);
  });
  console.log(`${args.frames} frames -> ${path.relative(process.cwd(), outDir)} (${((performance.now() - started) / args.frames).toFixed(1)} ms/frame incl. readback)`);
} finally {
  gpu.dispose();
  await rm(cacheDir, { recursive: true, force: true });
}

async function loadRenderer(slug) {
  await mkdir(cacheDir, { recursive: true });
  const entry = path.join(cacheDir, 'entry.ts');
  const bundle = path.join(cacheDir, 'bundle.mjs');
  await writeFile(entry, `export { renderSequence } from '../examples/${slug}/example';\n`);
  await build({
    entryPoints: [entry], outfile: bundle, bundle: true, platform: 'node', format: 'esm', sourcemap: false,
    external: ['vgpu', 'vgpu/node'], logLevel: 'silent',
    plugins: [{ name: 'docs-wgsl', setup(b) {
      b.onLoad({ filter: /\.wgsl$/ }, async (file) => {
        const source = await readFile(file.path, 'utf8');
        const result = await transformWgsl({ source, id: file.path });
        return { contents: result.code, loader: 'js', resolveDir: path.dirname(file.path) };
      });
    } }],
  });
  const module = await import(`${pathToFileURL(bundle).href}?t=${Date.now()}`);
  if (typeof module.renderSequence !== 'function') throw new Error(`Example '${slug}' does not export renderSequence.`);
  return module.renderSequence;
}

function parseArgs(argv) {
  const parsed = { slug: undefined, size: [1280, 720], frames: 8, dt: 1 / 60, out: path.join('..', '..', 'artifacts', 'example-sequence') };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--') continue;
    else if (arg === '--slug') parsed.slug = argv[++i];
    else if (arg === '--size') parsed.size = argv[++i].split('x').map(Number);
    else if (arg === '--frames') parsed.frames = Number(argv[++i]);
    else if (arg === '--dt') parsed.dt = Number(argv[++i]);
    else if (arg === '--out') parsed.out = argv[++i];
    else throw new Error(`Unknown argument '${arg}'.`);
  }
  if (!parsed.slug) throw new Error('Pass --slug <example>.');
  return parsed;
}
