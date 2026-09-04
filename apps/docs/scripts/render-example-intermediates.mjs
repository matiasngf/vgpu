// Headless debugging aid: renders one example's `renderThumb` with `vgpu/node`
// and writes the final frame plus every intermediate target the example
// reports through `onIntermediateRendered(kind, pixels, size)` as PNGs.
// Examples are responsible for previewing HDR targets into rgba8 first
// (readback only supports 8-bit formats).
//
//   node scripts/render-example-intermediates.mjs --slug spaceship-thrusters \
//     --size 640x360 --time 6.2 --out ../../artifacts/thrusters
//
// Requires a healthy `vgpu doctor` (a software Vulkan driver such as lavapipe
// is enough).

import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { init } from 'vgpu/node';
import { writePng } from '@vgpu/cli/lib/snapshot/png.js';
import { transformWgsl } from '@vgpu/wgsl/loader-vite';

const args = parseArgs(process.argv.slice(2));
const docsDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const cacheDir = path.join(docsDir, '.intermediates-cache');
const outDir = path.resolve(args.out);

await mkdir(outDir, { recursive: true });
const renderThumb = await loadRenderer(args.slug);
const gpu = await init();
const started = performance.now();
try {
  const target = gpu.target({ size: args.size, format: 'rgba8unorm', label: `debug-${args.slug}` });
  await renderThumb(gpu, target, {
    time: args.time,
    warmupFrames: args.warmupFrames,
    dt: 1 / 60,
    onIntermediateRendered: async (kind, pixels, size) => {
      const file = path.join(outDir, `${kind}.png`);
      await writePng(file, pixels, size[0], size[1]);
      console.log(`- ${kind}: ${size[0]}x${size[1]} -> ${path.relative(process.cwd(), file)}`);
    },
  });
  const pixels = await target.read();
  const file = path.join(outDir, 'final.png');
  await writePng(file, pixels, args.size[0], args.size[1]);
  console.log(`- final: ${args.size[0]}x${args.size[1]} -> ${path.relative(process.cwd(), file)} (${((performance.now() - started) / 1000).toFixed(1)}s)`);
} finally {
  gpu.dispose();
  await rm(cacheDir, { recursive: true, force: true });
}

async function loadRenderer(slug) {
  await mkdir(cacheDir, { recursive: true });
  const entry = path.join(cacheDir, 'entry.ts');
  const bundle = path.join(cacheDir, 'bundle.mjs');
  await writeFile(entry, `export { renderThumb } from '../examples/${slug}/example';\n`);
  await build({
    entryPoints: [entry],
    outfile: bundle,
    bundle: true,
    platform: 'node',
    format: 'esm',
    sourcemap: false,
    external: ['vgpu', 'vgpu/node'],
    plugins: [{
      name: 'docs-wgsl',
      setup(b) {
        b.onLoad({ filter: /\.wgsl$/ }, async (file) => {
          const source = await readFile(file.path, 'utf8');
          const result = await transformWgsl({ source, id: file.path });
          return { contents: result.code, loader: 'js', resolveDir: path.dirname(file.path) };
        });
      },
    }],
    logLevel: 'silent',
  });
  const module = await import(`${pathToFileURL(bundle).href}?t=${Date.now()}`);
  if (typeof module.renderThumb !== 'function') throw new Error(`Example '${slug}' does not export renderThumb.`);
  return module.renderThumb;
}

function parseArgs(argv) {
  const parsed = { slug: undefined, size: [640, 360], time: undefined, warmupFrames: 1, out: path.join('..', '..', 'artifacts', 'example-intermediates') };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--') continue;
    else if (arg === '--slug') parsed.slug = argv[++i];
    else if (arg === '--size') parsed.size = argv[++i].split('x').map(Number);
    else if (arg === '--time') parsed.time = Number(argv[++i]);
    else if (arg === '--warmup-frames') parsed.warmupFrames = Number(argv[++i]);
    else if (arg === '--out') parsed.out = argv[++i];
    else throw new Error(`Unknown argument '${arg}'.`);
  }
  if (!parsed.slug) throw new Error('Pass --slug <example>.');
  if (parsed.size.length !== 2 || parsed.size.some((n) => !Number.isInteger(n) || n <= 0)) throw new Error('--size must look like 640x360.');
  return parsed;
}
