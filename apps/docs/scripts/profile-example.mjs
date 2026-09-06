// Headless per-pass profiler: bundles an example that exports `profile(gpu,
// target, frames)` and prints median milliseconds per pass.
//
//   node scripts/profile-example.mjs --slug spaceship-thrusters --size 1280x720 --frames 20
//
// Numbers from a software Vulkan driver (lavapipe) are CPU rasterization
// times; they are only meaningful relative to each other.

import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { init } from 'vgpu/node';
import { transformWgsl } from '@vgpu/wgsl/loader-vite';

const args = parseArgs(process.argv.slice(2));
const docsDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const cacheDir = path.join(docsDir, '.profile-cache');

const profile = await loadProfile(args.slug);
const gpu = await init();
try {
  const target = gpu.target({ size: args.size, format: 'rgba8unorm', label: `profile-${args.slug}` });
  const result = await profile(gpu, target, args.frames);
  const total = Object.values(result.passes).reduce((sum, ms) => sum + ms, 0);
  console.log(`${args.slug} @ ${result.size[0]}x${result.size[1]} (fire ${result.fireSize[0]}x${result.fireSize[1]}), median of ${result.frames} frames:`);
  for (const [name, ms] of Object.entries(result.passes)) console.log(`  ${name.padEnd(10)} ${ms.toFixed(2).padStart(8)} ms  ${(100 * ms / total).toFixed(0).padStart(3)}%`);
  console.log(`  ${'total'.padEnd(10)} ${total.toFixed(2).padStart(8)} ms`);
} finally {
  gpu.dispose();
  await rm(cacheDir, { recursive: true, force: true });
}

async function loadProfile(slug) {
  await mkdir(cacheDir, { recursive: true });
  const entry = path.join(cacheDir, 'entry.ts');
  const bundle = path.join(cacheDir, 'bundle.mjs');
  await writeFile(entry, `export { profile } from '../examples/${slug}/example';\n`);
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
  if (typeof module.profile !== 'function') throw new Error(`Example '${slug}' does not export profile.`);
  return module.profile;
}

function parseArgs(argv) {
  const parsed = { slug: undefined, size: [1280, 720], frames: 20 };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--') continue;
    else if (arg === '--slug') parsed.slug = argv[++i];
    else if (arg === '--size') parsed.size = argv[++i].split('x').map(Number);
    else if (arg === '--frames') parsed.frames = Number(argv[++i]);
    else throw new Error(`Unknown argument '${arg}'.`);
  }
  if (!parsed.slug) throw new Error('Pass --slug <example>.');
  return parsed;
}
