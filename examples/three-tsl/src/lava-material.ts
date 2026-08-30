import * as THREE from "three/webgpu";
import { float, fwidth, mix, normalLocal, positionLocal, smoothstep, time, transformNormalToView, uniform, vec3 } from "three/tsl";
import type { Node } from "three/webgpu";
import lavaModule from "./lava.wgsl";
import { tslExports } from "./wgsl-tsl.ts";

const { lavaGlow, blackbody, crustHeight, crustRelief, crustSurface, crustPbr, lavaSink, microDetail, meltSkin } = tslExports(lavaModule, [
  "lavaGlow",
  "blackbody",
  "crustHeight",
  "crustRelief",
  "crustSurface",
  "crustPbr",
  "lavaSink",
  "microDetail",
  "meltSkin",
]);

export interface LavaMaterialOptions {
  /** Drives the flow animation; defaults to the TSL `time` node. */
  readonly timeNode?: Node;
}

export interface LavaMaterial {
  readonly material: THREE.MeshPhysicalNodeMaterial;
  /** Emissive strength of the molten channels. */
  readonly glowIntensity: ReturnType<typeof uniform<number>>;
  /** Spatial frequency of the lava field on the mesh. */
  readonly scale: ReturnType<typeof uniform<number>>;
}

/**
 * Cooling basalt crust over an incandescent molten interior. The heat field,
 * crust relief, and blackbody ramp all come from lava.wgsl through the vgpu
 * loader; three only sees TSL nodes.
 */
export function createLavaMaterial(options: LavaMaterialOptions = {}): LavaMaterial {
  const glowIntensity = uniform(1.6);
  const scale = uniform(1.0);
  const t = options.timeNode ?? time;

  const p = positionLocal.mul(scale);
  // The whole glow composition (laminar melt with striations and contact
  // rims, plus the grain-seeped ember fringe) lives in lava.wgsl: x = heat,
  // y = continuous-melt mask.
  const glow = lavaGlow({ position: p, t });
  const heat = glow.x;
  const molten = glow.y;
  const height = crustHeight({ position: p, t });
  const surface = crustSurface({ position: p, t });
  const tone = surface.x;
  const oxide = surface.y;
  const glass = surface.z;
  const pits = surface.w;

  // High-frequency surface detail: mineral grain plus flow-line streaks
  // frozen into the glassy skin. Shared by the roughness map and the micro
  // normal pass below.
  const micro = microDetail({ position: p });
  const grain = micro.x;
  const streaks = micro.y;

  // Band-limiting: fade each detail register out as its wavelength drops
  // under the pixel footprint, so distant/minified areas stay clean instead
  // of dissolving into per-pixel speckle.
  const footprint = fwidth(p).length();
  const microFade = smoothstep(0.022, 0.007, footprint);
  const striaeFade = smoothstep(0.012, 0.004, footprint);

  const material = new THREE.MeshPhysicalNodeMaterial({ metalness: 0 });

  // Basalt skin: warm dusty grey-brown, ridges catching more light than the
  // fissured low ground, rust staining on older patches, and pits plus
  // flake seams going almost black.
  const ash = mix(vec3(0.045, 0.04, 0.036), vec3(0.19, 0.17, 0.15), tone);
  const ridgeLight = height.mul(height).mul(0.6).add(0.6);
  const stained = mix(ash.mul(ridgeLight), vec3(0.20, 0.085, 0.05), oxide.mul(0.5));
  const basalt = mix(stained, stained.mul(0.4), pits);
  material.colorNode = mix(basalt, vec3(0.012, 0.01, 0.009), molten);

  // Incandescence: blackbody ramp over the composed heat field, crushed
  // slightly so contact rims go yellow-white while striation crests cool
  // through deep red.
  material.emissiveNode = blackbody({ t: heat.pow(1.35) }).mul(glowIntensity);

  // Roughness map, not a constant: rubble is matte with sharp grain breakup,
  // the glassy skin is polished but streaked by flow lines, vesicle pits and
  // dusty valleys scatter more, and molten rock is a glossy liquid.
  const crustRoughness = mix(float(0.94), float(0.55), glass)
    .add(grain.sub(0.5).mul(microFade.mul(0.14)))
    .add(streaks.sub(0.5).mul(0.12).mul(glass))
    .add(pits.mul(0.08))
    .add(height.oneMinus().mul(0.05));
  const moltenRoughness = float(0.32).add(streaks.sub(0.5).mul(0.1));
  material.roughnessNode = mix(crustRoughness, moltenRoughness, molten).clamp(0.05, 1);
  material.clearcoatNode = glass.mul(0.25).mul(molten.oneMinus());
  material.clearcoatRoughnessNode = float(0.22).add(grain.sub(0.5).mul(0.15)).clamp(0.05, 1);

  // PBR refinement, all from WGSL: cavity occlusion keeps crevices dark
  // under the environment light, specular mottling breaks up the sheen,
  // glinting mineral facets read as tiny metallic flakes, and the glassy
  // skin gets a faint thin-film iridescence.
  const pbr = crustPbr({ position: p, t });
  const cavity = pbr.x;
  const irid = pbr.y;
  const specMottle = pbr.z;
  const facets = pbr.w;
  material.aoNode = cavity;
  material.specularIntensityNode = mix(specMottle, float(1), molten);
  material.metalnessNode = facets.mul(glass.mul(0.25).add(0.05)).mul(molten.oneMinus());
  material.iridescenceNode = irid.mul(glass).mul(0.15);
  material.iridescenceIORNode = float(2.0);
  material.iridescenceThicknessNode = irid.mul(250).add(150);

  // Plates bulge up, molten channels sink. Vertices only see smooth fields:
  // crustRelief has no per-cell flake plateaus (those would stair-step on
  // the mesh grid) and the sink mask is wide and low-frequency.
  const sink = lavaSink({ position: p, t });
  const relief = crustRelief({ position: p, t }).mul(0.5).sub(sink.mul(0.4)).mul(0.12);
  material.positionNode = positionLocal.add(normalLocal.mul(relief));

  // Bump normals in two registers, both by finite differences projected onto
  // the surface: the crust height field at a coarse epsilon for plates and
  // ropes, and the micro grain at a fine epsilon for crisp mineral detail.
  const eps = 0.03;
  const grad = vec3(
    crustHeight({ position: p.add(vec3(eps, 0, 0)), t }).sub(height),
    crustHeight({ position: p.add(vec3(0, eps, 0)), t }).sub(height),
    crustHeight({ position: p.add(vec3(0, 0, eps)), t }).sub(height),
  ).div(eps);
  const tangentGrad = grad.sub(normalLocal.mul(grad.dot(normalLocal)));

  const microEps = 0.005;
  const microGrad = vec3(
    microDetail({ position: p.add(vec3(microEps, 0, 0)) }).x.sub(grain),
    microDetail({ position: p.add(vec3(0, microEps, 0)) }).x.sub(grain),
    microDetail({ position: p.add(vec3(0, 0, microEps)) }).x.sub(grain),
  ).div(microEps);
  const microTangent = microGrad.sub(normalLocal.mul(microGrad.dot(normalLocal)));

  // The cooled skin bands are physical ridges on the melt: finite-difference
  // the skin field so they emboss the surface — molten areas only, the rock
  // never carries these lines.
  const skinEps = 0.006;
  const skinBase = meltSkin({ position: p, t });
  const skinGrad = vec3(
    meltSkin({ position: p.add(vec3(skinEps, 0, 0)), t }).sub(skinBase),
    meltSkin({ position: p.add(vec3(0, skinEps, 0)), t }).sub(skinBase),
    meltSkin({ position: p.add(vec3(0, 0, skinEps)), t }).sub(skinBase),
  ).div(skinEps);
  const skinTangent = skinGrad.sub(normalLocal.mul(skinGrad.dot(normalLocal)));

  const bumped = normalLocal
    .sub(tangentGrad.mul(mix(float(0.16), float(0.04), molten)))
    .sub(microTangent.mul(mix(float(0.022), float(0.008), molten).mul(microFade)))
    .sub(skinTangent.mul(molten.mul(0.014).mul(striaeFade)))
    .normalize();
  material.normalNode = transformNormalToView(bumped);

  // The clearcoat is the frozen glass skin draped over the rock: it follows
  // the plates but not the mineral grain, so it gets its own smoother normal.
  const skinNormal = normalLocal.sub(tangentGrad.mul(0.1)).normalize();
  material.clearcoatNormalNode = transformNormalToView(skinNormal);

  return { material, glowIntensity, scale };
}
