import { fileURLToPath } from "node:url";
import { resolveShader } from "@vgpu/wgsl/runtime";
import { vec3 } from "three/tsl";
import { describe, expect, it } from "vitest";
import { forwardingWrapper, parseFunctionHeader, tslExports } from "../src/wgsl-tsl.ts";

const SOURCE = `
// A helper the wrapper must skip: fn decoy(x: f32) -> f32
fn valueNoise3(position: vec3f) -> f32 { return position.x; }

/* block comment with fn inside(a: f32) -> f32 */
fn fbm3(position: vec3f, octaves: u32) -> f32 {
  return valueNoise3(position) * f32(octaves);
}

fn remap(inRange: vec2<f32>, outRange: vec2<f32>, value: f32) -> f32 {
  return outRange.x + (value - inRange.x) * (outRange.y - outRange.x) / (inRange.y - inRange.x);
}

fn logOnly() { return; }
`;

describe("parseFunctionHeader", () => {
  it("reads name, params, and return type", () => {
    const header = parseFunctionHeader(SOURCE, "fbm3");
    expect(header.params).toBe("position: vec3f, octaves: u32");
    expect(header.paramNames).toEqual(["position", "octaves"]);
    expect(header.returnType).toBe("f32");
  });

  it("keeps generic parameter types intact", () => {
    const header = parseFunctionHeader(SOURCE, "remap");
    expect(header.paramNames).toEqual(["inRange", "outRange", "value"]);
    expect(header.returnType).toBe("f32");
  });

  it("supports functions without a return type", () => {
    const header = parseFunctionHeader(SOURCE, "logOnly");
    expect(header.params).toBe("");
    expect(header.paramNames).toEqual([]);
    expect(header.returnType).toBe("");
  });

  it("ignores fn mentions inside comments", () => {
    expect(() => parseFunctionHeader(SOURCE, "decoy")).toThrow(/no function named decoy/);
    expect(() => parseFunctionHeader(SOURCE, "inside")).toThrow(/no function named inside/);
  });
});

describe("forwardingWrapper", () => {
  it("forwards every parameter to the wrapped function", () => {
    const wrapper = forwardingWrapper(parseFunctionHeader(SOURCE, "fbm3"));
    expect(wrapper).toBe(
      "fn fbm3_vtsl(position: vec3f, octaves: u32) -> f32 { return fbm3(position, octaves); }",
    );
  });
});

describe("tslExports over a vgpu-resolved module", () => {
  it("wraps every lava.wgsl export, including functions from imported modules", async () => {
    const entry = fileURLToPath(new URL("../src/lava.wgsl", import.meta.url));
    const resolved = await resolveShader({ entry });

    const names = ["lavaGlow", "meltSkin", "blackbody", "crustHeight", "crustSurface", "crustPbr", "lavaSink", "microDetail"] as const;
    const nodes = tslExports(resolved.wgsl, names);
    for (const name of names) expect(typeof nodes[name]).toBe("function");

    // lavaGlow's signature survives the flatten+mangle round trip.
    const header = parseFunctionHeader(resolved.wgsl, "lavaGlow");
    expect(header.paramNames).toEqual(["position", "t"]);
    expect(header.returnType).toBe("vec2f");
  });

  it("wraps the flattened module graph by authored names", async () => {
    const entry = fileURLToPath(new URL("../src/lava.wgsl", import.meta.url));
    const resolved = await resolveShader({ entry });

    // Non-entry-point functions are mangled per module; export keywords are gone.
    expect(resolved.wgsl).toMatch(/fn _vgsl_[0-9a-f]{8}__lavaGlow\(/);
    expect(resolved.wgsl).toMatch(/_vgsl_[0-9a-f]{8}__voronoi3d/);
    expect(resolved.wgsl).not.toMatch(/\bexport\b/);

    // The helper resolves functions by their authored names.
    const header = parseFunctionHeader(resolved.wgsl, "blackbody");
    expect(header.resolvedName).toMatch(/^_vgsl_[0-9a-f]{8}__blackbody$/);
    expect(header.paramNames).toEqual(["t"]);
    expect(forwardingWrapper(header)).toContain(`return ${header.resolvedName}(t);`);

    // wgslFn returns a callable; invoking it with named inputs builds a call node.
    const nodes = tslExports(resolved.wgsl, ["lavaGlow", "blackbody"]);
    expect(typeof nodes.blackbody).toBe("function");
    const call = nodes.lavaGlow({ position: vec3(0, 0, 0), t: 6 });
    expect(call.isNode).toBe(true);
  });
});
