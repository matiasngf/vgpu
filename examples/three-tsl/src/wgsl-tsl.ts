import type { Node } from "three/webgpu";
import { wgsl, wgslFn } from "three/tsl";

/**
 * Bridges vgpu-resolved WGSL modules into three.js TSL nodes.
 *
 * The `@vgpu/wgsl` loader emits `{ version: 1, wgsl }` where `wgsl` is the
 * flattened module graph: a valid top-level WGSL library with no bindings.
 * Non-entry-point functions are mangled to `_vgsl_<pathHash>__<name>` to avoid
 * cross-module collisions, so this helper locates a function by its authored
 * name allowing an optional mangle prefix, and errors when two modules in the
 * graph would answer to the same name.
 *
 * TSL's `wgslFn` wraps a single WGSL function and passes every input as a
 * parameter, so for each requested export this helper emits a forwarding
 * wrapper (`fn <name>_vtsl(...)`) that calls the resolved (possibly mangled)
 * function, and attaches the full module as a shared `wgsl()` include.
 * Sharing one include node means the module text is emitted once per shader
 * no matter how many of its functions the material uses.
 */

export interface WgslShaderSource {
  readonly version: 1;
  readonly wgsl: string;
}

export interface WgslFunctionHeader {
  /** The authored name used to look the function up. */
  readonly name: string;
  /** The name as it appears in the resolved WGSL, possibly mangled. */
  readonly resolvedName: string;
  /** Verbatim parameter list, e.g. `position: vec3f, warp: f32`. */
  readonly params: string;
  readonly paramNames: readonly string[];
  /** Verbatim return type; empty string when the function returns nothing. */
  readonly returnType: string;
}

/** Callable TSL node: pass inputs keyed by WGSL parameter name. */
export type WgslFnNode = ReturnType<typeof wgslFn<Record<string, Node | number>>>;

/**
 * Returns one TSL node per named function of a loader-emitted WGSL module.
 *
 * ```ts
 * import marbleModule from "./marble.wgsl";
 * const { fbm3, marble } = tslExports(marbleModule, ["fbm3", "marble"]);
 * material.colorNode = mix(base, vein, marble({ position: positionLocal, warp: 6 }));
 * ```
 */
export function tslExports<Name extends string>(
  source: WgslShaderSource | string,
  names: readonly Name[],
): Record<Name, WgslFnNode> {
  const moduleWgsl = typeof source === "string" ? source : source.wgsl;
  const include = wgsl(moduleWgsl);
  const nodes = {} as Record<Name, WgslFnNode>;
  for (const name of names) {
    const header = parseFunctionHeader(moduleWgsl, name);
    nodes[name] = wgslFn(forwardingWrapper(header), [include]);
  }
  return nodes;
}

/** Builds `fn <name>_vtsl(<params>) -> <ret> { return <resolvedName>(<args>); }`. */
export function forwardingWrapper(header: WgslFunctionHeader): string {
  const call = `${header.resolvedName}(${header.paramNames.join(", ")})`;
  const returns = header.returnType === "" ? `${call};` : `return ${call};`;
  const arrow = header.returnType === "" ? "" : ` -> ${header.returnType}`;
  return `fn ${header.name}_vtsl(${header.params})${arrow} { ${returns} }`;
}

/**
 * Extracts the header of top-level `fn <name>(...)` from WGSL source, where
 * the declared name may carry vgpu's `_vgsl_<pathHash>__` mangle prefix.
 * Scans comment-stripped text, so the same offsets index the original source.
 */
export function parseFunctionHeader(source: string, name: string): WgslFunctionHeader {
  const scannable = blankComments(source);
  const declaration = new RegExp(`\\bfn\\s+((?:_vgsl_[0-9a-f]{8}__)?${name})\\s*\\(`, "g");
  const matches = [...scannable.matchAll(declaration)];
  if (matches.length === 0) throw new Error(`WGSL module has no function named ${name}`);
  const distinctNames = new Set(matches.map((item) => item[1]!));
  if (distinctNames.size > 1) {
    throw new Error(`WGSL module has multiple functions answering to ${name}: ${[...distinctNames].join(", ")}`);
  }
  const match = matches[0]!;
  const resolvedName = match[1]!;

  const paramsStart = match.index + match[0].length;
  let depth = 1;
  let cursor = paramsStart;
  while (cursor < scannable.length && depth > 0) {
    const char = scannable[cursor];
    if (char === "(") depth++;
    else if (char === ")") depth--;
    cursor++;
  }
  if (depth !== 0) throw new Error(`Unbalanced parameter list for WGSL function ${name}`);
  const params = source.slice(paramsStart, cursor - 1).trim();

  const bodyStart = scannable.indexOf("{", cursor);
  if (bodyStart === -1) throw new Error(`Missing body for WGSL function ${name}`);
  const between = source.slice(cursor, bodyStart).trim();
  if (between !== "" && !between.startsWith("->")) {
    throw new Error(`Unsupported header for WGSL function ${name}: ${between}`);
  }
  const returnType = between === "" ? "" : between.slice(2).trim();

  return { name, resolvedName, params, paramNames: parameterNames(params), returnType };
}

/** First identifier of each top-level comma segment (`position: vec3f` -> `position`). */
function parameterNames(params: string): string[] {
  if (params === "") return [];
  const names: string[] = [];
  let depth = 0;
  let segment = "";
  for (const char of params) {
    if (char === "<" || char === "(") depth++;
    else if (char === ">" || char === ")") depth--;
    if (char === "," && depth === 0) {
      names.push(segmentName(segment));
      segment = "";
    } else {
      segment += char;
    }
  }
  names.push(segmentName(segment));
  return names;
}

function segmentName(segment: string): string {
  const colon = segment.indexOf(":");
  const name = (colon === -1 ? segment : segment.slice(0, colon)).trim();
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
    throw new Error(`Cannot read WGSL parameter name from segment: ${segment.trim()}`);
  }
  return name;
}

/** Replaces `//` and (nesting) `/* *\/` comments with spaces, preserving offsets. */
function blankComments(source: string): string {
  let out = "";
  let i = 0;
  while (i < source.length) {
    if (source[i] === "/" && source[i + 1] === "/") {
      while (i < source.length && source[i] !== "\n") { out += " "; i++; }
    } else if (source[i] === "/" && source[i + 1] === "*") {
      let depth = 0;
      do {
        if (source[i] === "/" && source[i + 1] === "*") { depth++; out += "  "; i += 2; }
        else if (source[i] === "*" && source[i + 1] === "/") { depth--; out += "  "; i += 2; }
        else { out += source[i] === "\n" ? "\n" : " "; i++; }
      } while (i < source.length && depth > 0);
    } else {
      out += source[i];
      i++;
    }
  }
  return out;
}
