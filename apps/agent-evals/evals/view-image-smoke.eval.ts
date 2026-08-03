import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";
import { defineEval } from "eve/evals";
import { equals, satisfies } from "eve/evals/expect";
import { PNG } from "pngjs";
import { snapshotTarPath, snapshotWorkspaceDir } from "../agent/lib/paths.ts";
import { turnFailure } from "./lib/turn-failure.mjs";

/**
 * Fast, cheap, ~1-turn infra smoke test for `agent/tools/view-image.ts` — a
 * future CI candidate given its cost. Direct template: eve's own
 * `to-model-output-content-parts.eval.ts` fixture, which proves the same
 * `toModelOutput` content-part mechanism for `render-stripes`.
 *
 * The tool's NAME is its filename, `view-image` (kebab-case — eve derives the
 * tool name from the file, not from any string inside it; see
 * `agent/tools/view-image.ts`'s own comment). Any assertion here must use that
 * exact spelling, not the plan's original `view_image` placeholder.
 */
const TOOL_NAME = "view-image";

const PROMPT =
  "Use the `view-image` tool to look at known.png in /workspace, then reply " +
  "with the two colors you see, left to right, comma-separated.";

/**
 * Same six single-token colors `agent/sandbox/sandbox.ts`'s `TASK_EXTRAS`
 * bootstrap step for this task draws from (and the same palette eve's own
 * `render-stripes` fixture uses) — single tokens no model paraphrases (unlike
 * cyan/teal). Kept as a literal here, not imported: the source of truth is a
 * one-line `node -e` string embedded in `sandbox.ts`, not an importable value,
 * so this is a deliberate, small duplication rather than a shared module for
 * six numbers that essentially never change.
 */
const PALETTE: Record<string, readonly [number, number, number]> = {
  black: [0, 0, 0],
  blue: [0, 0, 255],
  green: [0, 160, 0],
  orange: [255, 140, 0],
  red: [255, 0, 0],
  yellow: [255, 220, 0],
};

const KNOWN_PNG_SIZE = 64;

/** Nearest named color in PALETTE, by squared Euclidean distance. */
function nearestColorName(pixel: readonly [number, number, number]): string {
  let best = "";
  let bestDistance = Infinity;
  for (const [name, color] of Object.entries(PALETTE)) {
    const distance = color.reduce((sum, channel, index) => sum + (channel - pixel[index]!) ** 2, 0);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = name;
    }
  }
  return best;
}

function pixelAt(png: PNG, x: number, y: number): readonly [number, number, number] {
  const index = (png.width * y + x) << 2;
  return [png.data[index]!, png.data[index + 1]!, png.data[index + 2]!];
}

/** True when `reply` names `colors`, in order, as whole words. */
function namesColorsInOrder(reply: string, colors: readonly string[]): boolean {
  const pattern = new RegExp(colors.map((color) => `\\b${color}\\b`).join("[\\s\\S]*"), "iu");
  return pattern.test(reply);
}

export default defineEval({
  description: "view-image-smoke: the view-image tool actually delivers pixels to the model",

  async test(t) {
    // Credentials first, before anything is spent. A missing key must skip,
    // not fail — see s2-gradient.eval.ts for the same reasoning.
    if (!process.env.AI_GATEWAY_API_KEY && !process.env.VERCEL_OIDC_TOKEN) {
      t.skip("no AI Gateway credential (set AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN)");
    }
    // The export hook writes the workspace tar to the HOST running the agent
    // runtime; against a remote target that host is not this machine.
    if (t.target.kind !== "local") {
      t.skip(`workspace export requires a local target (got ${t.target.kind})`);
    }

    const startedAt = Date.now();
    const turn = await t.send(PROMPT);

    // Same load-bearing guard as s2-gradient.eval.ts, copied verbatim: a model
    // that never answered (restricted provider, gateway outage, ...) is not an
    // agent that failed the task, and this must be checked before any gate
    // below or that distinction is lost. See s2-gradient.eval.ts's comment for
    // the two reasons this throws instead of `t.skip()`.
    if (turn.status === "failed") {
      throw new Error(`model/infra failure, not an agent result: ${turnFailure(turn.events)}`);
    }

    // ---- Gate (hard): the plumbing worked, at all --------------------------
    // The real fact under test, per eve's own render-stripes precedent: the
    // tool was called exactly once and actually delivered an image, not just
    // that the model produced some reply.
    t.calledTool(TOOL_NAME, { count: 1 });

    // ---- Ground truth, computed independently by the harness ---------------
    // known.png is generated randomly at bootstrap time (see sandbox.ts's
    // TASK_EXTRAS for "view-image-smoke"), so the correct answer is not known
    // until the workspace is exported and decoded here — never trusted from
    // the agent's own reply.
    const sessionId = turn.sessionId;
    const tarPath = snapshotTarPath(sessionId);
    await t.require(existsSync(tarPath), equals(true));
    t.check(statSync(tarPath).mtimeMs >= startedAt, equals(true))
      .gate()
      .label("workspace export is from this turn");

    const extracted = snapshotWorkspaceDir(sessionId);
    rmSync(extracted, { force: true, recursive: true });
    mkdirSync(extracted, { recursive: true });
    const untar = spawnSync("tar", ["-xf", tarPath, "-C", extracted], { encoding: "utf8" });
    if (untar.status !== 0) {
      throw new Error(`could not extract ${tarPath}: ${untar.stderr}`);
    }

    const knownPngPath = join(extracted, "known.png");
    await t.require(existsSync(knownPngPath), equals(true));
    const png = PNG.sync.read(readFileSync(knownPngPath));

    const half = KNOWN_PNG_SIZE / 2;
    const leftName = nearestColorName(pixelAt(png, Math.floor(half / 2), half));
    const rightName = nearestColorName(pixelAt(png, Math.floor(half + half / 2), half));
    t.log(`known.png ground truth: left=${leftName} right=${rightName}`);

    // ---- Soft: did it actually see the right colors ------------------------
    // Vision quality varies across models and runs; this is tracked, not
    // gated, mirroring eve's own render-stripes fixture's stated policy
    // ("Color recognition remains tracked rather than gated because live
    // vision quality varies").
    t.log(`reply: ${t.reply ?? "(no reply)"}`);
    t.check(
      t.reply ?? "",
      satisfies<string>(
        (reply) => namesColorsInOrder(reply, [leftName, rightName]),
        "reply names both colors in order",
      ),
    )
      .soft()
      .label("reply names both randomized colors, left to right");
  },
});
