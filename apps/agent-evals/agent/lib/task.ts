import { readdirSync } from "node:fs";
import { tasksDir } from "./paths.ts";

/**
 * WHICH TASK IS THIS PROCESS RUNNING?
 *
 * `defineSandbox`'s configuration — backend, revalidation key, bootstrap — is
 * evaluated once per OS process and shared by every eval that process runs. Two
 * tasks now need genuinely different seeds and different bootstrap work, so one
 * process must run exactly one task, and that choice arrives as an environment
 * variable rather than being inferred from whatever eval happens to execute
 * first.
 *
 * `scripts/agent-evals.mjs --task <id>` sets it and derives the eval filter from
 * the same flag, so the variable and the filter cannot drift apart.
 */
export function knownTaskIds(): string[] {
  try {
    return readdirSync(tasksDir(), { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort();
  } catch {
    // The directory is only unreachable when the seed path could not be
    // resolved (see tasksDir()). Returning [] keeps the error messages below
    // useful instead of replacing a clear "task not set" with an ENOENT.
    return [];
  }
}

export function requireTaskId(): string {
  const taskId = process.env.VGPU_EVALS_TASK;
  const known = knownTaskIds();
  const listing = known.length > 0 ? known.join(", ") : "(none discovered)";

  if (!taskId) {
    throw new Error(
      "VGPU_EVALS_TASK is not set, so the sandbox does not know which task's " +
        "workspace to seed.\n" +
        `  Known tasks: ${listing}\n` +
        "  Run `pnpm agent-evals --task <id>`, which sets it and scopes the eval " +
        "filter to match.",
    );
  }
  if (known.length > 0 && !known.includes(taskId)) {
    throw new Error(
      `VGPU_EVALS_TASK is "${taskId}", which has no seed directory under ` +
        "agent/sandbox/tasks/.\n" +
        `  Known tasks: ${listing}`,
    );
  }
  return taskId;
}
