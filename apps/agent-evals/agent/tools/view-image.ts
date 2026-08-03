import { defineTool, toolOutput, toolOutputPart } from "eve/tools";
import { z } from "zod";

/**
 * Let the agent LOOK at an image it produced.
 *
 * Read `agent/agent.ts` first: that file explains why this suite ships no
 * vgpu-aware tools, and why this one is the single, deliberately narrow
 * exception. Nothing here names vgpu, doctor, docs, shaders or hover effects —
 * it is the generic "see the pixels, do not guess" affordance a real coding
 * agent already has through its IDE or chat UI.
 *
 * The tool NAME is its filename, `view-image` (eve's own fixture proves this:
 * `agent/tools/render-stripes.ts` is asserted as `calledTool("render-stripes")`).
 * Any eval matching on this tool must use the same kebab-case spelling.
 */
const MEDIA_TYPES: Record<string, string> = {
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
};

function mediaTypeFor(path: string): string {
  const extension = path.slice(path.lastIndexOf(".")).toLowerCase();
  const mediaType = MEDIA_TYPES[extension];
  if (!mediaType) {
    // Guessing would hand the model a corrupt image and a confusing dead end.
    // A named error is something it can act on.
    throw new Error(
      `view-image: unsupported image type "${extension}" (supported: ${Object.keys(MEDIA_TYPES).join(", ")})`,
    );
  }
  return mediaType;
}

export default defineTool({
  description:
    "View an image file in your workspace (for example a PNG you just rendered) so you can see " +
    "its actual pixels instead of only its byte size. `path` is relative to /workspace.",
  inputSchema: z.object({
    path: z.string().describe('Path relative to /workspace, for example "out.png".'),
  }),
  async execute({ path }, ctx) {
    const sandbox = await ctx.getSandbox();
    const resolved = path.startsWith("/") ? path : `/workspace/${path}`;
    const mediaType = mediaTypeFor(resolved);
    const bytes = await sandbox.readBinaryFile({ path: resolved });
    if (!bytes) {
      throw new Error(`view-image: no file at ${resolved}`);
    }
    return {
      path: resolved,
      mediaType,
      imageBase64: Buffer.from(bytes).toString("base64"),
    };
  },
  // The image reaches the model here, as a file content part. Returning it from
  // `execute` alone would only put base64 text in the transcript.
  toModelOutput(output) {
    return toolOutput.content([
      toolOutputPart.text(`Viewing ${output.path}:`),
      toolOutputPart.file(output.imageBase64, {
        filename: output.path.split("/").pop(),
        mediaType: output.mediaType,
      }),
    ]);
  },
});
