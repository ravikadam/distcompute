/**
 * Generate a compiled .dsl file (forward + backward) for a GPT model, plus a
 * manifest describing every tensor so a worker/trainer can allocate and run it.
 *
 *   npx ts-node src/tools/gen_gpt1_dsl.ts
 *
 * Produces, under models/:
 *   gpt1.dsl / gpt1.manifest.json        — canonical GPT-1 (per-task B=1, S=512)
 *   tiny-gpt.dsl / tiny-gpt.manifest.json — browser-feasible Tiny-GPT
 *
 * The manifest lists parameters, inputs (token ids, position ids, causal mask,
 * targets) and the full shape table needed to allocate the VM.
 */
import * as fs from 'fs';
import * as path from 'path';
import { Compiler } from '../compiler/compiler';
import { GPT, GPTConfig } from '../compiler/modules';
import { Tensor, TensorVM } from '../public/vm';

interface GenOptions { name: string; cfg: GPTConfig; batch: number; seq: number; }

function generate(opts: GenOptions) {
  const { cfg, batch: B, seq: S } = opts;
  const N = B * S;
  const c = new Compiler();
  const tok = c.createNode('tok', [N]);
  const pos = c.createNode('pos', [N]);
  const mask = c.createNode('mask', [1, S, S]);
  const targets = c.createNode('targets', [N]);

  const gpt = new GPT(c, cfg);
  const logits = gpt.forward(c, tok, pos, mask, B, S);
  const { loss } = c.cross_entropy(logits, targets);
  const { dsl, shapes } = c.compile(loss, targets);

  const params = c.parameters.map(p => ({ name: p.name, shape: p.shape }));
  const paramCount = params.reduce((acc, p) => acc + p.shape.reduce((a, b) => a * b, 1), 0);

  const manifest = {
    name: opts.name,
    config: cfg,
    batch: B,
    seq: S,
    tokensPerTask: N,
    parameterCount: paramCount,
    instructionCount: dsl.split('\n').filter(l => l.trim() && !l.trim().startsWith('#')).length,
    inputs: [
      { name: 'tok', shape: [N], role: 'token ids (0..vocab-1)', dtype: 'fp32' },
      { name: 'pos', shape: [N], role: 'position ids (0..seq-1 per sequence)', dtype: 'fp32' },
      { name: 'mask', shape: [1, S, S], role: 'additive causal mask: 0 if key<=query else -1e9', dtype: 'fp32' },
      { name: 'targets', shape: [N], role: 'next-token class indices', dtype: 'fp32' }
    ],
    outputs: { logits: logits.name, loss: loss.name },
    parameters: params,
    shapes
  };

  fs.mkdirSync(path.join(process.cwd(), 'models'), { recursive: true });
  const dslPath = path.join('models', `${opts.name}.dsl`);
  const manPath = path.join('models', `${opts.name}.manifest.json`);
  fs.writeFileSync(dslPath, dsl);
  fs.writeFileSync(manPath, JSON.stringify(manifest, null, 2));
  console.log(`[gen] ${opts.name}: ${(paramCount / 1e6).toFixed(2)}M params, ${manifest.instructionCount} instructions -> ${dslPath}`);
  return { dsl, shapes, manifest, lossName: loss.name };
}

// Sanity-run a generated model in the VM to confirm the .dsl executes and the
// loss is finite (only for small models — the canonical one is too big to run).
function smokeRun(name: string, dsl: string, shapes: Record<string, number[]>, lossName: string, B: number, S: number) {
  const vm = new TensorVM() as any;
  for (const [n, s] of Object.entries(shapes)) vm.tensors[n] = new Tensor(s);
  for (const n of Object.keys(shapes)) {
    if (n.startsWith('g_') || n.startsWith('t')) continue;
    const t = vm.tensors[n]; if (!t) continue;
    if (n.endsWith('_g')) t.data.fill(1);
    else if (n.endsWith('_b')) t.data.fill(0);
    else if (n === 'wte' || n === 'wpe' || n.endsWith('_w')) for (let i = 0; i < t.data.length; i++) t.data[i] = (Math.random() - 0.5) * 0.1;
  }
  const N = B * S;
  for (let i = 0; i < N; i++) { vm.tensors['tok'].data[i] = i % shapes['wte'][0]; vm.tensors['pos'].data[i] = i % S; vm.tensors['targets'].data[i] = (i + 1) % shapes['wte'][0]; }
  const m = vm.tensors['mask'].data;
  for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) m[i * S + j] = j <= i ? 0 : -1e9;
  vm.execute(dsl);
  const l = vm.tensors[lossName].data[0];
  console.log(`[gen] ${name}: smoke-run loss = ${l.toFixed(4)} (${Number.isFinite(l) ? 'OK' : 'NON-FINITE!'})`);
  return Number.isFinite(l);
}

// Canonical GPT-1 — one sequence per task (B=1) so per-task memory is minimal.
generate({
  name: 'gpt1',
  cfg: { vocab: 40478, context: 512, dModel: 768, nHead: 12, dFF: 3072, nLayer: 12 },
  batch: 1,
  seq: 512
});

// Tiny-GPT — feasible on browser workers; also smoke-run to prove the file works.
const tiny = generate({
  name: 'tiny-gpt',
  cfg: { vocab: 256, context: 128, dModel: 128, nHead: 4, dFF: 512, nLayer: 4 },
  batch: 4,
  seq: 128
});
smokeRun('tiny-gpt', tiny.dsl, tiny.shapes, tiny.lossName, 4, 128);
