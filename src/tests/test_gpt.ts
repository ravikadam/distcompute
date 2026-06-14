/**
 * Gradient checks for the GPT building blocks (embedding, LayerNorm) and a
 * full tiny end-to-end GPT, validating the new compiler ops + VM kernels
 * against finite differences before we trust the generated gpt1.dsl.
 */
import { Compiler } from '../compiler/compiler';
import { GPT, GPTConfig } from '../compiler/modules';
import { Tensor, TensorVM } from '../public/vm';

let passed = 0, failed = 0;
function ok(name: string, cond: boolean, detail = '') {
  if (cond) { console.log(`✓ ${name}`); passed++; }
  else { console.error(`✗ ${name} ${detail}`); failed++; }
}

// Finite-difference check: for the named parameter tensor, compare analytic
// grad (g_<name>) against numerical grad of the scalar loss.
function gradCheck(label: string, vm: any, dsl: string, lossName: string, paramName: string, tol = 2e-2) {
  vm.execute(dsl);
  const analytic = Array.from(vm.tensors['g_' + paramName].data) as number[];
  const p = vm.tensors[paramName].data;
  const eps = 1e-3;
  let maxErr = 0;
  for (let i = 0; i < p.length; i++) {
    const orig = p[i];
    p[i] = orig + eps; vm.execute(dsl); const lp = vm.tensors[lossName].data[0];
    p[i] = orig - eps; vm.execute(dsl); const lm = vm.tensors[lossName].data[0];
    p[i] = orig;
    const num = (lp - lm) / (2 * eps);
    maxErr = Math.max(maxErr, Math.abs(num - analytic[i]));
  }
  ok(`${label}: grad(${paramName}) matches finite-diff`, maxErr < tol, `maxErr=${maxErr.toExponential(2)}`);
}

function alloc(vm: any, shapes: Record<string, number[]>) {
  for (const [n, s] of Object.entries(shapes)) vm.tensors[n] = new Tensor(s);
}
function rand(t: any, scale = 0.5) { for (let i = 0; i < t.data.length; i++) t.data[i] = (Math.random() - 0.5) * 2 * scale; }

// --- LayerNorm ---
function testLayerNorm() {
  const c = new Compiler();
  const X = c.createNode('X', [3, 4], true);
  const targets = c.createNode('targets', [3]);
  const gamma = c.registerParameter('ln_g', [4]);
  const beta = c.registerParameter('ln_b', [4]);
  const y = c.layernorm(X, gamma, beta);
  const { loss } = c.cross_entropy(y, targets);
  const { dsl, shapes } = c.compile(loss, targets);
  const vm = new TensorVM() as any;
  alloc(vm, shapes);
  rand(vm.tensors['X']);
  vm.tensors['ln_g'].data.set([1, 1, 1, 1]);
  vm.tensors['ln_b'].data.set([0, 0, 0, 0]);
  vm.tensors['targets'].data.set([1, 3, 0]);
  gradCheck('layernorm', vm, dsl, loss.name, 'ln_g');
  gradCheck('layernorm', vm, dsl, loss.name, 'ln_b');
  gradCheck('layernorm', vm, dsl, loss.name, 'X');
}

// --- Embedding ---
function testEmbedding() {
  const c = new Compiler();
  const ids = c.createNode('ids', [3]);
  const targets = c.createNode('targets', [3]);
  const table = c.registerParameter('wte', [5, 4]); // V=5, d=4
  const emb = c.embedding(table, ids);              // [3, 4]
  const { loss } = c.cross_entropy(emb, targets);
  const { dsl, shapes } = c.compile(loss, targets);
  const vm = new TensorVM() as any;
  alloc(vm, shapes);
  rand(vm.tensors['wte']);
  vm.tensors['ids'].data.set([2, 4, 0]);
  vm.tensors['targets'].data.set([1, 3, 2]);
  gradCheck('embedding', vm, dsl, loss.name, 'wte');
}

// --- Full tiny GPT (end-to-end, incl. weight-tied head + causal attention) ---
function testTinyGPT() {
  const cfg: GPTConfig = { vocab: 10, context: 4, dModel: 8, nHead: 2, dFF: 16, nLayer: 2 };
  const B = 2, S = 3, N = B * S;
  const c = new Compiler();
  const tok = c.createNode('tok', [N]);
  const pos = c.createNode('pos', [N]);
  const mask = c.createNode('mask', [1, S, S]);
  const targets = c.createNode('targets', [N]);
  const gpt = new GPT(c, cfg);
  const logits = gpt.forward(c, tok, pos, mask, B, S);
  const { loss } = c.cross_entropy(logits, targets);
  const { dsl, shapes } = c.compile(loss, targets);

  const vm = new TensorVM() as any;
  alloc(vm, shapes);
  // Initialise parameters by naming convention.
  for (const name of Object.keys(shapes)) {
    if (name.startsWith('g_') || name.startsWith('t')) continue;
    const t = vm.tensors[name];
    if (!t) continue;
    if (name.endsWith('_g')) t.data.fill(1);           // LayerNorm gain
    else if (name.endsWith('_b')) t.data.fill(0);      // biases / LN beta
    else if (name === 'wte' || name === 'wpe') rand(t, 0.05);
    else if (name.endsWith('_w')) rand(t, 0.1);        // linear weights
  }
  vm.tensors['tok'].data.set([1, 2, 3, 4, 5, 6]);
  vm.tensors['pos'].data.set([0, 1, 2, 0, 1, 2]);
  vm.tensors['targets'].data.set([2, 3, 4, 5, 6, 7]);
  // Causal additive mask: 0 if key<=query else -1e9.
  const m = vm.tensors['mask'].data;
  for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) m[i * S + j] = j <= i ? 0 : -1e9;

  vm.execute(dsl);
  const lossVal = vm.tensors[loss.name].data[0];
  ok('tiny-gpt: forward loss is finite', Number.isFinite(lossVal), `loss=${lossVal}`);

  // Gradient-check a representative parameter from each subsystem.
  for (const p of ['wte', 'wpe', 'h0_attn_wq_w', 'h0_attn_wo_w', 'h0_mlp_fc_w', 'h0_ln1_g', 'h1_mlp_proj_w', 'ln_f_g']) {
    gradCheck('tiny-gpt', vm, dsl, loss.name, p, 3e-2);
  }
}

console.log('=== GPT Building-Block Gradient Checks ===');
testLayerNorm();
testEmbedding();
testTinyGPT();
console.log(`\n${passed} passed, ${failed} failed.`);
process.exit(failed === 0 ? 0 : 1);
