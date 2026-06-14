/**
 * Regression tests for the bugs reported during cluster testing.
 *
 * Covered:
 *   1. Kick removes a worker (server-side behaviour).
 *   2. Scheduling scales past 4 workers — one slice per idle worker, up to
 *      thousands, with a configured-batch floor for small clusters.
 *   3. Worker log shows the FULL task id (no truncation).
 *   4. Worker UI "dancing" mitigations are present (tabular figures + a
 *      debounced computing indicator).
 *   5. Dashboard "Kick" uses event delegation so taps survive table re-renders.
 *   6. tsconfig.json no longer uses the deprecated `moduleResolution: "node"`.
 *   7. README documents the npm / tsc troubleshooting notes.
 *
 * 1 & 2 are behavioural tests against the real classes. 3-7 are static-source
 * guards for client/HTML/config changes that can't easily run in Node.
 *
 * Run with:  npx ts-node src/tests/test_bugfixes.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { Orchestrator } from '../server/orchestrator';
import { Trainer } from '../server/trainer';
import { estimateDslFlops } from '../server/dslStats';
import { WandbLogger } from '../server/wandb';
// vm.js is plain JS (UMD); pull in the VM for FP16 transport checks.
const { TensorVM } = require('../public/vm.js');

let passed = 0;
let failed = 0;

function check(name: string, condition: boolean, detail = '') {
  if (condition) {
    console.log(`✓ ${name}`);
    passed++;
  } else {
    console.error(`✗ ${name}${detail ? ' -> ' + detail : ''}`);
    failed++;
  }
}

/**
 * Minimal stand-in for a `ws` WebSocket so we can drive the Orchestrator
 * without opening real sockets. Records sent frames and close() calls, and
 * lets tests emit incoming 'message'/'close' events.
 */
class FakeWS {
  static OPEN = 1;
  readyState = 1;
  sent: any[] = [];
  closed = false;
  private handlers: Record<string, Function[]> = {};
  send(s: string) {
    try { this.sent.push(JSON.parse(s)); } catch { this.sent.push(s); }
  }
  close() { this.closed = true; this.emit('close'); }
  on(ev: string, cb: Function) { (this.handlers[ev] = this.handlers[ev] || []).push(cb); }
  emit(ev: string, ...args: any[]) { (this.handlers[ev] || []).forEach(f => f(...args)); }
}

function registerWorker(orch: Orchestrator, id: string): FakeWS {
  const ws = new FakeWS();
  orch.handleWorkerConnection(ws as any);
  ws.emit('message', JSON.stringify({
    type: 'register',
    workerId: id,
    deviceInfo: { browser: 'Chrome', platform: 'Test', cores: 8, userAgent: 'x' }
  }));
  return ws;
}

// ---------------------------------------------------------------------------
// Bug 1: Kick functionality removes the worker from the cluster.
// ---------------------------------------------------------------------------
function testKick() {
  const orch = new Orchestrator();
  const ws = registerWorker(orch, 'node-kick-1');

  check('kick: worker is registered before kick', orch.workers.has('node-kick-1'));

  orch.kickWorker('node-kick-1');

  check('kick: worker removed from registry', !orch.workers.has('node-kick-1'));
  check('kick: server closed the worker socket', ws.closed === true);
  check('kick: server sent a disconnect frame',
    ws.sent.some((m: any) => m && m.type === 'disconnect'));
}

// ---------------------------------------------------------------------------
// Bug 2: Scheduling scales past the 4th worker.
//   (a) planSlices() creates one slice per idle worker (thousands supported)
//       and never drops below the configured batch floor.
//   (b) the orchestrator actually assigns work to ALL idle workers, not 4.
// ---------------------------------------------------------------------------
function testScheduling() {
  const orch = new Orchestrator();
  const trainer = new Trainer(orch);
  trainer.batchSize = 128; // configured floor = 128 / 32 = 4 slices

  // (a) helper logic
  check('schedule: 6 workers -> 6 slices (was capped at 4)',
    trainer.planSlices(6).numSlices === 6, JSON.stringify(trainer.planSlices(6)));
  check('schedule: 2 workers -> 4 slices (configured floor preserved)',
    trainer.planSlices(2).numSlices === 4);
  check('schedule: scales to thousands of workers',
    trainer.planSlices(2000).numSlices === 2000);
  check('schedule: slice size stays 32 (matches compiled graph)',
    trainer.planSlices(50).sliceSize === 32);

  // (b) end-to-end distribution across 6 workers
  const ids = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6'];
  const sockets = ids.map(id => registerWorker(orch, id));
  // submit one task per worker (planSlices(6) would create 6)
  for (let i = 0; i < ids.length; i++) {
    orch.submitTask('# noop dsl', {}, {});
  }
  const assignedTo = new Set<string>();
  sockets.forEach((ws, i) => {
    if (ws.sent.some((m: any) => m && m.type === 'assign_task')) assignedTo.add(ids[i]);
  });
  check('schedule: all 6 workers received a task (none stuck idle)',
    assignedTo.size === 6, `assigned to ${assignedTo.size}/6`);
  check('schedule: 5th and 6th workers got work',
    assignedTo.has('w5') && assignedTo.has('w6'));
}

// ---------------------------------------------------------------------------
// Static-source guards for client / config / docs changes.
// ---------------------------------------------------------------------------
function read(rel: string): string {
  return fs.readFileSync(path.join(__dirname, rel), 'utf8');
}

function testWorkerTaskIdAndDancing() {
  const worker = read('../public/worker.html');

  // Bug 3: full task id, no truncation.
  check('worker log: no truncated taskId.substring(0, 12)',
    !/taskId\.substring\(\s*0\s*,\s*12\s*\)/.test(worker) &&
    !/data\.taskId\.substring/.test(worker) &&
    !/message\.taskId\.substring/.test(worker));
  check('worker log: prints full ${...taskId}',
    worker.includes('${data.taskId}') && worker.includes('${message.taskId}'));

  // Bug 4: "dancing" UI mitigations.
  check('worker ui: stat values use tabular figures',
    /tabular-nums/.test(worker));
  check('worker ui: debounced computing indicator (setComputing)',
    worker.includes('function setComputing') && worker.includes('idleRevertTimer'));
}

function testDashboardKickDelegation() {
  const dash = read('../public/dashboard.html');

  // Bug 5: delegation, not per-row inline onclick.
  check('dashboard: no inline onclick="kickWorker"',
    !/onclick=["']kickWorker/.test(dash));
  check('dashboard: kick button carries data-kick-id',
    /data-kick-id=/.test(dash));
  check('dashboard: delegated click listener on the tbody',
    /workersTableBody\.addEventListener\(\s*['"]click['"]/.test(dash));
}

function testTsconfigAndReadme() {
  const tsconfig = read('../../tsconfig.json');
  // Bug 6: deprecated moduleResolution removed.
  check('tsconfig: no deprecated moduleResolution "node"',
    !/"moduleResolution"\s*:\s*"node"/.test(tsconfig));
  check('tsconfig: uses a supported moduleResolution',
    /"moduleResolution"\s*:\s*"(node16|nodenext|bundler)"/i.test(tsconfig));

  const readme = read('../../README.md');
  // Bug/Doc updates: npm + tsc troubleshooting present.
  check('readme: documents "command not found: npm"',
    /command not found: npm/i.test(readme));
  check('readme: documents "tsc: command not found"',
    /tsc: command not found/i.test(readme));
}

// ---------------------------------------------------------------------------
// FP16 transport: half-precision encode/decode round-trips within tolerance,
// produces exactly half the bytes, and the trainer defaults to fp16.
// ---------------------------------------------------------------------------
function testFp16Transport() {
  const sample = new Float32Array([0, 1, -1, 0.5, -0.0123, 3.14159, 1234.5, 1e-3, -7.7]);
  const b16 = TensorVM.float32ArrayToFloat16Base64(sample);
  const b32 = TensorVM.float32ArrayToBase64(sample);
  const back = TensorVM.float16Base64ToFloat32Array(b16);

  let maxRel = 0;
  for (let i = 0; i < sample.length; i++) {
    maxRel = Math.max(maxRel, Math.abs(back[i] - sample[i]) / (Math.abs(sample[i]) || 1));
  }
  check('fp16: round-trip within half-precision tolerance (<2e-3)', maxRel < 2e-3, `maxRel=${maxRel}`);
  check('fp16: payload is ~half of fp32', b16.length <= b32.length * 0.55);
  check('fp16: integer targets must NOT use fp16 (vocab indices > 2048 corrupt)',
    TensorVM.float32ToHalf(40000) !== 40000); // demonstrates why targets stay fp32

  // dispatcher honours dtype
  const viaDispatch = TensorVM.decodeBase64(TensorVM.encodeBase64(sample, 'fp16'), 'fp16');
  check('fp16: encodeBase64/decodeBase64 dispatch by dtype', viaDispatch.length === sample.length);

  const trainer = new Trainer(new Orchestrator());
  check('fp16: trainer defaults to fp16 precision', trainer.precision === 'fp16');
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DSL FLOP estimation (the unit of "work" for the remaining-time forecast).
// ---------------------------------------------------------------------------
function testDslFlops() {
  // matmul out[2,3] = a[2,4] @ b[4,3] -> 2*|out|*K = 2*6*4 = 48 FLOPs
  const dsl = '# forward\nmatmul o, a, b\nadd o2, o, o';
  const shapes: Record<string, number[]> = { o: [2, 3], a: [2, 4], b: [4, 3], o2: [2, 3] };
  const c = estimateDslFlops(dsl, shapes);
  check('dslflops: matmul = 2·|out|·K', c.matmulFlops === 48, `got ${c.matmulFlops}`);
  check('dslflops: counts element-wise op (|out|=6)', c.flops === 48 + 6, `got ${c.flops}`);
  check('dslflops: ignores comments in instruction count', c.instructionCount === 2);
  check('dslflops: matmul count', c.matmulCount === 1);
}

// ---------------------------------------------------------------------------
// wandb logger (offline-testable parts: mutation field names + history chunk).
// ---------------------------------------------------------------------------
function testWandb() {
  const w = new WandbLogger();
  const mutation = w.buildUpsertMutation();
  // Regression guard for the reported "Unknown field projectName" error.
  check('wandb: upsert mutation uses modelName (not projectName)',
    mutation.includes('modelName') && !mutation.includes('projectName'));
  check('wandb: upsert mutation uses entityName', mutation.includes('entityName'));

  // History chunk: correct offset + one JSON line per row, offset advances.
  const c1 = w.buildHistoryChunk([{ step: 1, loss: 2.5 }]);
  const file = c1.body.files['wandb-history.jsonl'];
  check('wandb: history chunk starts at offset 0', file.offset === 0);
  check('wandb: history chunk has one line per row', file.content.length === 1);
  check('wandb: history line is valid JSON with metrics',
    JSON.parse(file.content[0]).loss === 2.5);
  check('wandb: history row carries _step for the chart x-axis',
    JSON.parse(file.content[0])._step === 0);
  check('wandb: nextOffset advances by row count', c1.nextOffset === 1);

  // Inactive logger must be a no-op (never throws / never blocks training).
  check('wandb: inactive logger is not active', w.active === false);
}

// ---------------------------------------------------------------------------
// WASM SIMD matmul fast path — numerically equivalent to the JS reference for
// large 2D and batched-3D matmuls (the case that routes through WASM).
// ---------------------------------------------------------------------------
function testWasmMatmul() {
  const { Tensor, matmul } = require('../public/vm.js');
  const jsRef = (A: any, B: any, M: number, K: number, N: number) => {
    const C = new Float32Array(M * N);
    for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j]; C[i * N + j] = s; }
    return C;
  };
  const M = 128, K = 64, N = 96; // 786,432 FLOPs > threshold → WASM path
  const a = new Tensor([M, K]); for (let i = 0; i < a.data.length; i++) a.data[i] = Math.sin(i) * 0.5;
  const b = new Tensor([K, N]); for (let i = 0; i < b.data.length; i++) b.data[i] = Math.cos(i) * 0.5;
  const out = new Tensor([M, N]); matmul(out, a, b);
  let e = 0; { const r = jsRef(a.data, b.data, M, K, N); for (let i = 0; i < M * N; i++) e = Math.max(e, Math.abs(out.data[i] - r[i])); }
  check('wasm matmul: 2D matches JS reference', e < 1e-3, `maxErr=${e.toExponential(2)}`);

  const Bb = 3;
  const a3 = new Tensor([Bb, M, K]); for (let i = 0; i < a3.data.length; i++) a3.data[i] = Math.sin(i * 1.3) * 0.4;
  const b3 = new Tensor([Bb, K, N]); for (let i = 0; i < b3.data.length; i++) b3.data[i] = Math.cos(i * 0.7) * 0.4;
  const o3 = new Tensor([Bb, M, N]); matmul(o3, a3, b3);
  let e3 = 0;
  for (let bb = 0; bb < Bb; bb++) {
    const r = jsRef(a3.data.subarray(bb * M * K, (bb + 1) * M * K), b3.data.subarray(bb * K * N, (bb + 1) * K * N), M, K, N);
    for (let i = 0; i < M * N; i++) e3 = Math.max(e3, Math.abs(o3.data[bb * M * N + i] - r[i]));
  }
  check('wasm matmul: 3D batched matches JS reference', e3 < 1e-3, `maxErr=${e3.toExponential(2)}`);
}

console.log('=== Running Bug-Fix Regression Tests ===');
testWasmMatmul();
testKick();
testScheduling();
testFp16Transport();
testDslFlops();
testWandb();
testWorkerTaskIdAndDancing();
testDashboardKickDelegation();
testTsconfigAndReadme();

console.log(`\n${passed} passed, ${failed} failed.`);
// Orchestrator starts background intervals; exit explicitly so the test ends.
process.exit(failed === 0 ? 0 : 1);
