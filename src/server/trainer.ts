import { Orchestrator } from './orchestrator';
import { Compiler } from '../compiler/compiler';
import { Linear } from '../compiler/modules';
import { TensorVM } from '../public/vm';
import { WandbLogger } from './wandb';
import { estimateDslFlops } from './dslStats';
import { CheckpointStore, CheckpointMeta, CheckpointBlob } from './checkpoints';
import * as fs from 'fs';
import * as path from 'path';

// Character indexing helper
function charToIdx(c: string): number {
  if (c === ' ') return 0;
  const code = c.toLowerCase().charCodeAt(0);
  if (code >= 97 && code <= 122) {
    return code - 97 + 1; // 1 to 26
  }
  return 0; // fallback to space
}

function idxToChar(idx: number): string {
  if (idx === 0) return ' ';
  return String.fromCharCode(idx - 1 + 97);
}

// A model the training loop can drive, decoupling the char-MLP from GPT so the
// same loop (planning, FP16 transport, gradient aggregation, Adam, forecast,
// wandb) serves both.
type SliceInputs = Record<string, { shape: number[]; data: string; dtype: string }>;
interface ModelRuntime {
  dsl: string;
  shapes: Record<string, number[]>;
  examplesPerSlice: number;                 // for throughput/work accounting
  numSlicesFor(idleWorkers: number): number;
  makeStepInputs(numSlices: number): SliceInputs[]; // non-weight inputs per task
  predict(): string;
}

export class Trainer {
  orchestrator: Orchestrator;
  isTraining = false;
  step = 0;
  epoch = 0;
  loss = 0.0;
  
  // Model hyperparameters
  vocabSize = 27; // 26 letters + space
  contextLen = 3; // predict next char based on 3 chars
  hiddenDim = 64;
  lr = 0.015;
  batchSize = 128;
  numWorkersPerBatch = 4; // split batch into 4 slices of size 32

  // Training target (in optimizer steps). 0 means "run indefinitely" — there
  // is then no defined amount of work remaining. When > 0, the dashboard can
  // show % complete and an ETA, and the loop stops once the target is reached.
  targetSteps = 0;

  // Preferred target: stop after this many training tokens/examples. 0 =
  // unlimited. Token-based so % complete and ETA stay stable (and ETA drops)
  // as workers are added, since examples/step scales with worker count.
  targetTokens = 0;

  // Learning-rate schedule. 'warmup_cosine' ramps the LR from 0 to `lr` over
  // `warmupSteps`, then cosine-anneals toward `minLrFrac × lr` across the run —
  // the standard transformer recipe that breaks the early loss plateau a flat
  // LR gets stuck on. 'constant' keeps the old fixed-LR behaviour.
  lrSchedule: 'warmup_cosine' | 'constant' = 'warmup_cosine';
  warmupSteps = 200;
  minLrFrac = 0.1;          // cosine floor = 10% of the base LR
  currentLr = 0.015;        // the LR actually applied this step (for graphing)

  // Optional Weights & Biases logger. Null unless a key has been configured.
  wandb: WandbLogger | null = null;

  // Wire precision for weights/gradients. 'fp16' halves the per-task payload
  // (compute stays FP32 on the worker). Integer targets are always sent as
  // fp32 regardless, since fp16 cannot represent class indices above 2048.
  precision: 'fp16' | 'fp32' = 'fp16';

  // Model selection. When dslFilePath points to a compiled .dsl (+ manifest),
  // we train that model (e.g. Tiny-GPT) byte-level; otherwise the built-in
  // char-MLP. Set from config.
  dslFilePath = '';
  mode: 'char' | 'gpt' = 'char';
  private gptBytes: Buffer | null = null;   // byte-level training corpus
  private gptSampleNote = '(start training a GPT model, then Generate a sample)';
  private gptCfg: { vocab: number; d: number; nLayer: number; nHead: number; dFF: number; context: number } | null = null;
  lastSample = '';                          // most recent generated text

  // Run identity + checkpointing. Each training run gets a unique id; the loop
  // saves weights + Adam state to disk every `checkpointEverySteps` so a run
  // survives a machine restart / network change and can be resumed later.
  runId = '';
  runCreatedAt = '';
  checkpointEverySteps = 20;
  checkpoints: CheckpointStore;

  // Rolling average of recent step durations (ms), used to estimate ETA.
  private recentStepMs: number[] = [];
  private lastStepTimestamp = 0;

  // Work accounting (FLOPs) derived from the compiled DSL, so "work done /
  // remaining" is measured in actual compute, not just step counts.
  gradNorm = 0;                      // global L2 norm of the last step's gradients
  flopsPerTask = 0;                  // forward+backward FLOPs for one task slice
  private cumulativeTasks = 0;       // total tasks dispatched across all steps
  private cumulativeExamples = 0;    // total training examples processed
  private cumulativeFlops = 0;       // total FLOPs processed
  private lastNumSlices = 0;         // slices in the most recent step
  private lastExamplesPerStep = 0;   // examples in the most recent step

  // Model parameters (Float32Array storage)
  weights: Record<string, Float32Array> = {};
  adamM: Record<string, Float32Array> = {};
  adamV: Record<string, Float32Array> = {};
  adamBeta1 = 0.9;
  adamBeta2 = 0.999;
  adamEps = 1e-8;
  tAdam = 0;

  // Server-side large dataset file seeking
  datasetFilePath = "";
  datasetFd: number | null = null;
  datasetFileSize = 0;
  corpus = "hello world distributed compute cluster training simple language model on browser workers and mobile phones to train deep neural networks";
  
  constructor(orchestrator: Orchestrator, checkpoints?: CheckpointStore) {
    this.orchestrator = orchestrator;
    this.checkpoints = checkpoints ?? new CheckpointStore();
    this.initWeights();
  }

  private generateRunId(): string {
    return `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  // Persist weights + Adam state + run metadata to disk (fail-soft).
  saveCheckpoint(): void {
    if (!this.runId) return;
    try {
      const enc = (rec: Record<string, Float32Array>) => {
        const o: Record<string, string> = {};
        for (const [k, v] of Object.entries(rec)) o[k] = TensorVM.float32ArrayToBase64(v);
        return o;
      };
      const meta: CheckpointMeta = {
        runId: this.runId,
        model: this.mode === 'gpt' ? path.basename(this.dslFilePath) : 'char-mlp',
        mode: this.mode,
        createdAt: this.runCreatedAt,
        updatedAt: new Date().toISOString(),
        step: this.step, epoch: this.epoch, tAdam: this.tAdam,
        loss: this.loss, gradNorm: this.gradNorm,
        cumulativeTasks: this.cumulativeTasks,
        cumulativeExamples: this.cumulativeExamples,
        cumulativeFlops: this.cumulativeFlops,
        config: {
          dslFilePath: this.dslFilePath, datasetFilePath: this.datasetFilePath,
          lr: this.lr, lrSchedule: this.lrSchedule, warmupSteps: this.warmupSteps,
          precision: this.precision, targetTokens: this.targetTokens, targetSteps: this.targetSteps,
          batchSize: this.batchSize, hiddenDim: this.hiddenDim, contextLen: this.contextLen
        }
      };
      const blob: CheckpointBlob = { weights: enc(this.weights), adamM: enc(this.adamM), adamV: enc(this.adamV) };
      this.checkpoints.save(this.runId, meta, blob);
    } catch (e) {
      console.error('[Trainer] checkpoint save failed:', e);
    }
  }

  // Dynamically update training parameters and reset model weights
  updateConfig(config: { lr?: number; batchSize?: number; hiddenDim?: number; contextLen?: number; corpus?: string; datasetFilePath?: string; targetSteps?: number; targetTokens?: number; precision?: 'fp16' | 'fp32'; lrSchedule?: 'warmup_cosine' | 'constant'; warmupSteps?: number }) {
    this.stop();

    if (config.lr !== undefined) this.lr = config.lr;
    if (config.batchSize !== undefined) this.batchSize = config.batchSize;
    if (config.hiddenDim !== undefined) this.hiddenDim = config.hiddenDim;
    if (config.contextLen !== undefined) this.contextLen = config.contextLen;
    if (config.targetSteps !== undefined) this.targetSteps = config.targetSteps;
    if (config.targetTokens !== undefined) this.targetTokens = config.targetTokens;
    if (config.lrSchedule !== undefined) this.lrSchedule = config.lrSchedule;
    if (config.warmupSteps !== undefined) this.warmupSteps = config.warmupSteps;
    if (config.precision !== undefined) this.precision = config.precision;
    
    // Reset server-side file tracking
    if (this.datasetFd !== null) {
      try {
        fs.closeSync(this.datasetFd);
      } catch (e) {}
      this.datasetFd = null;
      this.datasetFilePath = "";
      this.datasetFileSize = 0;
    }

    if (config.datasetFilePath && config.datasetFilePath.trim().length > 0) {
      try {
        const filePath = path.resolve(config.datasetFilePath);
        const stats = fs.statSync(filePath);
        this.datasetFileSize = stats.size;
        this.datasetFd = fs.openSync(filePath, 'r');
        this.datasetFilePath = filePath;
        console.log(`[Trainer] Successfully opened dataset on server: ${filePath} (${this.datasetFileSize.toLocaleString()} bytes)`);
      } catch (e: any) {
        throw new Error(`Failed to load server-side dataset: ${e.message}`);
      }
    } else if (config.corpus !== undefined && config.corpus.trim().length > 10) {
      this.corpus = config.corpus;
    }

    this.initWeights();
    this.step = 0;
    this.epoch = 0;
    this.loss = 0.0;
    this.tAdam = 0;
    this.broadcastStatsToDashboards();
    this.orchestrator.broadcastToDashboards();
  }

  // Initialize weights with He/Xavier-like initialization
  private initWeights() {
    // Clear any previous parameters (e.g. when switching back from a GPT model).
    this.weights = {};
    this.adamM = {};
    this.adamV = {};
    const w1Size = (this.contextLen * this.vocabSize) * this.hiddenDim;
    const b1Size = this.hiddenDim;
    const w2Size = this.hiddenDim * this.vocabSize;
    const b2Size = this.vocabSize;

    this.weights["fc_fc1_w"] = new Float32Array(w1Size);
    this.weights["fc_fc1_b"] = new Float32Array(b1Size);
    this.weights["fc_fc2_w"] = new Float32Array(w2Size);
    this.weights["fc_fc2_b"] = new Float32Array(b2Size);

    // Random initialization
    for (let i = 0; i < w1Size; i++) this.weights["fc_fc1_w"][i] = (Math.random() - 0.5) * 2 / Math.sqrt(this.contextLen * this.vocabSize);
    for (let i = 0; i < b1Size; i++) this.weights["fc_fc1_b"][i] = 0.0;
    for (let i = 0; i < w2Size; i++) this.weights["fc_fc2_w"][i] = (Math.random() - 0.5) * 2 / Math.sqrt(this.hiddenDim);
    for (let i = 0; i < b2Size; i++) this.weights["fc_fc2_b"][i] = 0.0;

    // Initialize Adam states
    for (const key of Object.keys(this.weights)) {
      this.adamM[key] = new Float32Array(this.weights[key].length);
      this.adamV[key] = new Float32Array(this.weights[key].length);
    }
  }

  // Generate dataset inputs & targets
  private getBatch(batchSize: number): { X: Float32Array; Y: Float32Array } {
    const X = new Float32Array(batchSize * this.contextLen * this.vocabSize);
    const Y = new Float32Array(batchSize);

    // Pre-allocate reading buffer for file descriptor seeking
    const fileBuffer = this.datasetFd !== null ? Buffer.alloc(this.contextLen + 4) : null;

    for (let b = 0; b < batchSize; b++) {
      let context = "";
      let targetChar = " ";

      if (this.datasetFd !== null && this.datasetFileSize > this.contextLen + 5) {
        // Read a slice directly from the local file without loading it all to memory
        const offset = Math.floor(Math.random() * (this.datasetFileSize - this.contextLen - 5));
        fs.readSync(this.datasetFd, fileBuffer!, 0, this.contextLen + 4, offset);
        const text = fileBuffer!.toString('utf8');
        context = text.substring(0, this.contextLen);
        targetChar = text[this.contextLen] || ' ';
      } else {
        // Fallback to memory corpus string
        const startIdx = Math.floor(Math.random() * (this.corpus.length - this.contextLen - 1));
        context = this.corpus.substring(startIdx, startIdx + this.contextLen);
        targetChar = this.corpus[startIdx + this.contextLen];
      }

      Y[b] = charToIdx(targetChar);

      // Encode X as concatenated one-hot vectors
      for (let c = 0; c < this.contextLen; c++) {
        const charIdx = charToIdx(context[c]);
        const flatIdx = b * (this.contextLen * this.vocabSize) + c * this.vocabSize + charIdx;
        X[flatIdx] = 1.0;
      }
    }

    return { X, Y };
  }

  // Decide how to split a training step across the available workers.
  //
  // Each task runs a fixed 32-row slice (the per-task batch dimension the
  // model graph is compiled for). We create one slice per idle worker so
  // every connected node gets work, scaling from a handful of devices to
  // thousands. We never drop below the configured batch worth of slices, so
  // small clusters still process a full batch (queued across fewer workers).
  planSlices(idleWorkerCount: number): { numSlices: number; sliceSize: number } {
    const sliceSize = 32;
    const configuredSlices = Math.max(1, Math.round(this.batchSize / sliceSize));
    const numSlices = Math.max(idleWorkerCount, configuredSlices);
    return { numSlices, sliceSize };
  }

  // Start the distributed training loop. Pass a checkpoint to resume that run.
  async start(resume?: { meta: CheckpointMeta; blob: CheckpointBlob }) {
    if (this.isTraining) return;
    this.isTraining = true;

    // Pick the model: a compiled .dsl file (e.g. Tiny-GPT) or the built-in char-MLP.
    this.mode = this.dslFilePath && this.dslFilePath.trim().length > 0 ? 'gpt' : 'char';
    let model: ModelRuntime;
    try {
      model = this.mode === 'gpt' ? this.buildGptModel() : this.buildCharModel();
    } catch (e: any) {
      console.error('Failed to build model:', e?.message || e);
      this.orchestrator.eventLog.record('error', `Model build failed: ${e?.message || e}`, { level: 'error' });
      this.isTraining = false;
      return;
    }
    const { dsl, shapes } = model;
    this.flopsPerTask = estimateDslFlops(dsl, shapes).flops;
    this.recentStepMs = [];
    this.lastStepTimestamp = 0;

    if (resume) {
      // Restore run identity, training state, and parameters from the checkpoint.
      this.runId = resume.meta.runId;
      this.runCreatedAt = resume.meta.createdAt;
      this.step = resume.meta.step;
      this.epoch = resume.meta.epoch;
      this.tAdam = resume.meta.tAdam;
      this.cumulativeTasks = resume.meta.cumulativeTasks || 0;
      this.cumulativeExamples = resume.meta.cumulativeExamples || 0;
      this.cumulativeFlops = resume.meta.cumulativeFlops || 0;
      const restore = (target: Record<string, Float32Array>, src: Record<string, string>) => {
        for (const [k, b64] of Object.entries(src)) {
          if (target[k]) target[k].set(TensorVM.base64ToFloat32Array(b64));
        }
      };
      restore(this.weights, resume.blob.weights);
      restore(this.adamM, resume.blob.adamM);
      restore(this.adamV, resume.blob.adamV);
      console.log(`Resumed run ${this.runId} at step ${this.step} (${this.mode} model).`);
    } else {
      // Fresh run: new id, zeroed counters.
      this.runId = this.generateRunId();
      this.runCreatedAt = new Date().toISOString();
      this.step = 0; this.epoch = 0; this.tAdam = 0;
      this.cumulativeTasks = 0; this.cumulativeExamples = 0; this.cumulativeFlops = 0;
      console.log(`Started run ${this.runId} (${this.mode} model, ${Object.keys(this.weights).length} param tensors).`);
    }
    // Tag the active run on the orchestrator so every (re)connecting worker's
    // tasks carry it, and the dashboard can show which run is live.
    this.orchestrator.currentRunId = this.runId;

    while (this.isTraining) {
      // Wait until we have enough active workers
      const activeWorkers = Array.from(this.orchestrator.workers.values()).filter(w => w.status === 'idle');
      if (activeWorkers.length === 0) {
        // console.log("Waiting for idle workers...");
        await new Promise(r => setTimeout(r, 1000));
        continue;
      }

      // One task per idle worker (the model decides the count + its own floor).
      const numSlices = model.numSlicesFor(activeWorkers.length);
      const currentBatchSize = numSlices * model.examplesPerSlice;

      // Non-weight inputs per task: X/targets for char; tok/pos/mask/targets for GPT.
      const sliceInputs = model.makeStepInputs(numSlices);

      // Encode the (identical) global weights ONCE per step and reuse the
      // strings across every slice. Previously this ran inside the per-slice
      // loop, so the single-threaded server base64-encoded the whole model
      // numSlices times per step — O(numSlices × modelSize) of redundant CPU
      // that grew with worker count and was a big reason adding workers slowed
      // each step down.
      const weightInputs: Record<string, { shape: number[]; data: string; dtype: string }> = {};
      for (const [key, val] of Object.entries(this.weights)) {
        weightInputs[key] = {
          shape: shapes[key],
          data: TensorVM.encodeBase64(val, this.precision),
          dtype: this.precision
        };
      }

      const tasksPromises: Promise<any>[] = [];
      for (let s = 0; s < numSlices; s++) {
        // Per-slice inputs (X/targets or tok/pos/mask/targets) + the shared,
        // already-encoded weights. The weight entries are read-only and
        // serialized per worker, so sharing the objects across tasks is safe.
        const taskInputs = { ...sliceInputs[s], ...weightInputs };
        tasksPromises.push(this.orchestrator.submitTask(dsl, shapes, taskInputs, this.precision));
      }

      try {
        // Collect completed slices up to a per-step deadline so a single
        // straggler doesn't pace the whole step. Synchronous SGD simply uses a
        // smaller batch that step (we average over whatever returned). The
        // deadline adapts to recent task round-trip time.
        const results: any[] = [];
        let settledCount = 0;
        for (const p of tasksPromises) {
          p.then(r => results.push(r)).catch(() => { /* rejected slice ignored */ }).finally(() => { settledCount++; });
        }
        const recentRt = this.orchestrator.getTaskTiming().avgRoundTripMs || 4000;
        const deadlineMs = Math.max(4000, recentRt * 1.5);
        const waitStart = Date.now();
        // Wait until every slice settles, or the deadline elapses with ≥1 result.
        while (settledCount < numSlices && (Date.now() - waitStart < deadlineMs || results.length === 0)) {
          await new Promise(r => setTimeout(r, 50));
        }
        const effectiveSlices = results.length;
        if (effectiveSlices === 0) {
          // Nothing returned at all (all workers slow/gone) — back off and retry.
          await new Promise(r => setTimeout(r, 500));
          continue;
        }
        const effExamples = effectiveSlices * model.examplesPerSlice;
        if (effectiveSlices < numSlices) {
          console.log(`Step ${this.step + 1}: ${effectiveSlices}/${numSlices} slices in (stragglers dropped this step)`);
        }

        // Aggregate gradients
        const grads: Record<string, Float32Array> = {};
        for (const key of Object.keys(this.weights)) {
          grads[key] = new Float32Array(this.weights[key].length);
        }

        let totalLossVal = 0;

        for (const res of results) {
          // In our VM cross_entropy outputs the loss in 't0' or whichever loss node was computed
          // Find the name of the loss node in outputs (it has shape [1])
          let lossKey = "";
          for (const k of Object.keys(res)) {
            if (res[k].shape.length === 1 && res[k].shape[0] === 1 && k.startsWith("t")) {
              lossKey = k;
              break;
            }
          }

          if (lossKey) {
            const lossData = TensorVM.decodeBase64(res[lossKey].data, (res[lossKey] as any).dtype || this.precision);
            totalLossVal += lossData[0];
          }

          // Accumulate parameter gradients: e.g. g_fc_fc1_w
          for (const key of Object.keys(this.weights)) {
            const gradKey = `g_${key}`;
            if (res[gradKey]) {
              const gradData = TensorVM.decodeBase64(res[gradKey].data, (res[gradKey] as any).dtype || this.precision);
              for (let i = 0; i < grads[key].length; i++) {
                grads[key][i] += gradData[i];
              }
            }
          }
        }

        // Average gradients and loss over the slices that actually returned.
        this.loss = totalLossVal / effectiveSlices;
        for (const key of Object.keys(this.weights)) {
          for (let i = 0; i < grads[key].length; i++) {
            grads[key][i] /= effectiveSlices;
          }
        }

        // Global gradient L2 norm — a training-health signal (spikes = instability,
        // collapse toward 0 = vanishing/stalled). Computed on the averaged grads.
        let gradSq = 0;
        for (const key of Object.keys(grads)) {
          const g = grads[key];
          for (let i = 0; i < g.length; i++) gradSq += g[i] * g[i];
        }
        this.gradNorm = Math.sqrt(gradSq);

        // Update weights using Adam optimizer (step is 1-based here so the LR
        // schedule sees the correct step number).
        this.step++;
        this.applyAdamStep(grads);

        // Track step duration for the ETA estimate (rolling window of 50).
        const nowTs = Date.now();
        if (this.lastStepTimestamp > 0) {
          this.recentStepMs.push(nowTs - this.lastStepTimestamp);
          if (this.recentStepMs.length > 50) this.recentStepMs.shift();
        }
        this.lastStepTimestamp = nowTs;

        // Accumulate work done (tasks, examples, FLOPs) for the forecast — based
        // on slices that actually completed this step, not the number dispatched.
        this.lastNumSlices = effectiveSlices;
        this.lastExamplesPerStep = effExamples;
        this.cumulativeTasks += effectiveSlices;
        this.cumulativeExamples += effExamples;
        this.cumulativeFlops += effectiveSlices * this.flopsPerTask;

        // Stream metrics to Weights & Biases (fail-soft; never blocks training).
        if (this.wandb && this.wandb.active) {
          const p = this.progress();
          this.wandb.log({
            step: this.step,
            loss: this.loss,
            grad_norm: this.gradNorm,
            lr: this.currentLr,
            epoch: this.epoch,
            workers: this.orchestrator.workers.size,
            examples_per_step: effExamples,
            ...(p.percent >= 0
              ? { progress_pct: p.percent, eta_seconds: p.etaSeconds, eta_minutes: p.etaSeconds / 60 }
              : {})
          });
        }

        if (this.step % 10 === 0) {
          this.epoch = Math.floor(this.step / 50);
          console.log(`Step ${this.step} | Loss: ${this.loss.toFixed(4)} | Prediction: ${this.samplePrediction()}`);
          this.broadcastStatsToDashboards();
        }

        // Checkpoint weights + Adam state periodically so the run survives a
        // crash / restart and can be resumed.
        if (this.step % this.checkpointEverySteps === 0) this.saveCheckpoint();

        // Stop once the configured training target is reached. Token target
        // takes precedence; step target is the legacy fallback.
        const hitTokenTarget = this.targetTokens > 0 && this.cumulativeExamples >= this.targetTokens;
        const hitStepTarget = this.targetTokens <= 0 && this.targetSteps > 0 && this.step >= this.targetSteps;
        if (hitTokenTarget || hitStepTarget) {
          const reason = hitTokenTarget
            ? `${this.cumulativeExamples} tokens (target ${this.targetTokens})`
            : `${this.step} steps (target ${this.targetSteps})`;
          console.log(`Reached training target: ${reason}. Stopping.`);
          this.saveCheckpoint(); // final checkpoint
          this.broadcastStatsToDashboards();
          if (this.wandb && this.wandb.active) await this.wandb.finish();
          this.isTraining = false;
          break;
        }

        // Tiny delay
        await new Promise(r => setTimeout(r, 50));
      } catch (err) {
        console.error("Failed to run batch step, rescheduling batch...", err);
        await new Promise(r => setTimeout(r, 1000));
      }
    }
  }

  stop() {
    this.isTraining = false;
  }

  // ---- Model runtimes -------------------------------------------------------

  // Built-in character-level MLP (the original toy model).
  private buildCharModel(): ModelRuntime {
    if (!this.weights['fc_fc1_w']) this.initWeights(); // ensure char params
    const compiler = new Compiler();
    const X_node = compiler.createNode('X', [32, this.contextLen * this.vocabSize]);
    const targets_node = compiler.createNode('targets', [32]);
    const fc = new Linear(compiler, 'fc_fc1', this.contextLen * this.vocabSize, this.hiddenDim);
    const fc2 = new Linear(compiler, 'fc_fc2', this.hiddenDim, this.vocabSize);
    const h = compiler.gelu(fc.forward(compiler, X_node));
    const logits = fc2.forward(compiler, h);
    const { loss } = compiler.cross_entropy(logits, targets_node);
    const { dsl, shapes } = compiler.compile(loss, targets_node);
    const ctxVocab = this.contextLen * this.vocabSize;

    return {
      dsl, shapes,
      examplesPerSlice: 32,
      numSlicesFor: (idle) => this.planSlices(idle).numSlices,
      makeStepInputs: (numSlices) => {
        const sliceSize = 32;
        const batch = this.getBatch(numSlices * sliceSize);
        const out: SliceInputs[] = [];
        for (let s = 0; s < numSlices; s++) {
          const xSlice = batch.X.subarray(s * sliceSize * ctxVocab, (s + 1) * sliceSize * ctxVocab);
          const ySlice = batch.Y.subarray(s * sliceSize, (s + 1) * sliceSize);
          out.push({
            X: { shape: [sliceSize, ctxVocab], data: TensorVM.encodeBase64(new Float32Array(xSlice), this.precision), dtype: this.precision },
            targets: { shape: [sliceSize], data: TensorVM.float32ArrayToBase64(new Float32Array(ySlice)), dtype: 'fp32' }
          });
        }
        return out;
      },
      predict: () => this.samplePrediction()
    };
  }

  // GPT model loaded from a compiled .dsl file + manifest, trained byte-level.
  private buildGptModel(): ModelRuntime {
    const dslPath = path.resolve(this.dslFilePath);
    const manifestPath = dslPath.replace(/\.dsl$/, '.manifest.json');
    if (!fs.existsSync(dslPath)) throw new Error(`DSL file not found: ${dslPath}`);
    if (!fs.existsSync(manifestPath)) throw new Error(`Manifest not found beside DSL: ${manifestPath}`);

    const dsl = fs.readFileSync(dslPath, 'utf8');
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const shapes: Record<string, number[]> = manifest.shapes;
    const B: number = manifest.batch, S: number = manifest.seq, N = B * S;
    // Remember the architecture for byte-level live sampling.
    const mc = manifest.config;
    this.gptCfg = { vocab: mc.vocab, d: mc.dModel, nLayer: mc.nLayer, nHead: mc.nHead, dFF: mc.dFF, context: mc.context };

    // Byte-level training corpus (server dataset path, else the in-memory corpus).
    if (this.datasetFilePath && this.datasetFilePath.trim()) {
      this.gptBytes = fs.readFileSync(path.resolve(this.datasetFilePath));
    } else {
      this.gptBytes = Buffer.from(this.corpus, 'utf8');
    }
    if (this.gptBytes.length < S + 2) throw new Error('Dataset is smaller than one sequence; choose a larger corpus');

    this.initGptWeights(manifest.parameters);

    // Causal additive mask is constant for the whole run — encode once.
    const maskArr = new Float32Array(S * S);
    for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) maskArr[i * S + j] = j <= i ? 0 : -1e9;
    const maskInput = { shape: [1, S, S], data: TensorVM.float32ArrayToBase64(maskArr), dtype: 'fp32' };
    const bytes = this.gptBytes;

    return {
      dsl, shapes,
      examplesPerSlice: N, // tokens processed per task
      numSlicesFor: (idle) => Math.max(1, idle),
      makeStepInputs: (numSlices) => {
        const out: SliceInputs[] = [];
        for (let t = 0; t < numSlices; t++) {
          const tok = new Float32Array(N), pos = new Float32Array(N), tgt = new Float32Array(N);
          for (let b = 0; b < B; b++) {
            const start = Math.floor(Math.random() * (bytes.length - S - 1));
            for (let i = 0; i < S; i++) {
              tok[b * S + i] = bytes[start + i];
              tgt[b * S + i] = bytes[start + i + 1]; // next-byte target
              pos[b * S + i] = i;
            }
          }
          // Token/position/target indices stay fp32 (integer-valued).
          out.push({
            tok: { shape: [N], data: TensorVM.float32ArrayToBase64(tok), dtype: 'fp32' },
            pos: { shape: [N], data: TensorVM.float32ArrayToBase64(pos), dtype: 'fp32' },
            targets: { shape: [N], data: TensorVM.float32ArrayToBase64(tgt), dtype: 'fp32' },
            mask: maskInput
          });
        }
        return out;
      },
      predict: () => this.gptSampleNote
    };
  }

  // Initialise GPT parameters from the manifest (GPT-1 style init).
  private initGptWeights(params: { name: string; shape: number[] }[]) {
    this.weights = {};
    this.adamM = {};
    this.adamV = {};
    for (const p of params) {
      const size = p.shape.reduce((a, b) => a * b, 1);
      const arr = new Float32Array(size);
      if (p.name.endsWith('_g')) arr.fill(1);           // LayerNorm gain
      else if (p.name.endsWith('_b')) arr.fill(0);      // biases / LN beta
      else for (let i = 0; i < size; i++) arr[i] = this.randn() * 0.02; // weights + embeddings
      this.weights[p.name] = arr;
      this.adamM[p.name] = new Float32Array(size);
      this.adamV[p.name] = new Float32Array(size);
    }
  }

  private randn(): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /**
   * Training progress, work accounting, throughput, per-task timing and an ETA.
   *
   * Work is measured in FLOPs derived from the compiled DSL (`flopsPerTask`), so
   * "how much work is left" is real compute, not just a step count. The forecast
   * combines measured throughput with the remaining work. With targetSteps === 0
   * the run is open-ended, so target-relative fields are reported as -1.
   */
  progress() {
    const avgStepMs = this.recentStepMs.length
      ? this.recentStepMs.reduce((a, b) => a + b, 0) / this.recentStepMs.length
      : 0;
    const stepSec = avgStepMs / 1000;

    // Measured throughput from the most recent step.
    const examplesPerSec = stepSec > 0 ? this.lastExamplesPerStep / stepSec : 0;
    const tasksPerSec = stepSec > 0 ? this.lastNumSlices / stepSec : 0;
    const flopsPerSec = stepSec > 0 ? (this.lastNumSlices * this.flopsPerTask) / stepSec : 0;

    const timing = this.orchestrator.getTaskTiming();

    const base = {
      // counters
      step: this.step,
      gradNorm: this.gradNorm,
      currentLr: this.currentLr,
      tasksCompleted: timing.totalCompleted,
      examplesProcessed: this.cumulativeExamples,
      flopsProcessed: this.cumulativeFlops,
      flopsPerTask: this.flopsPerTask,
      // throughput
      examplesPerSec,
      tasksPerSec,
      flopsPerSec,
      avgStepMs,
      avgTaskComputeMs: timing.avgComputeMs,
      avgTaskRoundTripMs: timing.avgRoundTripMs,
    };

    // Resolve the target into EXAMPLES (tokens). Token target is preferred; a
    // legacy step target is converted using the current examples/step.
    const examplesPerStepNow = this.lastExamplesPerStep || this.batchSize;
    let targetExamples = 0;
    if (this.targetTokens > 0) targetExamples = this.targetTokens;
    else if (this.targetSteps > 0) targetExamples = this.targetSteps * examplesPerStepNow;

    if (targetExamples <= 0) {
      // Open-ended run: no fixed amount of work remaining.
      return {
        ...base,
        targetSteps: this.targetSteps, targetTokens: this.targetTokens, targetExamples: 0,
        remaining: -1, percent: -1, etaSeconds: -1,
        flopsTotal: -1, flopsRemaining: -1, examplesRemaining: -1, workPercent: -1
      };
    }

    // Everything is measured in examples/tokens, so progress and ETA are stable
    // as worker count changes — and ETA *drops* when throughput rises.
    const examplesRemaining = Math.max(0, targetExamples - this.cumulativeExamples);
    const percent = Math.min(100, (this.cumulativeExamples / targetExamples) * 100);
    const etaSeconds = examplesPerSec > 0 ? examplesRemaining / examplesPerSec : -1;

    const flopsPerExample = examplesPerStepNow > 0
      ? (this.lastNumSlices * this.flopsPerTask) / examplesPerStepNow
      : 0;
    const flopsRemaining = examplesRemaining * flopsPerExample;
    const flopsTotal = this.cumulativeFlops + flopsRemaining;
    const workPercent = flopsTotal > 0 ? (this.cumulativeFlops / flopsTotal) * 100 : percent;

    return {
      ...base,
      targetSteps: this.targetSteps, targetTokens: this.targetTokens, targetExamples,
      remaining: examplesRemaining, percent, etaSeconds,
      flopsTotal, flopsRemaining, examplesRemaining, workPercent
    };
  }

  // Learning rate for the current step under the active schedule.
  effectiveLr(): number {
    if (this.lrSchedule !== 'warmup_cosine') return this.lr;
    const step = this.step; // step already incremented for this update
    if (step <= this.warmupSteps) {
      return this.lr * (step / Math.max(1, this.warmupSteps)); // linear warmup from ~0
    }
    // Cosine anneal over the run's step horizon. Derive the horizon from the
    // target (steps directly, or tokens/examples-per-step); if open-ended, hold
    // at the base LR after warmup (no horizon to decay across).
    let horizon = 0;
    const examplesPerStepNow = this.lastExamplesPerStep || this.batchSize;
    if (this.targetSteps > 0 && this.targetTokens <= 0) horizon = this.targetSteps;
    else if (this.targetTokens > 0 && examplesPerStepNow > 0) horizon = this.targetTokens / examplesPerStepNow;
    if (horizon <= this.warmupSteps) return this.lr; // unknown/short horizon → constant
    const prog = Math.min(1, Math.max(0, (step - this.warmupSteps) / (horizon - this.warmupSteps)));
    const minLr = this.lr * this.minLrFrac;
    return minLr + 0.5 * (this.lr - minLr) * (1 + Math.cos(Math.PI * prog));
  }

  // Apply Adam optimization updates
  private applyAdamStep(grads: Record<string, Float32Array>) {
    this.tAdam++;
    this.currentLr = this.effectiveLr();
    const lrEff = this.currentLr * Math.sqrt(1 - Math.pow(this.adamBeta2, this.tAdam)) / (1 - Math.pow(this.adamBeta1, this.tAdam));

    for (const key of Object.keys(this.weights)) {
      const w = this.weights[key];
      const g = grads[key];
      const m = this.adamM[key];
      const v = this.adamV[key];

      for (let i = 0; i < w.length; i++) {
        m[i] = this.adamBeta1 * m[i] + (1 - this.adamBeta1) * g[i];
        v[i] = this.adamBeta2 * v[i] + (1 - this.adamBeta2) * g[i] * g[i];
        w[i] -= lrEff * m[i] / (Math.sqrt(v[i]) + this.adamEps);
      }
    }
  }

  // ---- GPT byte-level live sampling -----------------------------------------

  // Tight JS forward pass for a single sequence; returns the next-token logits
  // for the LAST position. Mirrors the compiled GPT exactly (pre-norm blocks,
  // causal multi-head attention, GELU MLP, weight-tied head, LN eps 1e-5).
  private gptForwardLogits(ids: number[]): Float32Array {
    const cfg = this.gptCfg!;
    const d = cfg.d, H = cfg.nHead, dh = d / H, dFF = cfg.dFF, V = cfg.vocab, L = cfg.nLayer;
    const S = ids.length, eps = 1e-5, W = this.weights, scale = 1 / Math.sqrt(dh);

    const x = new Float32Array(S * d);
    for (let t = 0; t < S; t++)
      for (let k = 0; k < d; k++) x[t * d + k] = W['wte'][ids[t] * d + k] + W['wpe'][t * d + k];

    const layernorm = (inp: Float32Array, g: Float32Array, b: Float32Array) => {
      const out = new Float32Array(S * d);
      for (let t = 0; t < S; t++) {
        let m = 0; for (let k = 0; k < d; k++) m += inp[t * d + k]; m /= d;
        let v = 0; for (let k = 0; k < d; k++) { const c = inp[t * d + k] - m; v += c * c; } v /= d;
        const r = 1 / Math.sqrt(v + eps);
        for (let k = 0; k < d; k++) out[t * d + k] = (inp[t * d + k] - m) * r * g[k] + b[k];
      }
      return out;
    };
    const linear = (inp: Float32Array, w: Float32Array, b: Float32Array, inD: number, outD: number) => {
      const out = new Float32Array(S * outD);
      for (let t = 0; t < S; t++)
        for (let o = 0; o < outD; o++) {
          let s = b[o]; const base = t * inD;
          for (let k = 0; k < inD; k++) s += inp[base + k] * w[k * outD + o];
          out[t * outD + o] = s;
        }
      return out;
    };

    for (let l = 0; l < L; l++) {
      const p = `h${l}_`;
      const h = layernorm(x, W[p + 'ln1_g'], W[p + 'ln1_b']);
      const Q = linear(h, W[p + 'attn_wq_w'], W[p + 'attn_wq_b'], d, d);
      const K = linear(h, W[p + 'attn_wk_w'], W[p + 'attn_wk_b'], d, d);
      const Vv = linear(h, W[p + 'attn_wv_w'], W[p + 'attn_wv_b'], d, d);
      const ctxv = new Float32Array(S * d);
      for (let hd = 0; hd < H; hd++) {
        const off = hd * dh;
        for (let i = 0; i < S; i++) {
          const sc = new Float32Array(i + 1); let mx = -Infinity;
          for (let j = 0; j <= i; j++) {
            let dot = 0; for (let k = 0; k < dh; k++) dot += Q[i * d + off + k] * K[j * d + off + k];
            sc[j] = dot * scale; if (sc[j] > mx) mx = sc[j];
          }
          let sum = 0; for (let j = 0; j <= i; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
          for (let k = 0; k < dh; k++) {
            let acc = 0; for (let j = 0; j <= i; j++) acc += (sc[j] / sum) * Vv[j * d + off + k];
            ctxv[i * d + off + k] = acc;
          }
        }
      }
      const proj = linear(ctxv, W[p + 'attn_wo_w'], W[p + 'attn_wo_b'], d, d);
      for (let n = 0; n < S * d; n++) x[n] += proj[n];
      const h2 = layernorm(x, W[p + 'ln2_g'], W[p + 'ln2_b']);
      const ff = linear(h2, W[p + 'mlp_fc_w'], W[p + 'mlp_fc_b'], d, dFF);
      for (let n = 0; n < ff.length; n++) { const u = ff[n]; ff[n] = 0.5 * u * (1 + Math.tanh(0.7978845608 * (u + 0.044715 * u * u * u))); }
      const ff2 = linear(ff, W[p + 'mlp_proj_w'], W[p + 'mlp_proj_b'], dFF, d);
      for (let n = 0; n < S * d; n++) x[n] += ff2[n];
    }

    const xf = layernorm(x, W['ln_f_g'], W['ln_f_b']);
    const last = (S - 1) * d, wte = W['wte'], logits = new Float32Array(V);
    for (let v = 0; v < V; v++) { let s = 0; const base = v * d; for (let k = 0; k < d; k++) s += xf[last + k] * wte[base + k]; logits[v] = s; }
    return logits;
  }

  // Autoregressively generate text from a prompt (byte-level).
  generateSample(prompt: string, nTokens = 48, temperature = 0.8): string {
    if (this.mode !== 'gpt' || !this.gptCfg || !this.weights['wte']) return this.samplePrediction();
    const ctxLen = this.gptCfg.context;
    let ids = Array.from(Buffer.from(prompt && prompt.length ? prompt : ' ', 'utf8'));
    if (ids.length === 0) ids = [32];
    const n = Math.min(Math.max(1, nTokens), 256);
    for (let i = 0; i < n; i++) {
      const logits = this.gptForwardLogits(ids.slice(-ctxLen));
      let next: number;
      if (temperature <= 0) {
        let bi = 0, bv = -Infinity;
        for (let v = 0; v < logits.length; v++) if (logits[v] > bv) { bv = logits[v]; bi = v; }
        next = bi;
      } else {
        let mx = -Infinity; for (let v = 0; v < logits.length; v++) if (logits[v] > mx) mx = logits[v];
        let sum = 0; const probs = new Float32Array(logits.length);
        for (let v = 0; v < logits.length; v++) { probs[v] = Math.exp((logits[v] - mx) / temperature); sum += probs[v]; }
        let r = Math.random() * sum, acc = 0; next = logits.length - 1;
        for (let v = 0; v < logits.length; v++) { acc += probs[v]; if (r <= acc) { next = v; break; } }
      }
      ids.push(next);
    }
    this.lastSample = Buffer.from(ids).toString('utf8');
    return this.lastSample;
  }

  // Generate text (sampling) based on current model weights
  samplePrediction(): string {
    // GPT: show the most recent on-demand sample (full generation is via generateSample()).
    if (this.mode === 'gpt') return this.lastSample || this.gptSampleNote;
    let context = "hel"; // starting prompt
    let out = context;

    // Helper matrices
    const w1 = this.weights["fc_fc1_w"];
    const b1 = this.weights["fc_fc1_b"];
    const w2 = this.weights["fc_fc2_w"];
    const b2 = this.weights["fc_fc2_b"];

    const w1InDim = this.contextLen * this.vocabSize;
    const w1OutDim = this.hiddenDim;
    const w2OutDim = this.vocabSize;

    for (let charGen = 0; charGen < 20; charGen++) {
      // 1. One-hot encode context
      const x = new Float32Array(w1InDim);
      for (let c = 0; c < this.contextLen; c++) {
        const charIdx = charToIdx(context[c]);
        x[c * this.vocabSize + charIdx] = 1.0;
      }

      // 2. FC1: x @ w1 + b1
      const h = new Float32Array(w1OutDim);
      for (let j = 0; j < w1OutDim; j++) {
        let sum = b1[j];
        for (let i = 0; i < w1InDim; i++) {
          sum += x[i] * w1[i * w1OutDim + j];
        }
        // GELU activation
        const cVal = 0.7978845608;
        h[j] = 0.5 * sum * (1 + Math.tanh(cVal * (sum + 0.044715 * sum * sum * sum)));
      }

      // 3. FC2: h @ w2 + b2
      const logits = new Float32Array(w2OutDim);
      let maxLogit = -Infinity;
      for (let j = 0; j < w2OutDim; j++) {
        let sum = b2[j];
        for (let i = 0; i < w1OutDim; i++) {
          sum += h[i] * w2[i * w2OutDim + j];
        }
        logits[j] = sum;
        if (sum > maxLogit) maxLogit = sum;
      }

      // 4. Softmax & sample character
      let sumExp = 0;
      const probs = new Float32Array(w2OutDim);
      for (let j = 0; j < w2OutDim; j++) {
        probs[j] = Math.exp(logits[j] - maxLogit);
        sumExp += probs[j];
      }
      for (let j = 0; j < w2OutDim; j++) {
        probs[j] /= sumExp;
      }

      // Greedy or argmax sampling
      let bestIdx = 0;
      let maxP = -1;
      for (let j = 0; j < w2OutDim; j++) {
        if (probs[j] > maxP) {
          maxP = probs[j];
          bestIdx = j;
        }
      }

      const nextChar = idxToChar(bestIdx);
      out += nextChar;
      context = context.substring(1) + nextChar;
    }

    return out;
  }

  // Send updates to dashboard clients
  broadcastStatsToDashboards() {
    for (const ws of this.orchestrator.dashboardSockets) {
      if (ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify({
          type: 'training_update',
          step: this.step,
          epoch: this.epoch,
          loss: this.loss,
          prediction: this.samplePrediction()
        }));
      }
    }
  }

  // Export current model weights as JSON
  exportWeights(): string {
    const serializable: Record<string, number[]> = {};
    for (const [key, val] of Object.entries(this.weights)) {
      serializable[key] = Array.from(val);
    }
    return JSON.stringify(serializable, null, 2);
  }
}
