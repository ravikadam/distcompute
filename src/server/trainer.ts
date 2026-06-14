import { Orchestrator } from './orchestrator';
import { Compiler } from '../compiler/compiler';
import { Linear } from '../compiler/modules';
import { TensorVM } from '../public/vm';
import { WandbLogger } from './wandb';
import { estimateDslFlops } from './dslStats';
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
  private gptSampleNote = '(live sampling not yet available for GPT models)';

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
  
  constructor(orchestrator: Orchestrator) {
    this.orchestrator = orchestrator;
    this.initWeights();
  }

  // Dynamically update training parameters and reset model weights
  updateConfig(config: { lr?: number; batchSize?: number; hiddenDim?: number; contextLen?: number; corpus?: string; datasetFilePath?: string; targetSteps?: number; targetTokens?: number; precision?: 'fp16' | 'fp32' }) {
    this.stop();

    if (config.lr !== undefined) this.lr = config.lr;
    if (config.batchSize !== undefined) this.batchSize = config.batchSize;
    if (config.hiddenDim !== undefined) this.hiddenDim = config.hiddenDim;
    if (config.contextLen !== undefined) this.contextLen = config.contextLen;
    if (config.targetSteps !== undefined) this.targetSteps = config.targetSteps;
    if (config.targetTokens !== undefined) this.targetTokens = config.targetTokens;
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

  // Start the distributed training loop
  async start() {
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
    console.log(`Distributed Training Loop Started (${this.mode} model, ${Object.keys(this.weights).length} param tensors).`);

    this.tAdam = 0;
    // Reset ETA timing for this run.
    this.recentStepMs = [];
    this.lastStepTimestamp = 0;

    // Measure the per-task compute cost from the compiled DSL so we can report
    // work done / remaining in FLOPs.
    this.flopsPerTask = estimateDslFlops(dsl, shapes).flops;
    this.cumulativeTasks = 0;
    this.cumulativeExamples = 0;
    this.cumulativeFlops = 0;

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
        // Await all slices of the batch
        const results = await Promise.all(tasksPromises);

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

        // Average gradients and loss
        this.loss = totalLossVal / numSlices;
        for (const key of Object.keys(this.weights)) {
          for (let i = 0; i < grads[key].length; i++) {
            grads[key][i] /= numSlices;
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

        // Update weights using Adam optimizer
        this.applyAdamStep(grads);
        this.step++;

        // Track step duration for the ETA estimate (rolling window of 50).
        const nowTs = Date.now();
        if (this.lastStepTimestamp > 0) {
          this.recentStepMs.push(nowTs - this.lastStepTimestamp);
          if (this.recentStepMs.length > 50) this.recentStepMs.shift();
        }
        this.lastStepTimestamp = nowTs;

        // Accumulate work done (tasks, examples, FLOPs) for the forecast.
        this.lastNumSlices = numSlices;
        this.lastExamplesPerStep = currentBatchSize;
        this.cumulativeTasks += numSlices;
        this.cumulativeExamples += currentBatchSize;
        this.cumulativeFlops += numSlices * this.flopsPerTask;

        // Stream metrics to Weights & Biases (fail-soft; never blocks training).
        if (this.wandb && this.wandb.active) {
          const p = this.progress();
          this.wandb.log({
            step: this.step,
            loss: this.loss,
            grad_norm: this.gradNorm,
            epoch: this.epoch,
            workers: this.orchestrator.workers.size,
            examples_per_step: currentBatchSize,
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

        // Stop once the configured training target is reached. Token target
        // takes precedence; step target is the legacy fallback.
        const hitTokenTarget = this.targetTokens > 0 && this.cumulativeExamples >= this.targetTokens;
        const hitStepTarget = this.targetTokens <= 0 && this.targetSteps > 0 && this.step >= this.targetSteps;
        if (hitTokenTarget || hitStepTarget) {
          const reason = hitTokenTarget
            ? `${this.cumulativeExamples} tokens (target ${this.targetTokens})`
            : `${this.step} steps (target ${this.targetSteps})`;
          console.log(`Reached training target: ${reason}. Stopping.`);
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

  // Apply Adam optimization updates
  private applyAdamStep(grads: Record<string, Float32Array>) {
    this.tAdam++;
    const lrEff = this.lr * Math.sqrt(1 - Math.pow(this.adamBeta2, this.tAdam)) / (1 - Math.pow(this.adamBeta1, this.tAdam));

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

  // Generate text (sampling) based on current model weights
  samplePrediction(): string {
    // GPT live sampling needs a separate B=1 forward graph; not wired yet.
    if (this.mode === 'gpt') return this.gptSampleNote;
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
