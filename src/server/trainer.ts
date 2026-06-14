import { Orchestrator } from './orchestrator';
import { Compiler } from '../compiler/compiler';
import { Linear } from '../compiler/modules';
import { TensorVM } from '../public/vm';
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
  updateConfig(config: { lr?: number; batchSize?: number; hiddenDim?: number; contextLen?: number; corpus?: string; datasetFilePath?: string }) {
    this.stop();

    if (config.lr !== undefined) this.lr = config.lr;
    if (config.batchSize !== undefined) this.batchSize = config.batchSize;
    if (config.hiddenDim !== undefined) this.hiddenDim = config.hiddenDim;
    if (config.contextLen !== undefined) this.contextLen = config.contextLen;
    
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

  // Start the distributed training loop
  async start() {
    if (this.isTraining) return;
    this.isTraining = true;
    console.log("Distributed Training Loop Started.");

    // Setup compiler
    const compiler = new Compiler();
    const X_node = compiler.createNode("X", [32, this.contextLen * this.vocabSize]);
    const targets_node = compiler.createNode("targets", [32]);

    const fc = new Linear(compiler, "fc_fc1", this.contextLen * this.vocabSize, this.hiddenDim);
    const fc2 = new Linear(compiler, "fc_fc2", this.hiddenDim, this.vocabSize);

    const h = compiler.gelu(fc.forward(compiler, X_node));
    const logits = fc2.forward(compiler, h);
    const { loss } = compiler.cross_entropy(logits, targets_node);

    const { dsl, shapes } = compiler.compile(loss, targets_node);

    this.tAdam = 0;

    while (this.isTraining) {
      // Wait until we have enough active workers
      const activeWorkers = Array.from(this.orchestrator.workers.values()).filter(w => w.status === 'idle');
      if (activeWorkers.length === 0) {
        // console.log("Waiting for idle workers...");
        await new Promise(r => setTimeout(r, 1000));
        continue;
      }

      const currentBatchSize = this.batchSize;
      const sliceSize = 32;
      const numSlices = currentBatchSize / sliceSize; // 128 / 32 = 4 tasks

      const batch = this.getBatch(currentBatchSize);

      // Prepare task inputs
      const tasksPromises: Promise<any>[] = [];

      for (let s = 0; s < numSlices; s++) {
        // Slice X and Y
        const xSlice = batch.X.subarray(s * sliceSize * this.contextLen * this.vocabSize, (s + 1) * sliceSize * this.contextLen * this.vocabSize);
        const ySlice = batch.Y.subarray(s * sliceSize, (s + 1) * sliceSize);

        const taskInputs: Record<string, { shape: number[]; data: string }> = {
          "X": {
            shape: [sliceSize, this.contextLen * this.vocabSize],
            data: TensorVM.float32ArrayToBase64(new Float32Array(xSlice))
          },
          "targets": {
            shape: [sliceSize],
            data: TensorVM.float32ArrayToBase64(new Float32Array(ySlice))
          }
        };

        // Inject current global weights
        for (const [key, val] of Object.entries(this.weights)) {
          taskInputs[key] = {
            shape: shapes[key],
            data: TensorVM.float32ArrayToBase64(val)
          };
        }

        // Submit task
        tasksPromises.push(this.orchestrator.submitTask(dsl, shapes, taskInputs));
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
            const lossData = TensorVM.base64ToFloat32Array(res[lossKey].data);
            totalLossVal += lossData[0];
          }

          // Accumulate parameter gradients: e.g. g_fc_fc1_w
          for (const key of Object.keys(this.weights)) {
            const gradKey = `g_${key}`;
            if (res[gradKey]) {
              const gradData = TensorVM.base64ToFloat32Array(res[gradKey].data);
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

        // Update weights using Adam optimizer
        this.applyAdamStep(grads);
        this.step++;
        if (this.step % 10 === 0) {
          this.epoch = Math.floor(this.step / 50);
          console.log(`Step ${this.step} | Loss: ${this.loss.toFixed(4)} | Prediction: ${this.samplePrediction()}`);
          this.broadcastStatsToDashboards();
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
