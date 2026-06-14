import * as fs from 'fs';
import * as path from 'path';

/**
 * Persistent training configuration.
 *
 * Previously the dataset path, hyperparameters, etc. had to be re-entered on
 * the dashboard after every server restart. PersistentConfig stores them in a
 * JSON file (config.json) and reloads them on boot, so the operator configures
 * once. It also holds the new fields added in this round: the DSL model file
 * path, a training target (for progress/ETA), and the wandb credentials.
 *
 * NOTE: config.json may contain a wandb API key, so it is git-ignored.
 */

export interface TrainingConfig {
  lr: number;
  batchSize: number;
  hiddenDim: number;
  contextLen: number;
  /** Absolute path to a server-side dataset text file (for large corpora). */
  datasetFilePath: string;
  /** Absolute path to a compiled .dsl model file to train from (optional). */
  dslFilePath: string;
  /** Stop target in optimizer steps. 0 = run indefinitely (no fixed target). */
  targetSteps: number;
  /** Wire precision for weights/gradients: 'fp16' (default) or 'fp32'. */
  precision: 'fp16' | 'fp32';
  /** Weights & Biases credentials (optional). */
  wandbApiKey: string;
  wandbProject: string;
  wandbEntity: string;
}

export const DEFAULT_CONFIG: TrainingConfig = {
  lr: 0.015,
  batchSize: 128,
  hiddenDim: 64,
  contextLen: 3,
  datasetFilePath: '',
  dslFilePath: '',
  targetSteps: 0,
  precision: 'fp16',
  wandbApiKey: '',
  wandbProject: 'distcompute',
  wandbEntity: ''
};

export class PersistentConfig {
  private readonly file: string;
  data: TrainingConfig;

  constructor(file = path.join(process.cwd(), 'config.json')) {
    this.file = file;
    this.data = { ...DEFAULT_CONFIG };
    this.load();
  }

  /** Load config.json if present, merging over defaults so new fields appear. */
  load(): void {
    try {
      if (fs.existsSync(this.file)) {
        const parsed = JSON.parse(fs.readFileSync(this.file, 'utf8'));
        this.data = { ...DEFAULT_CONFIG, ...parsed };
        console.log(`[Config] Loaded persisted configuration from ${this.file}`);
      }
    } catch (e) {
      console.warn(`[Config] Could not read ${this.file}, using defaults:`, e);
      this.data = { ...DEFAULT_CONFIG };
    }
  }

  /**
   * Apply a partial update (ignoring undefined values so callers can send only
   * the fields they want to change) and persist immediately.
   */
  update(patch: Partial<TrainingConfig>): TrainingConfig {
    for (const [k, v] of Object.entries(patch)) {
      if (v !== undefined && v !== null) {
        (this.data as any)[k] = v;
      }
    }
    this.save();
    return this.data;
  }

  save(): void {
    try {
      fs.writeFileSync(this.file, JSON.stringify(this.data, null, 2));
    } catch (e) {
      console.error(`[Config] Failed to persist ${this.file}:`, e);
    }
  }

  /** Config safe to send to the browser (API key redacted to a boolean). */
  redacted(): Omit<TrainingConfig, 'wandbApiKey'> & { wandbConfigured: boolean } {
    const { wandbApiKey, ...rest } = this.data;
    return { ...rest, wandbConfigured: !!wandbApiKey };
  }
}
