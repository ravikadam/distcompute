import * as fs from 'fs';
import * as path from 'path';

/**
 * On-disk checkpoint store for training runs.
 *
 * Each run gets its own directory under checkpoints/<runId>/ with:
 *   - meta.json    — small, fast to list: run id, model, step, loss, config,
 *                    cumulative counters, timestamps.
 *   - latest.ckpt  — the heavy state: base64-encoded weights + Adam m/v.
 *
 * This lets training survive a machine restart / network change: the loop saves
 * a checkpoint every N steps, and the dashboard can list past runs and resume
 * the latest checkpoint of any of them.
 */

export interface CheckpointMeta {
  runId: string;
  model: string;            // dsl filename or 'char-mlp'
  mode: 'char' | 'gpt';
  createdAt: string;
  updatedAt: string;
  step: number;
  epoch: number;
  tAdam: number;
  loss: number;
  gradNorm: number;
  cumulativeTasks: number;
  cumulativeExamples: number;
  cumulativeFlops: number;
  config: Record<string, any>;
}

// The big blob: each is { paramName: base64(Float32Array) }.
export interface CheckpointBlob {
  weights: Record<string, string>;
  adamM: Record<string, string>;
  adamV: Record<string, string>;
}

export class CheckpointStore {
  readonly dir: string;

  constructor(dir = path.join(process.cwd(), 'checkpoints')) {
    this.dir = dir;
    try { fs.mkdirSync(dir, { recursive: true }); } catch { /* ignore */ }
  }

  private runDir(runId: string): string {
    return path.join(this.dir, runId);
  }

  /** Persist a checkpoint (overwrites the run's latest). Fail-soft. */
  save(runId: string, meta: CheckpointMeta, blob: CheckpointBlob): boolean {
    try {
      const d = this.runDir(runId);
      fs.mkdirSync(d, { recursive: true });
      // Write the heavy blob first, then meta — so a half-written checkpoint
      // never advertises a step it doesn't actually have data for.
      fs.writeFileSync(path.join(d, 'latest.ckpt'), JSON.stringify(blob));
      fs.writeFileSync(path.join(d, 'meta.json'), JSON.stringify(meta, null, 2));
      return true;
    } catch (e) {
      console.error(`[Checkpoints] save failed for ${runId}:`, e);
      return false;
    }
  }

  /** List all runs (metadata only), newest-updated first. */
  listRuns(): CheckpointMeta[] {
    const runs: CheckpointMeta[] = [];
    try {
      for (const name of fs.readdirSync(this.dir)) {
        const metaPath = path.join(this.dir, name, 'meta.json');
        if (!fs.existsSync(metaPath)) continue;
        try { runs.push(JSON.parse(fs.readFileSync(metaPath, 'utf8'))); }
        catch { /* skip corrupt */ }
      }
    } catch { /* dir may not exist */ }
    runs.sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''));
    return runs;
  }

  exists(runId: string): boolean {
    return fs.existsSync(path.join(this.runDir(runId), 'latest.ckpt'));
  }

  /** Load a run's latest checkpoint (meta + blob), or null if missing. */
  load(runId: string): { meta: CheckpointMeta; blob: CheckpointBlob } | null {
    try {
      const d = this.runDir(runId);
      const meta = JSON.parse(fs.readFileSync(path.join(d, 'meta.json'), 'utf8'));
      const blob = JSON.parse(fs.readFileSync(path.join(d, 'latest.ckpt'), 'utf8'));
      return { meta, blob };
    } catch (e) {
      console.error(`[Checkpoints] load failed for ${runId}:`, e);
      return null;
    }
  }
}
