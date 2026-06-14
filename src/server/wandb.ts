import * as https from 'https';

/**
 * Minimal server-side Weights & Biases logger (no official Node SDK exists).
 *
 * It speaks the same HTTP protocol the official clients use:
 *   1. GraphQL `viewer` to resolve the default entity (if none supplied).
 *   2. GraphQL `upsertBucket` to create/open a run.
 *   3. POST metric rows to the run's `file_stream` endpoint as a growing
 *      `wandb-history.jsonl` file.
 *
 * Design goals:
 *   - Fail-soft: a wandb outage or bad key must NEVER interrupt training.
 *     Every network call is wrapped and errors are reported, not thrown.
 *   - Self-contained: only Node's built-in `https`, no extra dependencies.
 *
 * Because this cannot be exercised against a live wandb account in this
 * environment, the payload-building logic (`buildHistoryChunk`) is split out so
 * it can be unit-tested offline; the network plumbing is best-effort.
 */

const WANDB_HOST = 'api.wandb.ai';

export interface WandbInit {
  apiKey: string;
  project: string;
  entity?: string;
  runName?: string;
  config?: Record<string, any>;
}

export class WandbLogger {
  private apiKey = '';
  private entity = '';
  private project = '';
  private runId = '';
  private runName = '';
  private historyOffset = 0;
  private startTimeMs = 0;
  enabled = false;
  lastError: string | null = null;

  /** True only after a run has been successfully created. */
  get active(): boolean {
    return this.enabled && !!this.runId;
  }

  get runUrl(): string {
    return this.active ? `https://wandb.ai/${this.entity}/${this.project}/runs/${this.runId}` : '';
  }

  /**
   * Build one file_stream chunk for a batch of metric rows starting at the
   * current history offset. Pure/inspectable so it can be unit-tested.
   */
  buildHistoryChunk(rows: Record<string, any>[]): { body: any; nextOffset: number } {
    // wandb plots metrics against its internal `_step` axis and expects
    // `_runtime`/`_timestamp` per row. Without `_step` the charts show
    // "no data for the selected runs". We assign `_step` from the history
    // offset so it increases by one per logged row.
    const now = Date.now();
    const lines = rows.map((r, i) => JSON.stringify({
      ...r,
      _step: this.historyOffset + i,
      _runtime: this.startTimeMs ? (now - this.startTimeMs) / 1000 : 0,
      _timestamp: now / 1000
    }));
    const body = {
      files: {
        'wandb-history.jsonl': { offset: this.historyOffset, content: lines }
      }
    };
    return { body, nextOffset: this.historyOffset + lines.length };
  }

  /** Authenticate and create a run. Resolves to true on success. */
  async init(opts: WandbInit): Promise<boolean> {
    this.apiKey = opts.apiKey;
    this.project = opts.project || 'distcompute';
    this.runName = opts.runName || `dist-run-${Date.now()}`;
    this.runId = '';
    this.historyOffset = 0;
    this.startTimeMs = Date.now();
    this.lastError = null;

    if (!this.apiKey) {
      this.enabled = false;
      return false;
    }

    try {
      // Resolve entity (the wandb "team"/username) if the operator didn't give one.
      this.entity = opts.entity || (await this.fetchDefaultEntity());
      if (!this.entity) throw new Error('Could not resolve wandb entity for this API key');

      // Create the run.
      const runId = await this.upsertRun(opts.config || {});
      if (!runId) throw new Error('upsertBucket returned no run id');
      this.runId = runId;
      this.enabled = true;
      console.log(`[wandb] Logging to run ${this.runUrl}`);
      return true;
    } catch (e: any) {
      this.enabled = false;
      this.lastError = e?.message || String(e);
      console.error('[wandb] init failed (training continues without wandb):', this.lastError);
      return false;
    }
  }

  /** Log a single metric row (e.g. { step, loss, throughput, workers }). */
  async log(row: Record<string, any>): Promise<void> {
    if (!this.active) return;
    const { body, nextOffset } = this.buildHistoryChunk([row]);
    try {
      await this.postJson(`/files/${this.entity}/${this.project}/${this.runId}/file_stream`, body);
      this.historyOffset = nextOffset;
    } catch (e: any) {
      this.lastError = e?.message || String(e);
      console.error('[wandb] log failed (continuing):', this.lastError);
    }
  }

  /** Mark the run finished (best-effort). */
  async finish(): Promise<void> {
    if (!this.active) return;
    try {
      await this.postJson(`/files/${this.entity}/${this.project}/${this.runId}/file_stream`, {
        complete: true,
        exitcode: 0
      });
    } catch { /* ignore */ }
  }

  // ---- internal HTTP helpers ------------------------------------------------

  private async fetchDefaultEntity(): Promise<string> {
    const data = await this.graphql('query { viewer { entity } }', {});
    return data?.viewer?.entity || '';
  }

  // Exposed (not private) so a test can guard the field names offline. wandb's
  // UpsertBucketInput uses `modelName` for the project and `entityName` for the
  // entity; `projectName` is rejected as 'Unknown field.'
  buildUpsertMutation(): string {
    return `
      mutation Upsert($entity: String!, $project: String!, $name: String!, $config: JSONString) {
        upsertBucket(input: { entityName: $entity, modelName: $project, name: $name, config: $config }) {
          bucket { id name }
        }
      }`;
  }

  private async upsertRun(config: Record<string, any>): Promise<string> {
    const mutation = this.buildUpsertMutation();
    const variables = {
      entity: this.entity,
      project: this.project,
      name: this.runName,
      config: JSON.stringify(config)
    };
    const data = await this.graphql(mutation, variables);
    return data?.upsertBucket?.bucket?.name || '';
  }

  private graphql(query: string, variables: Record<string, any>): Promise<any> {
    return this.postJson('/graphql', { query, variables }).then(res => {
      if (res.errors) throw new Error(JSON.stringify(res.errors));
      return res.data;
    });
  }

  /** POST JSON with wandb HTTP Basic auth ("api:<key>"). */
  private postJson(path: string, payload: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const data = Buffer.from(JSON.stringify(payload));
      const auth = 'Basic ' + Buffer.from(`api:${this.apiKey}`).toString('base64');
      const req = https.request(
        {
          host: WANDB_HOST,
          path,
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Content-Length': data.length,
            Authorization: auth
          }
        },
        res => {
          const chunks: Buffer[] = [];
          res.on('data', c => chunks.push(c));
          res.on('end', () => {
            const text = Buffer.concat(chunks).toString('utf8');
            if ((res.statusCode || 500) >= 400) {
              return reject(new Error(`HTTP ${res.statusCode}: ${text.slice(0, 300)}`));
            }
            try { resolve(text ? JSON.parse(text) : {}); }
            catch { resolve({}); }
          });
        }
      );
      req.on('error', reject);
      req.write(data);
      req.end();
    });
  }
}
