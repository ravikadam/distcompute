import * as fs from 'fs';
import * as path from 'path';

/**
 * Structured event log for worker lifecycle problems.
 *
 * The orchestrator was silently closing workers (heartbeat/task timeouts), so
 * there was no record to troubleshoot from afterwards. EventLog captures those
 * events both to a persistent JSON-lines file (logs/worker-events.log) for
 * post-mortem analysis and to an in-memory ring buffer that the dashboard can
 * poll to show the most recent issues live.
 */

export type EventLevel = 'info' | 'warn' | 'error';
export type EventType =
  | 'register'
  | 'disconnect'
  | 'heartbeat_timeout'
  | 'task_timeout'
  | 'kick'
  | 'task_failed'
  | 'task_retry_exhausted'
  | 'error';

export interface WorkerEvent {
  time: string;            // human-readable ISO timestamp
  ts: number;              // epoch millis (for sorting/filtering)
  level: EventLevel;
  type: EventType;
  message: string;
  workerId?: string;
  taskId?: string;
  meta?: Record<string, any>;
}

export class EventLog {
  private buffer: WorkerEvent[] = [];
  private readonly maxBuffer: number;
  private readonly logFile: string;

  constructor(opts: { logDir?: string; maxBuffer?: number } = {}) {
    const logDir = opts.logDir ?? path.join(process.cwd(), 'logs');
    this.maxBuffer = opts.maxBuffer ?? 1000;
    this.logFile = path.join(logDir, 'worker-events.log');
    // Best-effort directory creation; logging must never crash the server.
    try { fs.mkdirSync(logDir, { recursive: true }); } catch { /* ignore */ }
  }

  /** Record an event: appended to disk and kept in the recent-events buffer. */
  record(
    type: EventType,
    message: string,
    opts: { level?: EventLevel; workerId?: string; taskId?: string; meta?: Record<string, any> } = {}
  ): WorkerEvent {
    const ev: WorkerEvent = {
      time: new Date().toISOString(),
      ts: Date.now(),
      level: opts.level ?? 'info',
      type,
      message,
      workerId: opts.workerId,
      taskId: opts.taskId,
      meta: opts.meta
    };

    this.buffer.push(ev);
    if (this.buffer.length > this.maxBuffer) this.buffer.shift();

    // Append as one JSON object per line so the file is both grep-able and
    // machine-parsable. Wrapped in try/catch: a failed write must not break
    // the orchestrator.
    try {
      fs.appendFileSync(this.logFile, JSON.stringify(ev) + '\n');
    } catch (e) {
      console.error('[EventLog] failed to write log file:', e);
    }
    return ev;
  }

  /** Most recent events (newest first), optionally filtered by level. */
  recent(limit = 100, minLevel?: EventLevel): WorkerEvent[] {
    const rank: Record<EventLevel, number> = { info: 0, warn: 1, error: 2 };
    let events = this.buffer;
    if (minLevel) events = events.filter(e => rank[e.level] >= rank[minLevel]);
    return events.slice(-limit).reverse();
  }

  /** Absolute path of the on-disk log, surfaced to the dashboard/operator. */
  get filePath(): string {
    return this.logFile;
  }
}
