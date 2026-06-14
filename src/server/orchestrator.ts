import { WebSocket } from 'ws';
import { EventLog } from './eventlog';

export interface WorkerNode {
  id: string;
  ws: WebSocket;
  status: 'idle' | 'busy' | 'offline';
  deviceInfo: {
    browser: string;
    platform: string;
    cores: number;
    userAgent: string;
  };
  lastSeen: number;
  completedCount: number;
  failedCount: number;
  throughput: number; // FLOPS or processed elements/sec
}

export interface ComputeTask {
  id: string;
  dsl: string;
  shapes: Record<string, number[]>;
  // Each input carries its own dtype so e.g. integer targets stay fp32 while
  // weights ride along as fp16. `precision` is the dtype the worker should use
  // when encoding its gradient/loss outputs back.
  inputs: Record<string, { shape: number[]; data: string; dtype?: string }>;
  precision: string;
  assignedWorkerId: string | null;
  startedTime: number | null;
  retries: number;
  resolve: (outputs: Record<string, { shape: number[]; data: string }>) => void;
  reject: (err: any) => void;
}

export class Orchestrator {
  workers = new Map<string, WorkerNode>();
  taskQueue: ComputeTask[] = [];
  activeTasks = new Map<string, ComputeTask>();
  dashboardSockets = new Set<WebSocket>();
  private taskCounter = 0;
  // Persistent troubleshooting log for worker lifecycle problems. Injected so
  // it can be shared with the rest of the server (and stubbed in tests).
  eventLog: EventLog;

  // Per-task timing (rolling) used to forecast remaining time. computeMs is the
  // worker's reported execution time; roundTripMs is queue+network+compute as
  // seen by the server.
  totalTasksCompleted = 0;
  private recentComputeMs: number[] = [];
  private recentRoundTripMs: number[] = [];
  private static readonly TIMING_WINDOW = 200;

  constructor(eventLog?: EventLog) {
    this.eventLog = eventLog ?? new EventLog();
    // Start background checks for heartbeats and task timeouts
    setInterval(() => this.checkHeartbeats(), 3000);
    setInterval(() => this.checkTaskTimeouts(), 5000);
  }

  // Submit a task to the queue
  submitTask(
    dsl: string,
    shapes: Record<string, number[]>,
    inputs: Record<string, { shape: number[]; data: string; dtype?: string }>,
    precision: string = 'fp32'
  ): Promise<Record<string, { shape: number[]; data: string; dtype?: string }>> {
    return new Promise((resolve, reject) => {
      const task: ComputeTask = {
        id: `task-${Date.now()}-${this.taskCounter++}`,
        dsl,
        shapes,
        inputs,
        precision,
        assignedWorkerId: null,
        startedTime: null,
        retries: 0,
        resolve,
        reject
      };

      this.taskQueue.push(task);
      this.schedule();
    });
  }

  // Handle a new WebSocket connection from a worker
  handleWorkerConnection(ws: WebSocket) {
    let workerId: string | null = null;

    ws.on('message', (message: string) => {
      try {
        const payload = JSON.parse(message);
        
        if (payload.type === 'register') {
          workerId = payload.workerId;
          const node: WorkerNode = {
            id: workerId!,
            ws,
            status: 'idle',
            deviceInfo: payload.deviceInfo || { browser: 'Unknown', platform: 'Unknown', cores: 1, userAgent: '' },
            lastSeen: Date.now(),
            completedCount: 0,
            failedCount: 0,
            throughput: 0
          };
          this.workers.set(workerId!, node);
          console.log(`Worker registered: ${workerId} (${node.deviceInfo.platform}, ${node.deviceInfo.browser})`);
          this.eventLog.record('register', `Worker registered (${node.deviceInfo.platform}, ${node.deviceInfo.browser})`, {
            workerId: workerId!,
            meta: { deviceInfo: node.deviceInfo }
          });
          this.broadcastToDashboards();
          this.schedule();
        }

        else if (payload.type === 'heartbeat') {
          if (workerId && this.workers.has(workerId)) {
            const worker = this.workers.get(workerId)!;
            worker.lastSeen = Date.now();
            worker.throughput = payload.throughput || 0;
            this.broadcastToDashboards();
          }
        }

        else if (payload.type === 'task_completed') {
          const { taskId, outputs } = payload;
          const task = this.activeTasks.get(taskId);
          if (task) {
            // Record how long this task took (worker compute + server round-trip).
            const roundTripMs = task.startedTime ? Date.now() - task.startedTime : 0;
            const computeMs = typeof payload.durationMs === 'number' ? payload.durationMs : roundTripMs;
            this.recordTaskTiming(computeMs, roundTripMs);

            this.activeTasks.delete(taskId);
            if (workerId && this.workers.has(workerId)) {
              const worker = this.workers.get(workerId)!;
              worker.status = 'idle';
              worker.completedCount++;
              worker.lastSeen = Date.now();
            }
            task.resolve(outputs);
            this.broadcastToDashboards();
            this.schedule();
          }
        }

        else if (payload.type === 'task_failed') {
          const { taskId, error } = payload;
          const task = this.activeTasks.get(taskId);
          if (task) {
            console.warn(`Task ${taskId} failed on worker ${workerId}: ${error}`);
            this.eventLog.record('task_failed', `Task failed on worker: ${error}`, {
              level: 'error', workerId: workerId ?? undefined, taskId, meta: { error }
            });
            this.activeTasks.delete(taskId);
            if (workerId && this.workers.has(workerId)) {
              const worker = this.workers.get(workerId)!;
              worker.status = 'idle';
              worker.failedCount++;
            }
            this.rescheduleTask(task, `Worker reported error: ${error}`);
          }
        }
      } catch (err: any) {
        console.error('Failed processing message from worker:', err);
      }
    });

    ws.on('close', () => {
      if (workerId) {
        this.handleWorkerDisconnect(workerId, 'WebSocket closed');
      }
    });

    ws.on('error', (err) => {
      console.error(`Socket error on worker ${workerId}:`, err);
      if (workerId) {
        this.handleWorkerDisconnect(workerId, `Socket error: ${(err as any)?.message || err}`);
      }
    });
  }

  // Handle worker disconnects. `reason` is recorded to the event log so the
  // operator can later see *why* each worker dropped (closed, timed out, kicked).
  private handleWorkerDisconnect(workerId: string, reason = 'unknown') {
    if (!this.workers.has(workerId)) return;
    console.log(`Worker disconnected: ${workerId} (${reason})`);
    this.eventLog.record('disconnect', `Worker disconnected: ${reason}`, {
      level: reason.toLowerCase().includes('error') ? 'error' : 'info',
      workerId,
      meta: { reason }
    });
    this.workers.delete(workerId);

    // Reschedule any task it was running
    for (const [taskId, task] of this.activeTasks.entries()) {
      if (task.assignedWorkerId === workerId) {
        this.activeTasks.delete(taskId);
        this.rescheduleTask(task, `Worker ${workerId} disconnected`);
      }
    }

    this.broadcastToDashboards();
    this.schedule();
  }

  // Reschedule a task with retry limit
  private rescheduleTask(task: ComputeTask, reason: string) {
    if (task.retries < 3) {
      task.retries++;
      task.assignedWorkerId = null;
      task.startedTime = null;
      console.log(`Rescheduling task ${task.id} (Retry ${task.retries}/3) due to: ${reason}`);
      this.taskQueue.unshift(task); // Put it back at the front
      this.schedule();
    } else {
      console.error(`Task ${task.id} failed after maximum retries. Reason: ${reason}`);
      this.eventLog.record('task_retry_exhausted', `Task abandoned after ${task.retries} retries: ${reason}`, {
        level: 'error', taskId: task.id, meta: { reason, retries: task.retries }
      });
      task.reject(new Error(`Failed after max retries: ${reason}`));
    }
  }

  // Record one task's timing into the rolling windows.
  private recordTaskTiming(computeMs: number, roundTripMs: number) {
    this.totalTasksCompleted++;
    this.recentComputeMs.push(computeMs);
    this.recentRoundTripMs.push(roundTripMs);
    if (this.recentComputeMs.length > Orchestrator.TIMING_WINDOW) this.recentComputeMs.shift();
    if (this.recentRoundTripMs.length > Orchestrator.TIMING_WINDOW) this.recentRoundTripMs.shift();
  }

  // Aggregate per-task timing for the dashboard / forecast.
  getTaskTiming(): { totalCompleted: number; avgComputeMs: number; avgRoundTripMs: number; sampleSize: number } {
    const avg = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
    return {
      totalCompleted: this.totalTasksCompleted,
      avgComputeMs: avg(this.recentComputeMs),
      avgRoundTripMs: avg(this.recentRoundTripMs),
      sampleSize: this.recentComputeMs.length
    };
  }

  // Core Scheduling Loop
  schedule() {
    if (this.taskQueue.length === 0) return;

    // Get all idle workers
    const idleWorkers = Array.from(this.workers.values()).filter(w => w.status === 'idle');
    if (idleWorkers.length === 0) return;

    for (const worker of idleWorkers) {
      if (this.taskQueue.length === 0) break;

      const task = this.taskQueue.shift()!;
      task.assignedWorkerId = worker.id;
      task.startedTime = Date.now();
      
      worker.status = 'busy';
      this.activeTasks.set(task.id, task);

      // Send assign_task command
      worker.ws.send(JSON.stringify({
        type: 'assign_task',
        taskId: task.id,
        dsl: task.dsl,
        shapes: task.shapes,
        inputs: task.inputs,
        precision: task.precision
      }));
    }

    this.broadcastToDashboards();
  }

  // Monitor heartbeats and drop unresponsive workers
  private checkHeartbeats() {
    const now = Date.now();
    const heartbeatTimeout = 15000; // 15 seconds

    for (const [id, worker] of this.workers.entries()) {
      if (now - worker.lastSeen > heartbeatTimeout) {
        const silentFor = ((now - worker.lastSeen) / 1000).toFixed(1);
        console.warn(`Worker ${id} timed out (no heartbeat for ${silentFor}s). Disconnecting.`);
        this.eventLog.record('heartbeat_timeout', `No heartbeat for ${silentFor}s — dropping worker`, {
          level: 'warn', workerId: id, meta: { silentForSeconds: Number(silentFor), thresholdMs: heartbeatTimeout }
        });
        worker.ws.close();
        this.handleWorkerDisconnect(id, `Heartbeat timeout (${silentFor}s silent)`);
      }
    }
  }

  // Monitor long running tasks and reschedule them
  private checkTaskTimeouts() {
    const now = Date.now();
    const taskTimeout = 40000; // 40 seconds

    for (const [taskId, task] of this.activeTasks.entries()) {
      if (task.startedTime && now - task.startedTime > taskTimeout) {
        const ranForMs = now - task.startedTime;
        console.warn(`Task ${taskId} timed out on worker ${task.assignedWorkerId}. Rescheduling.`);
        this.eventLog.record('task_timeout', `Task exceeded ${taskTimeout / 1000}s — cancelling and dropping worker`, {
          level: 'warn', workerId: task.assignedWorkerId ?? undefined, taskId,
          meta: { ranForMs, thresholdMs: taskTimeout }
        });

        // Terminate worker if possible, or just disconnect it
        const worker = this.workers.get(task.assignedWorkerId!);
        if (worker) {
          worker.ws.send(JSON.stringify({ type: 'terminate_task', taskId }));
          worker.ws.close(); // Force reconnect
          this.handleWorkerDisconnect(worker.id, `Task ${taskId} timed out after ${(ranForMs / 1000).toFixed(1)}s`);
        } else {
          this.activeTasks.delete(taskId);
          this.rescheduleTask(task, 'Task execution timed out');
        }
      }
    }
  }

  // Disconnect a worker by command from the dashboard
  kickWorker(workerId: string) {
    const worker = this.workers.get(workerId);
    if (worker) {
      console.log(`Kicking worker by administrative request: ${workerId}`);
      this.eventLog.record('kick', 'Worker kicked by administrator from dashboard', {
        level: 'warn', workerId
      });
      worker.ws.send(JSON.stringify({ type: 'disconnect' }));
      worker.ws.close();
      this.handleWorkerDisconnect(workerId, 'Kicked by administrator');
    }
  }

  // Register Dashboard WS client
  registerDashboard(ws: WebSocket) {
    this.dashboardSockets.add(ws);
    // Send initial stats
    this.sendDashboardStats(ws);

    ws.on('close', () => {
      this.dashboardSockets.delete(ws);
    });
  }

  // Coalesce dashboard broadcasts.
  //
  // broadcastToDashboards() is called on every worker event (register,
  // heartbeat, task completion, etc.). With thousands of workers each
  // heartbeating, that is hundreds of broadcasts per second, and every
  // broadcast serializes the full worker list to every dashboard — O(workers)
  // each time. That floods the network and forces the dashboard to rebuild
  // its entire table constantly (which also made the "Kick" button unclickable
  // because the row was destroyed mid-tap). Instead we collapse bursts of
  // events into at most one broadcast per BROADCAST_INTERVAL_MS.
  private broadcastScheduled = false;
  private static readonly BROADCAST_INTERVAL_MS = 250;

  broadcastToDashboards() {
    if (this.broadcastScheduled) return; // a flush is already queued
    this.broadcastScheduled = true;
    setTimeout(() => {
      this.broadcastScheduled = false;
      this.flushDashboards();
    }, Orchestrator.BROADCAST_INTERVAL_MS);
  }

  // Actually push current stats to every connected dashboard client.
  private flushDashboards() {
    for (const ws of this.dashboardSockets) {
      if (ws.readyState === WebSocket.OPEN) {
        this.sendDashboardStats(ws);
      }
    }
  }

  private sendDashboardStats(ws: WebSocket) {
    const workersList = Array.from(this.workers.values()).map(w => ({
      id: w.id,
      status: w.status,
      deviceInfo: w.deviceInfo,
      completedCount: w.completedCount,
      failedCount: w.failedCount,
      throughput: w.throughput,
      latency: Date.now() - w.lastSeen
    }));

    ws.send(JSON.stringify({
      type: 'stats_update',
      workers: workersList,
      queueSize: this.taskQueue.length,
      activeCount: this.activeTasks.size
    }));
  }
}
