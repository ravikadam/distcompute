import { WebSocket } from 'ws';

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
  inputs: Record<string, { shape: number[]; data: string }>;
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

  constructor() {
    // Start background checks for heartbeats and task timeouts
    setInterval(() => this.checkHeartbeats(), 3000);
    setInterval(() => this.checkTaskTimeouts(), 5000);
  }

  // Submit a task to the queue
  submitTask(
    dsl: string,
    shapes: Record<string, number[]>,
    inputs: Record<string, { shape: number[]; data: string }>
  ): Promise<Record<string, { shape: number[]; data: string }>> {
    return new Promise((resolve, reject) => {
      const task: ComputeTask = {
        id: `task-${Date.now()}-${this.taskCounter++}`,
        dsl,
        shapes,
        inputs,
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
        this.handleWorkerDisconnect(workerId);
      }
    });

    ws.on('error', (err) => {
      console.error(`Socket error on worker ${workerId}:`, err);
      if (workerId) {
        this.handleWorkerDisconnect(workerId);
      }
    });
  }

  // Handle worker disconnects
  private handleWorkerDisconnect(workerId: string) {
    if (!this.workers.has(workerId)) return;
    console.log(`Worker disconnected: ${workerId}`);
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
      task.reject(new Error(`Failed after max retries: ${reason}`));
    }
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
        inputs: task.inputs
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
        console.warn(`Worker ${id} timed out (no heartbeat for ${(now - worker.lastSeen)/1000}s). Disconnecting.`);
        worker.ws.close();
        this.handleWorkerDisconnect(id);
      }
    }
  }

  // Monitor long running tasks and reschedule them
  private checkTaskTimeouts() {
    const now = Date.now();
    const taskTimeout = 40000; // 40 seconds

    for (const [taskId, task] of this.activeTasks.entries()) {
      if (task.startedTime && now - task.startedTime > taskTimeout) {
        console.warn(`Task ${taskId} timed out on worker ${task.assignedWorkerId}. Rescheduling.`);
        
        // Terminate worker if possible, or just disconnect it
        const worker = this.workers.get(task.assignedWorkerId!);
        if (worker) {
          worker.ws.send(JSON.stringify({ type: 'terminate_task', taskId }));
          worker.ws.close(); // Force reconnect
          this.handleWorkerDisconnect(worker.id);
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
      worker.ws.send(JSON.stringify({ type: 'disconnect' }));
      worker.ws.close();
      this.handleWorkerDisconnect(workerId);
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

  // Broadcast to all dashboard clients
  broadcastToDashboards() {
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
