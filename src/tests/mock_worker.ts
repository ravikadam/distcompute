import WebSocket from 'ws';
import { Tensor, TensorVM } from '../public/vm';

const serverUrl = process.argv[2] || 'ws://localhost:3000/worker';
const workerId = process.argv[3] || `node-mock-${Math.floor(Math.random() * 900 + 100)}`;

console.log(`[Mock Worker ${workerId}] Starting and connecting to ${serverUrl}...`);

const ws = new WebSocket(serverUrl);
let heartInterval: NodeJS.Timeout | null = null;

ws.on('open', () => {
  console.log(`[Mock Worker ${workerId}] Connected to orchestrator.`);

  // Register
  ws.send(JSON.stringify({
    type: 'register',
    workerId,
    deviceInfo: {
      browser: 'NodeJS-Mock',
      platform: 'macOS (Console)',
      cores: 4,
      userAgent: 'node-ws-client'
    }
  }));

  // Start heartbeat
  heartInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'heartbeat',
        throughput: 850000 // mock elements/sec
      }));
    }
  }, 5000);
});

ws.on('message', (data: string) => {
  try {
    const message = JSON.parse(data);

    if (message.type === 'assign_task') {
      const { taskId, dsl, shapes, inputs } = message;
      // console.log(`[Mock Worker ${workerId}] Assigned task ${taskId.substring(0, 12)}...`);

      const vm = new TensorVM() as any;

      // 1. Allocate tensors
      for (const [name, shape] of Object.entries(shapes)) {
        vm.tensors[name] = new Tensor(shape as number[]);
      }

      // 2. Load inputs
      for (const [name, input] of Object.entries(inputs)) {
        const flatData = TensorVM.base64ToFloat32Array((input as any).data);
        vm.tensors[name].data.set(flatData);
      }

      // 3. Execute
      const start = performance.now();
      vm.execute(dsl);
      const end = performance.now();
      const durationMs = end - start;

      // 4. Gather outputs
      const outputs: any = {};
      for (const name of Object.keys(vm.tensors)) {
        const isGrad = name.startsWith('g_');
        const isLoss = name.startsWith('t') && vm.tensors[name].size === 1;

        if (isGrad || isLoss) {
          const t = vm.tensors[name];
          const cont = t.contiguous();
          outputs[name] = {
            shape: t.shape,
            data: TensorVM.float32ArrayToBase64(cont.data)
          };
        }
      }

      // Send results back
      ws.send(JSON.stringify({
        type: 'task_completed',
        taskId,
        outputs
      }));
    } 
    
    else if (message.type === 'disconnect') {
      console.log(`[Mock Worker ${workerId}] Received disconnect request from server.`);
      cleanup();
    }
  } catch (err: any) {
    console.error(`[Mock Worker ${workerId}] Error executing task:`, err.message);
    // Send failure back
    ws.send(JSON.stringify({
      type: 'task_failed',
      error: err.message
    }));
  }
});

ws.on('close', () => {
  console.log(`[Mock Worker ${workerId}] Connection closed.`);
  cleanup();
});

ws.on('error', (err) => {
  console.error(`[Mock Worker ${workerId}] WebSocket error:`, err);
  cleanup();
});

function cleanup() {
  if (heartInterval) {
    clearInterval(heartInterval);
    heartInterval = null;
  }
  process.exit(0);
}
