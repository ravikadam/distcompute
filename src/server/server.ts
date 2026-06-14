import express from 'express';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import * as path from 'path';
import * as os from 'os';
import { Orchestrator } from './orchestrator';
import { Trainer } from './trainer';

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ noServer: true });

const PORT = 3000;

const orchestrator = new Orchestrator();
const trainer = new Trainer(orchestrator);

// Serve static public assets
app.use(express.static(path.join(__dirname, '../public')));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Helper to find local IP address
function getLocalIpAddress(): string {
  const interfaces = os.networkInterfaces();
  for (const devName in interfaces) {
    const iface = interfaces[devName];
    if (iface) {
      for (const alias of iface) {
        if (alias.family === 'IPv4' && !alias.internal) {
          return alias.address;
        }
      }
    }
  }
  return '127.0.0.1';
}

// API endpoints
app.post('/api/training/configure', (req, res) => {
  try {
    const { lr, batchSize, hiddenDim, contextLen, corpus, datasetFilePath } = req.body;
    trainer.updateConfig({
      lr: lr ? parseFloat(lr) : undefined,
      batchSize: batchSize ? parseInt(batchSize) : undefined,
      hiddenDim: hiddenDim ? parseInt(hiddenDim) : undefined,
      contextLen: contextLen ? parseInt(contextLen) : undefined,
      corpus: corpus || undefined,
      datasetFilePath: datasetFilePath || undefined
    });
    res.json({ status: 'configured' });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/training/start', (req, res) => {
  if (trainer.isTraining) {
    res.json({ status: 'already_running', loss: trainer.loss, step: trainer.step });
  } else {
    trainer.start().catch(err => console.error("Trainer error:", err));
    res.json({ status: 'started' });
  }
});

app.post('/api/training/stop', (req, res) => {
  trainer.stop();
  res.json({ status: 'stopped' });
});

app.get('/api/training/status', (req, res) => {
  res.json({
    isTraining: trainer.isTraining,
    step: trainer.step,
    epoch: trainer.epoch,
    loss: trainer.loss,
    prediction: trainer.samplePrediction(),
    workersCount: orchestrator.workers.size,
    workerJoinUrl: `http://${localIp}:${PORT}/worker.html`,
    config: {
      lr: trainer.lr,
      batchSize: trainer.batchSize,
      hiddenDim: trainer.hiddenDim,
      contextLen: trainer.contextLen,
      corpusLength: trainer.corpus.length,
      datasetFilePath: trainer.datasetFilePath
    }
  });
});

app.get('/api/model/weights', (req, res) => {
  res.setHeader('Content-disposition', 'attachment; filename=model_weights.json');
  res.setHeader('Content-type', 'application/json');
  res.send(trainer.exportWeights());
});

// Upgrade HTTP to WS
server.on('upgrade', (request, socket, head) => {
  const pathname = request.url ? new URL(request.url, `http://${request.headers.host}`).pathname : '';

  if (pathname === '/worker' || pathname === '/dashboard') {
    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit('connection', ws, request);
    });
  } else {
    socket.destroy();
  }
});

// WebSocket router
wss.on('connection', (ws: WebSocket, req) => {
  const url = req.url || '';
  if (url.startsWith('/worker')) {
    orchestrator.handleWorkerConnection(ws);
  } else if (url.startsWith('/dashboard')) {
    orchestrator.registerDashboard(ws);
    ws.on('message', (message: string) => {
      try {
        const payload = JSON.parse(message);
        if (payload.type === 'kick') {
          orchestrator.kickWorker(payload.workerId);
        } else if (payload.type === 'start') {
          trainer.start().catch(err => console.error("Trainer error:", err));
        } else if (payload.type === 'stop') {
          trainer.stop();
        }
      } catch (err) {
        console.error("Dashboard command parse error:", err);
      }
    });
  }
});

const localIp = getLocalIpAddress();

server.listen(PORT, '0.0.0.0', () => {
  console.log(`====================================================`);
  console.log(`🚀 Distributed Compute Server listening on 0.0.0.0:${PORT}`);
  console.log(`🖥️  Local Dashboard: http://localhost:${PORT}/dashboard.html`);
  console.log(`🖥️  Network Dashboard: http://${localIp}:${PORT}/dashboard.html`);
  console.log(`📱 Worker Join URL: http://${localIp}:${PORT}/worker.html`);
  console.log(`====================================================`);
});
