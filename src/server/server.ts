import express from 'express';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';
import { Orchestrator } from './orchestrator';
import { Trainer } from './trainer';
import { EventLog } from './eventlog';
import { PersistentConfig } from './persistentConfig';
import { WandbLogger } from './wandb';
import { PRESETS, transformerParamCount } from './presets';

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ noServer: true });

const PORT = 3000;

// Shared troubleshooting log + persisted configuration.
const eventLog = new EventLog();
const persistentConfig = new PersistentConfig();

const orchestrator = new Orchestrator(eventLog);
const trainer = new Trainer(orchestrator);

// Apply any previously-saved configuration on boot so the operator doesn't
// have to re-enter the dataset path / hyperparameters after a restart.
(function applyPersistedConfig() {
  const c = persistentConfig.data;
  try {
    trainer.updateConfig({
      lr: c.lr,
      batchSize: c.batchSize,
      hiddenDim: c.hiddenDim,
      contextLen: c.contextLen,
      datasetFilePath: c.datasetFilePath || undefined,
      targetSteps: c.targetSteps,
      targetTokens: c.targetTokens,
      precision: c.precision,
      lrSchedule: c.lrSchedule,
      warmupSteps: c.warmupSteps
    });
    trainer.dslFilePath = c.dslFilePath || '';
    console.log('[Config] Restored persisted training configuration.');
  } catch (e: any) {
    console.warn('[Config] Failed to restore persisted config:', e?.message || e);
  }
})();

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
    const { lr, batchSize, hiddenDim, contextLen, corpus, datasetFilePath, targetSteps } = req.body;
    trainer.updateConfig({
      lr: lr ? parseFloat(lr) : undefined,
      batchSize: batchSize ? parseInt(batchSize) : undefined,
      hiddenDim: hiddenDim ? parseInt(hiddenDim) : undefined,
      contextLen: contextLen ? parseInt(contextLen) : undefined,
      corpus: corpus || undefined,
      datasetFilePath: datasetFilePath || undefined,
      targetSteps: targetSteps !== undefined ? parseInt(targetSteps) : undefined
    });
    // Persist hyperparameters so they survive restarts (corpus text itself is
    // not persisted here — only the server-side dataset path is).
    persistentConfig.update({
      lr: trainer.lr,
      batchSize: trainer.batchSize,
      hiddenDim: trainer.hiddenDim,
      contextLen: trainer.contextLen,
      datasetFilePath: trainer.datasetFilePath,
      targetSteps: trainer.targetSteps
    });
    res.json({ status: 'configured' });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// Non-model settings (training target, DSL file path, wandb) — persisted but
// does NOT reset weights, unlike /configure.
app.post('/api/settings', (req, res) => {
  try {
    const { targetSteps, targetTokens, dslFilePath, precision, lrSchedule, warmupSteps } = req.body;
    if (targetSteps !== undefined) trainer.targetSteps = parseInt(targetSteps) || 0;
    if (targetTokens !== undefined) trainer.targetTokens = parseInt(targetTokens) || 0;
    if (precision === 'fp16' || precision === 'fp32') trainer.precision = precision;
    if (lrSchedule === 'warmup_cosine' || lrSchedule === 'constant') trainer.lrSchedule = lrSchedule;
    if (warmupSteps !== undefined) trainer.warmupSteps = parseInt(warmupSteps) || 0;
    if (dslFilePath !== undefined) trainer.dslFilePath = String(dslFilePath);
    persistentConfig.update({
      targetSteps: trainer.targetSteps,
      targetTokens: trainer.targetTokens,
      precision: trainer.precision,
      lrSchedule: trainer.lrSchedule,
      warmupSteps: trainer.warmupSteps,
      ...(dslFilePath !== undefined ? { dslFilePath: String(dslFilePath) } : {})
    });
    res.json({ status: 'ok', config: persistentConfig.redacted() });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// Connect Weights & Biases: store the key and open a run. Fail-soft.
app.post('/api/wandb/connect', async (req, res) => {
  try {
    const { apiKey, project, entity } = req.body;
    const key = apiKey || persistentConfig.data.wandbApiKey;
    if (!key) return res.status(400).json({ error: 'wandb API key is required' });

    persistentConfig.update({
      wandbApiKey: key,
      wandbProject: project || persistentConfig.data.wandbProject,
      wandbEntity: entity || persistentConfig.data.wandbEntity
    });

    const logger = new WandbLogger();
    const ok = await logger.init({
      apiKey: key,
      project: persistentConfig.data.wandbProject,
      entity: persistentConfig.data.wandbEntity || undefined,
      config: { lr: trainer.lr, batchSize: trainer.batchSize, hiddenDim: trainer.hiddenDim, contextLen: trainer.contextLen }
    });
    trainer.wandb = ok ? logger : null;
    res.json({ status: ok ? 'connected' : 'failed', runUrl: logger.runUrl, error: logger.lastError });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// Model/training presets (canonical GPT-1 + a browser-feasible Tiny-GPT).
app.get('/api/presets', (req, res) => {
  res.json(PRESETS);
});

// Available compiled .dsl models (scans the models/ directory for manifests).
// Used by the dashboard so the operator picks a DSL and sees its architecture.
app.get('/api/models', (req, res) => {
  const dir = path.join(process.cwd(), 'models');
  const models: any[] = [];
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith('.manifest.json')) continue;
      try {
        const m = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
        models.push({
          name: m.name,
          dslPath: path.join(dir, f.replace('.manifest.json', '.dsl')),
          config: m.config,
          parameterCount: m.parameterCount,
          batch: m.batch,
          seq: m.seq,
          tokensPerTask: m.tokensPerTask,
          instructionCount: m.instructionCount
        });
      } catch { /* skip malformed manifest */ }
    }
  } catch { /* models dir may not exist yet */ }
  res.json({ models });
});

// Recent worker lifecycle events (disconnects, timeouts, kicks, failures).
app.get('/api/logs', (req, res) => {
  const limit = req.query.limit ? parseInt(String(req.query.limit)) : 100;
  const minLevel = req.query.level as any;
  res.json({
    file: eventLog.filePath,
    events: eventLog.recent(limit, minLevel)
  });
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
    progress: trainer.progress(),
    wandb: {
      configured: !!persistentConfig.data.wandbApiKey,
      active: !!(trainer.wandb && trainer.wandb.active),
      runUrl: trainer.wandb ? trainer.wandb.runUrl : '',
      lastError: trainer.wandb ? trainer.wandb.lastError : null
    },
    recentErrors: eventLog.recent(5, 'warn'),
    config: {
      lr: trainer.lr,
      batchSize: trainer.batchSize,
      hiddenDim: trainer.hiddenDim,
      contextLen: trainer.contextLen,
      corpusLength: trainer.corpus.length,
      datasetFilePath: trainer.datasetFilePath,
      targetSteps: trainer.targetSteps,
      targetTokens: trainer.targetTokens,
      precision: trainer.precision,
      lrSchedule: trainer.lrSchedule,
      warmupSteps: trainer.warmupSteps,
      dslFilePath: persistentConfig.data.dslFilePath
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
