# Changelog — DistCompute

All notable changes to this project, newest first.

**Type tags:** 🐛 Bug Fix · ⚡ Enhancement · ✨ New Feature · 🧪 Trial · 🚀 Major Rewrite · 🔬 Diagnostic

**Status of bullets:** ✅ active in current build · 🔄 replaced by later version · ❌ removed

**Timing:** per-change durations weren't instrumented in this project, so entry **Started**/**Shipped** times are derived from file modification times (minute precision) and the **Duration** is wall-clock for the release window (includes gaps for on-cluster testing between iterations).

---

## Total Effort

- **Versions shipped:** 4 (v1.0.0 baseline + v1.1.0 + v1.2.0 + v1.3.0)
- **Sessions:** 2026-06-14
- **First change:** 2026-06-14 ~10:30 UTC
- **Latest change:** 2026-06-14 ~15:23 UTC
- **Time spent (wall-clock window):** ~4h 50m (one session; includes gaps for on-cluster testing between iterations)
- **🐛 Bug Fixes:** 12
- **⚡ Enhancements:** 1 (tolerant heartbeat timeout)
- **✨ New Features:** 13
- **🚀 Major Rewrites:** 2 (GPT-1 compiler/VM extension, trainer `ModelRuntime` refactor)
- **🧪 Tests added:** finite-difference autodiff checks for the full GPT stack (13) + a regression suite (38)
- **Files:** compiler (compiler · modules) · VM (vm.js) · server (server · orchestrator · trainer · eventlog · persistentConfig · wandb · presets · dslStats) · client (dashboard · worker · worker_thread) · tools (gen_gpt1_dsl) · tests × 3 · models (gpt1.dsl · tiny-gpt.dsl + manifests) · README · CHANGELOG

---

## [Unreleased]

*(work in flight — see commits for incremental status)*

---

## v1.3.0 — Scaling: encode weights once per step + token-based target [⚡ Enhancement + 🐛 Bug Fix]

**Started:** 2026-06-14 ~15:35 UTC
**Shipped:** 2026-06-14 ~15:55 UTC
**Duration:** ~20 min

- **Drove this:** Adding workers was *increasing* the forecast time and slowing each step — the opposite of what distribution should do. Two root causes: (1) the trainer base64-encoded the full model weights **once per slice**, so the single-threaded server did O(numSlices × modelSize) of redundant encoding every step, growing with worker count; (2) the target was measured in **steps**, but examples/step scales with worker count, so adding workers inflated both the projected work and the per-step time in the step-based ETA.

- **What we did:**
  - ✅ **Encode weights once per step** (`trainer.ts`) — the identical global weights are base64-encoded a single time per step and the strings reused across all slices, instead of re-encoding inside the per-slice loop. Removes the `numSlices×` redundant work that blocked the event loop and scaled with worker count.
  - ✅ **Token/example-based training target** (`targetTokens`) — stop and forecast by training tokens/examples processed rather than optimizer steps. `progress()` now computes `% complete`, ETA, and work-remaining from examples and measured `examples/sec`, so the numbers are stable as workers join and **ETA drops when you add workers** (verified: 4→8 workers halves ETA). Legacy `targetSteps` still works as a fallback when no token target is set. Dashboard field relabeled "Target Tokens / Examples".

- **How it helps:** Per-step server CPU no longer grows with the number of workers, and the forecast finally reflects reality — more workers means more tokens/sec and a shorter ETA, not a longer one.

- **Known limits:** Each step is still a synchronous barrier (`Promise.all`), so a single straggler still paces the step; softening that (per-step deadline / async gradient application) and moving serialization to `worker_threads` or binary frames remain open follow-ups.

- **Roadmap status:** Encode-once + token target → Done. Open: straggler barrier, multi-threaded server serialization, cosine LR schedule, GPT live sampling.

---

## v1.2.0 — Worker resilience, accurate stats & training-health graphs [🐛 Bug Fix + ⚡ Enhancement + ✨ New Feature]

**Started:** 2026-06-14 ~13:50 UTC
**Shipped:** 2026-06-14 ~15:23 UTC
**Duration:** ~1h 35m wall-clock (resilience, then accurate stats and grad/ETA metrics folded into the same release)

- **Drove this:** The event log showed healthy `MacIntel/Chrome` workers being dropped with "heartbeat timeout (15–16s silent)" a few minutes after registering. Root cause: browsers throttle background-tab timers and suspend WebSocket traffic, so the worker's 5s heartbeat stalls whenever its tab isn't foregrounded; the server then dropped it after 15s — and the worker had **no auto-reconnect**, so it stayed gone until a manual reload.

- **What we did:**
  - ✅ **Worker auto-reconnect with backoff** (`worker.html`). An unexpected socket close/error now schedules a reconnect (2s → 4s → … capped at 30s); the backoff resets on a successful connection. Refactored teardown into `cleanupConnection()`; `disconnect()` is the user/server-initiated stop (no reconnect), `handleDrop()` is the unexpected path (reconnects).
  - ✅ **No reconnect when intentional** — a `manualStop` flag means clicking Disconnect or receiving the server's `disconnect` (kick) tears down for good; only genuine drops reconnect.
  - ✅ **Tolerant heartbeat timeout** (`orchestrator.ts`) raised 15s → 45s to avoid false drops during brief throttling. An actively computing worker already stays alive via `task_completed` updating `lastSeen`.
  - ✅ **Accurate worker stats** (`orchestrator.ts`) — a worker's `completedCount` is now **preserved across reconnects** (it was reset to 0 on every re-register, so the dashboard undercounted) and a worker is **credited for every task it finishes** even if that task was already reassigned/timed out (the count/throughput/liveness update moved outside the "task still active" gate). Fixes the dashboard COMPLETED lagging each worker's own tally and workers showing throughput with 0 completed.
  - ✅ **Gradient-norm + remaining-time metrics** — the trainer computes the global gradient L2 norm each step and logs `grad_norm` plus `eta_seconds`/`eta_minutes` to wandb (and exposes `gradNorm` on the dashboard work card), so you can graph training health and time-remaining instead of just step counters.

- **How it helps:** Workers survive tab-switches, network blips, and short sleeps — they rejoin automatically instead of silently disappearing, and the server stops dropping healthy nodes over momentary heartbeat gaps. The dashboard's per-worker COMPLETED now matches each worker's own tally, and the new `grad_norm` / time-remaining graphs make it possible to actually diagnose a run (e.g. spot the LR-driven loss plateau) rather than just watch a step counter climb.

- **Known limits:** A fully backgrounded tab can still be throttled to ~1 timer/min by the browser; auto-reconnect recovers it once it's foregrounded again. For long unattended runs, keep worker tabs foregrounded and disable sleep (see the worker join guide).

- **Roadmap status:** Worker resilience → Done. Open follow-ups unchanged (GPT live sampling, LR schedule, FP16 compute).

---

## v1.1.0 — FP16 transport, full observability, and a faithful GPT-1 DSL + Tiny-GPT training [✨ New Feature + 🐛 Bug Fix + 🚀 Major Rewrite]

**Started:** 2026-06-14 ~10:30 UTC
**Shipped:** 2026-06-14 13:42 UTC
**Duration:** ~3h 12m wall-clock (one session, including on-cluster testing between iterations)

- **Drove this:** The project started as a toy char-MLP distributed trainer. A round of real testing surfaced concrete bugs (broken Kick, only 4 workers ever used, jittery mobile UI, a TS build error), and the user then pushed it much further: run thousands of workers, halve bandwidth with FP16, see how much work is left and forecast time, log to Weights & Biases, persist config, and — the headline — compile a *faithful GPT-1* to a `.dsl` and actually train a byte-level Tiny-GPT across the cluster.

- **What we did:**

  **🐛 Bug fixes (with regression tests):**
  - ✅ **Build:** replaced deprecated `moduleResolution: "node"` with `node16` in `tsconfig.json`, clearing TS 5.9 error `TS5107` so `npm run build` is clean.
  - ✅ **Kick button:** the worker table rebuilt its `innerHTML` on every stats update, destroying per-row inline handlers mid-tap. Now one delegated listener on the persistent `<tbody>` — taps always register.
  - ✅ **Scheduler / scaling:** training was hard-capped at 4 data-parallel slices, stranding workers beyond the 4th. `Trainer.planSlices()` now scales slices to idle workers (to thousands) with a configured-batch floor.
  - ✅ **Worker UI "dancing":** tabular figures + a debounced compute indicator stop the mobile jitter.
  - ✅ **Worker log:** shows the full task id (was truncated to 12 chars, making distinct tasks look identical).
  - ✅ **Batched-matmul autodiff:** the compiler's matmul backward assumed 2D operands and broke 3D batched matmul inside attention; it now transposes the last two dims correctly.
  - ✅ **wandb GraphQL:** `upsertBucket` uses `modelName` (not `projectName`, which errored), and each history row carries `_step`/`_runtime` so charts plot instead of showing "no data".

  **⚡ Performance:**
  - ✅ **Dashboard broadcasts** coalesced to ≤1 / 250 ms (was hundreds/sec with many workers heartbeating).

  **✨ FP16 transport:**
  - ✅ Weights/activations/gradients ship as 16-bit half-floats (default), halving per-task payload and server string memory; compute stays FP32 on the worker. Integer class targets stay FP32 (FP16 is exact only to 2048). Toggleable on the dashboard.

  **✨ Observability:**
  - ✅ **Worker event log** — disconnects, heartbeat/task timeouts, kicks, failures recorded with a reason to `logs/worker-events.log` + a buffer; `GET /api/logs` + dashboard panel.
  - ✅ **Persistent config** — `config.json` saves hyperparameters, dataset/DSL paths, target, precision, wandb settings; restored on boot.
  - ✅ **Weights & Biases** — fail-soft server-side REST logger (`/api/wandb/connect`); logs loss/throughput/step/workers per step.
  - ✅ **Target steps + progress/ETA** and a **DSL-based work forecast** — work measured in FLOPs from the compiled DSL (`estimateDslFlops`); every task timed; dashboard shows work done/remaining, throughput (tasks/s, examples/s, GFLOP/s), and forecast time remaining.
  - ✅ **Units / memory estimator** — parameters, weight size, payload, activation & RAM estimates for the current model and a GPT-1 preset.

  **🚀 Faithful GPT-1 + distributed Tiny-GPT training:**
  - ✅ New compiler/VM ops, each gradient-checked vs finite differences: **embedding** (gather + scatter-add) and a fused **LayerNorm** (full backward).
  - ✅ GPT modules: `Embedding`, `LayerNorm`, `CausalSelfAttention` (multi-head + causal mask), `TransformerBlock`, `GPT` (weight-tied LM head).
  - ✅ `npm run gen:dsl` emits `models/gpt1.dsl` (canonical GPT-1: 12 layers, 116.54M params, 1849 instructions) and `models/tiny-gpt.dsl` (0.84M), each with a manifest.
  - ✅ The trainer loads a `.dsl` + manifest and trains it **byte-level** over the workers via a model-agnostic `ModelRuntime` (next-byte prediction, GPT-1-style init, causal mask, FP16). Verified: loss falls from ≈ln(256)=5.55 to ~2.8.
  - ✅ Dashboard **model picker** (`GET /api/models`) shows the chosen model's architecture and hides char-MLP-only fields (hidden dim / context length / batch size) that come from the DSL.

- **How it helps:** The cluster is now usable and observable end-to-end — you can see why workers drop, how fast work is going, and how long it'll take; payloads are half the size; and you can train a real transformer architecture, not just a toy, with a clear path from Tiny-GPT to the GPT-1 reference.

- **Known limits (acknowledged, not fixed here):**
  - The canonical `gpt1.dsl` is a correct *reference*, not practically trainable on browser workers (≈466 MB weights/task + multi-GB activations). Use **Tiny-GPT** for real runs; the dashboard estimator shows why.
  - **Live text sampling for GPT models is not wired yet** (prediction box shows a placeholder); loss + wandb are the signal.
  - The GPT-1 **warmup + cosine LR schedule** is defined in the preset but not yet applied in the training loop.

- **Roadmap status:** Bug fixes → Done. FP16 transport, observability, GPT-1 DSL, Tiny-GPT training → Done. Open follow-ups: GPT live sampling, LR schedule, FP16 *compute* (mixed precision), DOM-free large-corpus byte streaming.

> **Verification:** `npm test` runs the Tensor VM math, the original autodiff gradient check, the 38-check regression suite, and the 13 GPT gradient checks — all green. `npm run gen:dsl` regenerates the DSL files and smoke-runs Tiny-GPT to a finite loss.

---

## v1.0.0 — Baseline: distributed browser-worker char-MLP trainer [✨ New Feature]

**Shipped:** (pre-existing)

- **Drove this:** Original project — a real-time, fault-tolerant distributed deep-learning trainer that orchestrates browser-based compute workers, compiling a PyTorch-like symbolic API into a register-based tensor DSL.

- **What we did:**
  - ✅ Symbolic autodiff compiler → DSL; register-based strided-tensor VM (matmul, broadcasting, softmax, GELU, cross-entropy).
  - ✅ WebSocket orchestrator with batch slicing, heartbeats, and fault-tolerant task reassignment.
  - ✅ Zero-RAM dataset seeker; Web Worker execution; admin dashboard + worker join page; Adam optimizer on the server.

- **Roadmap status:** Superseded/extended by v1.1.0.
