# ⚡ DistCompute: Distributed Browser-Based Deep Learning Training Cluster

A real-time, fault-tolerant, distributed deep learning training system that orchestrates browser-based compute workers (such as mobile phones, tablets, or secondary PCs) to train neural network models. 

By compiling neural networks from a high-level PyTorch-like symbolic API into a custom register-based DSL, the orchestrator divides training batches, schedules slices to connected worker nodes via WebSockets, accumulates their Float32Array gradients, and applies optimization steps (Adam) on the central server.

---

## 📐 System Architecture

The following diagram shows the end-to-end data flow, network boundaries, and multi-threaded execution pipeline of the cluster:

```mermaid
flowchart TB
    subgraph LAN["Local Area Network (LAN) Boundary"]
        subgraph CentralServer["Central Server (Node.js & TypeScript)"]
            direction TB
            ExpressServer["Express HTTP Server<br>(Port 3000)"]
            WSServer["WebSocket Server<br>(/worker & /dashboard)"]
            
            Compiler["Symbolic Compiler<br>(Topological Autodiff)"]
            Trainer["Trainer Loop<br>(Adam Optimizer)"]
            Scheduler["Orchestrator Scheduler<br>(Fault-Tolerant Queue)"]
            DiskData["Zero-RAM Dataset Seeker<br>(fs.readSync seeking)"]
            
            Trainer <-->|"Constructs Models"| Compiler
            Trainer <-->|"Batches & Gradients"| Scheduler
            Trainer <-->|"On-demand Slices"| DiskData
            ExpressServer <--> WSServer
        end

        subgraph ClientDash["Management Interface"]
            Dashboard["Admin Dashboard<br>(dashboard.html)"]
        end

        subgraph WorkerNodes["Compute Worker Grid"]
            direction LR
            Worker1["Worker Node 1<br>(Mobile Phone / PC Browser)"]
            Worker2["Worker Node 2<br>(Mobile Phone / PC Browser)"]
            WorkerN["Worker Node N<br>(Mobile Phone / PC Browser)"]
        end
    end

    subgraph InternalWorker["Inside Each Worker Node (Sandbox)"]
        MainThread["Browser UI Thread<br>(WebSocket Client & Logger)"]
        WebWorker["Web Worker Thread<br>(worker_thread.js)"]
        VM["Tensor VM Interpreter<br>(vm.js)"]
        
        MainThread <-->|"postMessage (Task Data)"| WebWorker
        WebWorker -->|"Executes DSL instructions"| VM
    end

    %% Communications
    Dashboard <-->|"WebSockets: Stats, Controls, Logs"| WSServer
    Dashboard --->|"HTTP POST: Configure Hyperparameters / Small Datasets"| ExpressServer
    
    WSServer <-->|"WebSockets: Tasks & Gradients"| Worker1
    WSServer <-->|"WebSockets: Tasks & Gradients"| Worker2
    WSServer <-->|"WebSockets: Tasks & Gradients"| WorkerN
    
    Worker1 <-->|"Internal Threading"| MainThread
    
    classDef server fill:#1e1e2f,stroke:#8b5cf6,stroke-width:2px,color:#fff;
    classDef worker fill:#0c0c14,stroke:#10b981,stroke-width:2px,color:#fff;
    classDef dash fill:#12121d,stroke:#60a5fa,stroke-width:2px,color:#fff;
    classDef thread fill:#1f1f2e,stroke:#f59e0b,stroke-width:1px,color:#fff;
    
    class CentralServer,ExpressServer,WSServer,Compiler,Trainer,Scheduler,DiskData server;
    class WorkerNodes,Worker1,Worker2,WorkerN worker;
    class ClientDash,Dashboard dash;
    class InternalWorker,MainThread,WebWorker,VM thread;
```

### Architectural Component Breakdown

#### 1. PyTorch-like Symbolic Compiler (`src/compiler/`)
* **Dynamic Computational Graph**: Builds a Directed Acyclic Graph (DAG) when the model layers (e.g. `Linear`, `MLP`, `SelfAttention`) are run symbolically.
* **Topological Automatic Differentiation**: Performs a topological sort on active nodes and traverses the DAG in reverse order to inject gradient accumulation instructions.
* **Auto-Broadcasting Reductions**: Automatically detects and inserts broadcasting shape correction nodes (via axis-summing and reshaping opcodes) during backpropagation to match broadcast shapes.
* **DSL Code Generator**: Emits a combined forward and backward assembly instruction script for the VM.

#### 2. Register-Based Tensor VM (`src/public/vm.js`)
* **Strided Layout Tensor**: Tensors store shapes and strides in a 1D Float32Array, enabling zero-copy $O(1)$ transposition and reshaping.
* **Mathematical Operations**: Custom implementations of `matmul` (2D and 3D batched), broadcasting operations (`add`, `sub`, `mul`, `div`), reduction operations (`sum`, `mean` along specified axes), activations (`relu`, `gelu`), softmax, and categorical cross-entropy.
* **Assembly Execution**: Parses and runs compiled DSL instruction sets sequentially, updating local registers.

#### 3. Asynchronous Orchestrator Scheduler (`src/server/orchestrator.ts`)
* **Batch Slicing**: Splits a global training batch (e.g., size 128) into smaller data-parallel slices (e.g., 4 tasks of size 32).
* **Scheduling Queue**: Matches tasks to idle workers and delivers them over WebSocket connection payloads.
* **Fault Tolerance & Heartbeats**:
  - Monitors worker heartbeats every 3 seconds. Workers failing to ping within 15 seconds are dropped.
  - Monitors task completion. If a worker does not return gradients within 40 seconds, the scheduler cancels the task, marks the worker as failed, and pushes the task back to the front of the queue to be processed by a healthy worker.

#### 4. Zero-RAM Dataset Seeker (`src/server/trainer.ts`)
* Prevents browser and server out-of-memory (OOM) crashes on large files (>1GB).
* Rather than loading text contents into memory, the server reads data directly from disk using low-level file descriptor seeks (`fs.readSync`) to fetch randomized training offsets on-demand. This uses $O(1)$ RAM regardless of dataset size.

#### 5. Client Web Workers (`src/public/worker_thread.js`)
* Offloads execution of the Tensor VM to a separate background thread (`Web Worker`). This isolates CPU-intensive matrix calculations, keeping the main browser UI thread fully responsive.

---

## 📜 DSL Instruction Reference

The compiled assembly language uses a simple text format. Lines beginning with `#` are comments. Each instruction specifies an operation followed by register outputs and inputs:

| Opcode | Arguments | Description |
| :--- | :--- | :--- |
| `matmul` | `out, in1, in2` | Performs matrix multiplication `out = in1 @ in2` (supports 2D and 3D batched inputs). |
| `transpose` | `out, in` | Transposes the last two dimensions of `in` in $O(1)$ time by swapping strides. |
| `add` | `out, in1, in2` | Element-wise addition with broadcasting support. |
| `sub` | `out, in1, in2` | Element-wise subtraction with broadcasting support. |
| `mul` | `out, in1, in2` | Element-wise multiplication with broadcasting support. |
| `div` | `out, in1, in2` | Element-wise division with broadcasting support. |
| `sum` | `out, in, axis` | Sums elements along the specified axis dimension. |
| `mean` | `out, in, axis` | Computes the mean of elements along the specified axis dimension. |
| `reshape` | `out, in, dim1, dim2...` | Reshapes `in` tensor to the target dimensions in $O(1)$ time. |
| `relu` | `out, in` | Applies the Rectified Linear Unit activation function. |
| `relu_grad` | `out, grad, in` | Computes the backpropagation gradient of the ReLU function. |
| `gelu` | `out, in` | Applies the Gaussian Error Linear Unit activation function. |
| `gelu_grad` | `out, grad, in` | Computes the backpropagation gradient of the GELU function. |
| `cross_entropy` | `loss, grad, logits, target` | Computes categorical cross-entropy loss and its symbolic gradient. |
| `assign` | `out, in` | Copies values and metadata from `in` to `out`. |

---

## ⚙️ Installation Guide

### Prerequisites
* **Node.js**: Version 18.0 or higher
* **npm**: Version 9.0 or higher (bundled with Node.js — installing Node.js installs npm automatically)

> **`zsh: command not found: npm`?** This means Node.js is not installed or is not on your `PATH`. Install Node.js (which includes npm) from [nodejs.org](https://nodejs.org/) or via a version manager such as [nvm](https://github.com/nvm-sh/nvm), then restart your terminal and verify with `node -v` and `npm -v`.

### 1. Server Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/ravikadam/distcompute.git
cd distcompute
npm install
```

### 2. Build the Project
Compile the TypeScript files and copy static frontend assets to the build folder:
```bash
npm run build
```

> **`sh: tsc: command not found`?** TypeScript is installed locally as a project dev dependency, not globally, so the bare `tsc` command is not on your `PATH`. Always run the build through npm (`npm run build`) or invoke the local binary with `npx tsc`. Run `npm install` first so the local `tsc` exists in `node_modules/.bin`. (Alternatively, install it globally with `npm install -g typescript`.)

### 3. Run Math and VM Tests
Verify the mathematical accuracy of the Tensor VM operations:
```bash
npm test
```

### 4. Verify Autodiff & Gradient Checking
Verify the compiler's symbolic backpropagation against numerical finite-difference gradients:
```bash
npx ts-node src/tests/test_compiler.ts
```

---

## 📱 Compute Worker Node Join Guide

Workers execute training computations in their browser sandbox. Setting up a compute node requires no software downloads or NPM installations—it is entirely client-side.

```
                    ┌────────────────────────┐
                    │   Central Orchestrator │
                    │   Server (192.168.1.5) │
                    └───────────▲────────────┘
                                │ (Port 3000)
              ┌─────────────────┴─────────────────┐
              │   Local Wi-Fi Router / LAN        │
              └─────────▲──────────────────▲──────┘
                        │                  │
               ┌────────┴──────┐    ┌──────┴─────────┐
               │ Mobile Phone  │    │ Laptop Browser │
               │ (Compute Node)│    │ (Compute Node) │
               └───────────────┘    └────────────────┘
```

### Step 1: Network Requirements
* Ensure that the server machine and the worker devices (phones, tablets, laptops) are connected to the **same local area network (Wi-Fi or LAN)**.
* Devices on guest networks or isolated APs will not be able to ping the server.

### Step 2: Access the Worker Interface
1. Run the server (see **Usage Guide** below). Note the `Worker Join URL` displayed in the terminal logs (e.g., `http://192.168.1.5:3000/worker.html`).
2. Open the browser (Safari, Chrome, Firefox) on the worker device and enter the `Worker Join URL`.
3. The page will load a dashboard and automatically attempt to establish a WebSocket connection.

### Step 3: Keep the Worker Node Active
* **Disable Screen Lock (Critical for Mobile)**:
  * **iOS**: Go to **Settings** ➔ **Display & Brightness** ➔ **Auto-Lock** ➔ Set to **Never**.
  * **Android**: Go to **Settings** ➔ **Display** ➔ **Screen Timeout** ➔ Set to maximum allowed (or enable "Stay Awake" in Developer Options while charging).
* **Foregrounding**: Keep the browser tab active and in the foreground. Mobile operating systems suspend WebSocket traffic and Web Workers when a tab is minimized or backgrounded.

### Step 4: Custom Connection (Optional)
If a node does not auto-connect:
1. Enter the server's WebSocket address manually in the **"Orchestrator WebSocket Endpoint"** field (e.g. `ws://192.168.1.5:3000/worker`).
2. Click **"Connect Node"**.
3. A successful connection is indicated by the green status light (**"Connected"**).

---

## 🚀 Usage Guide

### 1. Start the Server
Run the startup command:
```bash
npm start
```
The server binds to `0.0.0.0` (all interfaces) to allow network access. The startup log will display connection details:
```text
🚀 Distributed Compute Server listening on 0.0.0.0:3000
🖥️  Local Dashboard: http://localhost:3000/dashboard.html
🖥️  Network Dashboard: http://192.168.1.5:3000/dashboard.html
📱 Worker Join URL: http://192.168.1.5:3000/worker.html
```

### 2. Configure Training Parameters
1. Open the **Admin Dashboard** (`http://localhost:3000/dashboard.html`).
2. Adjust model and hyperparameter values:
   * **Learning Rate**: Step scale of Adam updates.
   * **Hidden Dimension**: Layer capacity of the MLP model.
   * **Context Length**: Number of characters the model reads to predict the next character.
   * **Batch Size**: Global training batch size.
3. Configure the training text corpus:
   * **Files under 20MB**: Drag-and-drop or select the file using the dashboard file uploader.
   * **Files over 1GB**: Place the text file on the server's local storage disk, copy its absolute file path, and paste it into the **"OR Server-Side Dataset File Path"** input field.
4. Click **"Apply Parameters & Reset Weights"**. The compiler will generate the model DSL and reset weights.

### 3. Run Training
1. Open the **Worker Join URL** on your mobile phones or other devices. Verify they show up in the **Active Compute Workers** table on the Admin Dashboard.
2. Click **"Start Training"** on the dashboard.
3. The dashboard will display live progress metrics:
   * **Loss Chart**: Real-time line graph plotting convergence.
   * **Global Throughput**: Examples processed per second.
   * **Sampled Prediction Stream**: Live text completions generated by the model.
4. To backup or export the trained model parameters, click the **"Download Weights"** button. This downloads a structured JSON file containing all weights, biases, and character mapping dictionaries.

---

## 🛠️ Troubleshooting Connection and Firewalls

If workers fail to connect or display network timeouts:

### 1. Firewall Blocks
By default, macOS and Windows block incoming TCP traffic on port `3000`.
* **macOS**: Go to **System Settings** ➔ **Network** ➔ **Firewall** ➔ Disable, or add an inbound exception rule for Node.js.
* **Windows**: Run the following in Administrator PowerShell to allow port 3000 traffic:
  ```powershell
  New-NetFirewallRule -DisplayName "DistCompute" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 3000
  ```

### 2. Ping Test
On a laptop worker connected to the same Wi-Fi, open the terminal and ping the server's LAN IP:
```bash
ping 192.168.1.5
```
If packets are lost, double-check that the devices are on the same Wi-Fi router subnet and that "Client Isolation" is disabled in the router settings.
