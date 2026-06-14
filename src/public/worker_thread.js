// Web Worker Thread for Tensor VM Execution
importScripts('vm.js');

self.onmessage = function (e) {
  const payload = e.data;

  if (payload.type === 'execute_task') {
    const { taskId, dsl, shapes, inputs } = payload;
    
    try {
      const vm = new self.TensorVM();

      // 1. Allocate all tensors of required shapes
      for (const [name, shape] of Object.entries(shapes)) {
        vm.tensors[name] = new self.Tensor(shape);
      }

      // 2. Load input values
      for (const [name, input] of Object.entries(inputs)) {
        if (!vm.tensors[name]) {
          throw new Error(`Input tensor "${name}" was not allocated in shapes dictionary`);
        }
        const flatData = self.TensorVM.base64ToFloat32Array(input.data);
        vm.tensors[name].data.set(flatData);
      }

      // 3. Execute DSL script and measure time
      const start = performance.now();
      vm.execute(dsl);
      const end = performance.now();
      const durationMs = end - start;

      // 4. Calculate compute throughput metric
      // sum up active elements of outputs to measure work done
      let totalElementsProcessed = 0;
      for (const t of Object.values(vm.tensors)) {
        totalElementsProcessed += t.size;
      }
      const durationSec = durationMs / 1000;
      const throughput = durationSec > 0 ? (totalElementsProcessed / durationSec) : 0;

      // 5. Gather only outputs (gradients starting with "g_" and the scalar loss starting with "t")
      const outputs = {};
      for (const name of Object.keys(vm.tensors)) {
        const isGrad = name.startsWith('g_');
        const isLoss = name.startsWith('t') && vm.tensors[name].size === 1; // loss node

        if (isGrad || isLoss) {
          const t = vm.tensors[name];
          // Ensure we copy the data from any offset
          const contiguousTensor = t.contiguous();
          outputs[name] = {
            shape: t.shape,
            data: self.TensorVM.float32ArrayToBase64(contiguousTensor.data)
          };
        }
      }

      // Send results back to main thread
      self.postMessage({
        type: 'task_completed',
        taskId,
        outputs,
        durationMs,
        throughput
      });

    } catch (err) {
      self.postMessage({
        type: 'task_failed',
        taskId,
        error: err.message
      });
    }
  }
};
