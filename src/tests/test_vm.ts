import { Tensor, TensorVM } from '../public/vm';
import * as assert from 'assert';

function approxEqual(a: number, b: number, epsilon = 1e-4): boolean {
  return Math.abs(a - b) < epsilon;
}

function runTests() {
  console.log("=== Running Tensor VM Tests ===");

  // 1. Basic Tensor Creation & Strides
  {
    const t = new Tensor([2, 3]);
    assert.strictEqual(t.size, 6);
    assert.deepStrictEqual(t.strides, [3, 1]);
    console.log("✓ Basic Tensor Creation & Strides");
  }

  // 2. 2D Matmul Test
  {
    const a = new Tensor([2, 3]);
    a.data.set([1, 2, 3, 4, 5, 6]);
    const b = new Tensor([3, 2]);
    b.data.set([7, 8, 9, 10, 11, 12]);
    const out = new Tensor([2, 2]);

    const { matmul } = require('../public/vm');
    matmul(out, a, b);

    // Expected:
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
    // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
    assert.ok(approxEqual(out.data[0], 58));
    assert.ok(approxEqual(out.data[1], 64));
    assert.ok(approxEqual(out.data[2], 139));
    assert.ok(approxEqual(out.data[3], 154));
    console.log("✓ 2D Matmul");
  }

  // 3. Broadcasting Add Test
  {
    const a = new Tensor([2, 2]);
    a.data.set([1, 2, 3, 4]);
    const b = new Tensor([2]); // should broadcast to [2, 2]
    b.data.set([10, 20]);
    const out = new Tensor([2, 2]);

    const { broadcastOp } = require('../public/vm');
    broadcastOp(out, a, b, 'add');

    // Expected:
    // [1 + 10, 2 + 20] = [11, 22]
    // [3 + 10, 4 + 20] = [13, 24]
    assert.deepStrictEqual(Array.from(out.data), [11, 22, 13, 24]);
    console.log("✓ Broadcasting Add");
  }

  // 4. Scalar Operations Test
  {
    const a = new Tensor([2, 2]);
    a.data.set([1, 2, 3, 4]);
    const out = new Tensor([2, 2]);

    const { scalarOp } = require('../public/vm');
    scalarOp(out, a, 10, 'mul');

    assert.deepStrictEqual(Array.from(out.data), [10, 20, 30, 40]);
    console.log("✓ Scalar Operations");
  }

  // 5. Reduction Sum Test
  {
    const a = new Tensor([2, 3]);
    a.data.set([1, 2, 3, 4, 5, 6]);
    const out = new Tensor([3]); // sum along axis 0

    const { reduceSum } = require('../public/vm');
    reduceSum(out, a, 0);

    // Expected: [1+4, 2+5, 3+6] = [5, 7, 9]
    assert.deepStrictEqual(Array.from(out.data), [5, 7, 9]);
    console.log("✓ Reduction Sum (axis 0)");
  }

  // 6. Cross Entropy Loss & Grad Test
  {
    // B=2, C=3
    const logits = new Tensor([2, 3]);
    logits.data.set([1, 2, 0, 0, 5, 2]);
    const targets = new Tensor([2]);
    targets.data.set([1, 1]); // target classes are 1 (val 2) and 1 (val 5)
    const loss = new Tensor([1]);
    const grad = new Tensor([2, 3]);

    const { crossEntropy } = require('../public/vm');
    crossEntropy(loss, grad, logits, targets);

    // Batch 0: logits [1, 2, 0]
    // max=2. exps = [exp(-1), exp(0), exp(-2)] = [0.367879, 1.0, 0.135335]. sum = 1.503214
    // logSumExp = 2 + log(1.503214) = 2.4076059
    // target logit = 2. Loss = 2.4076059 - 2 = 0.4076059
    // probs = [0.244728, 0.66524, 0.09002]
    // grads = [(0.244728)/2, (0.66524 - 1)/2, (0.09002)/2] = [0.122364, -0.16738, 0.04501]

    // Batch 1: logits [0, 5, 2]
    // max=5. exps = [exp(-5), exp(0), exp(-3)] = [0.0067379, 1.0, 0.049787]. sum = 1.056525
    // logSumExp = 5 + log(1.056525) = 5.054992
    // target logit = 5. Loss = 5.054992 - 5 = 0.054992
    // probs = [0.006377, 0.9465, 0.04712]
    // grads = [0.006377/2, (0.9465 - 1)/2, 0.04712/2] = [0.003188, -0.02675, 0.02356]

    // Total Loss = (0.4076059 + 0.054992)/2 = 0.231299
    assert.ok(approxEqual(loss.data[0], 0.231299, 1e-4));
    assert.ok(approxEqual(grad.data[1], -0.16738, 1e-3));
    assert.ok(approxEqual(grad.data[4], -0.02675, 1e-3));
    console.log("✓ Cross Entropy Loss & Gradients");
  }

  // 7. DSL Parser and Execution Test
  {
    const vm = new TensorVM() as any;
    vm.tensors["X"] = new Tensor([2, 2]);
    vm.tensors["X"].data.set([1, 2, 3, 4]);
    vm.tensors["W"] = new Tensor([2, 2]);
    vm.tensors["W"].data.set([2, 0, 0, 2]);
    vm.tensors["out"] = new Tensor([2, 2]);

    const script = `
      // simple matmul
      matmul out, X, W
      # scalar multiply
      mul out, out, 2.5
    `;

    vm.execute(script);

    // X * W = [2, 4, 6, 8]
    // (X * W) * 2.5 = [5, 10, 15, 20]
    assert.deepStrictEqual(Array.from(vm.tensors["out"].data), [5, 10, 15, 20]);
    console.log("✓ DSL Parser & Executor");
  }

  console.log("All tests passed successfully!");
}

runTests();
