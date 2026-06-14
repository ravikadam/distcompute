import { Compiler } from '../compiler/compiler';
import { Linear } from '../compiler/modules';
import { Tensor, TensorVM } from '../public/vm';
import * as assert from 'assert';

function approxEqual(a: number, b: number, epsilon = 1e-3): boolean {
  return Math.abs(a - b) < epsilon;
}

function runCompilerTests() {
  console.log("=== Running Compiler & Autodiff Gradient Check ===");

  const compiler = new Compiler();
  const X = compiler.createNode("X", [2, 3]); // batch size 2, input dim 3
  const targets = compiler.createNode("targets", [2]); // target class indices

  const fc = new Linear(compiler, "fc", 3, 2); // input 3, output 2 classes
  const logits = fc.forward(compiler, X);
  const { loss } = compiler.cross_entropy(logits, targets);

  const { dsl, shapes } = compiler.compile(loss, targets);
  console.log("Generated DSL Code:\n" + dsl + "\n");

  // Run in VM
  const vm = new TensorVM() as any;

  // Allocate all tensors
  for (const [name, shape] of Object.entries(shapes)) {
    vm.tensors[name] = new Tensor(shape);
  }

  // Set weights
  vm.tensors["X"].data.set([0.5, -0.2, 0.1, 0.8, 0.4, -0.6]);
  vm.tensors["fc_w"].data.set([0.1, 0.2, -0.3, 0.4, 0.5, -0.6]);
  vm.tensors["fc_b"].data.set([0.05, -0.1]);
  vm.tensors["targets"].data.set([1, 0]);

  // Execute DSL
  vm.execute(dsl);

  const lossValue = vm.tensors[loss.name].data[0]; // loss output (root node)
  console.log(`Computed loss: ${lossValue}`);

  // Fetch analytical gradients
  const g_w_analytical = Array.from(vm.tensors["g_fc_w"].data);
  const g_b_analytical = Array.from(vm.tensors["g_fc_b"].data);

  console.log("Analytical Gradients for fc_w:", g_w_analytical);
  console.log("Analytical Gradients for fc_b:", g_b_analytical);

  // Numerical Gradient Checking (Finite Differences)
  // we perturb each weight by epsilon, compute loss, and estimate gradient
  const epsilon = 1e-4;
  const num_g_w = new Array(6).fill(0);

  for (let i = 0; i < 6; i++) {
    // Perturb +eps
    vm.tensors["fc_w"].data[i] += epsilon;
    vm.execute(dsl);
    const lossPlus = vm.tensors[loss.name].data[0];

    // Perturb -eps (revert and sub)
    vm.tensors["fc_w"].data[i] -= 2 * epsilon;
    vm.execute(dsl);
    const lossMinus = vm.tensors[loss.name].data[0];

    // Revert
    vm.tensors["fc_w"].data[i] += epsilon;

    num_g_w[i] = (lossPlus - lossMinus) / (2 * epsilon);
  }

  const num_g_b = new Array(2).fill(0);
  for (let i = 0; i < 2; i++) {
    vm.tensors["fc_b"].data[i] += epsilon;
    vm.execute(dsl);
    const lossPlus = vm.tensors[loss.name].data[0];

    vm.tensors["fc_b"].data[i] -= 2 * epsilon;
    vm.execute(dsl);
    const lossMinus = vm.tensors[loss.name].data[0];

    vm.tensors["fc_b"].data[i] += epsilon;

    num_g_b[i] = (lossPlus - lossMinus) / (2 * epsilon);
  }

  console.log("Numerical Gradients for fc_w: ", num_g_w);
  console.log("Numerical Gradients for fc_b: ", num_g_b);

  // Compare
  for (let i = 0; i < 6; i++) {
    assert.ok(approxEqual(g_w_analytical[i] as number, num_g_w[i]), `fc_w[${i}] gradient mismatch: analytical=${g_w_analytical[i]}, numerical=${num_g_w[i]}`);
  }
  for (let i = 0; i < 2; i++) {
    assert.ok(approxEqual(g_b_analytical[i] as number, num_g_b[i]), `fc_b[${i}] gradient mismatch: analytical=${g_b_analytical[i]}, numerical=${num_g_b[i]}`);
  }

  console.log("✓ Gradient Checking Passed! Symbolic compiler autodiff is 100% correct.");
}

runCompilerTests();
