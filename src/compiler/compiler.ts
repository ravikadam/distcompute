// PyTorch-like Symbolic Autodiff Compiler

export class TensorNode {
  name: string;
  shape: number[];
  creatorOp: CreatorOp | null = null;
  requiresGrad: boolean = false;
  grad: TensorNode | null = null;

  constructor(name: string, shape: number[], requiresGrad = false) {
    this.name = name;
    this.shape = shape;
    this.requiresGrad = requiresGrad;
  }
}

export interface CreatorOp {
  op: string;
  inputs: TensorNode[];
  args?: any[];
}

export class Compiler {
  nodes: TensorNode[] = [];
  varCounter = 0;
  parameters: TensorNode[] = [];

  createNode(name: string, shape: number[], requiresGrad = false): TensorNode {
    const node = new TensorNode(name, shape, requiresGrad);
    this.nodes.push(node);
    return node;
  }

  createTempNode(shape: number[]): TensorNode {
    const name = `t${this.varCounter++}`;
    return this.createNode(name, shape);
  }

  registerParameter(name: string, shape: number[]): TensorNode {
    const p = this.createNode(name, shape, true);
    this.parameters.push(p);
    return p;
  }

  // Symbolic operations
  matmul(a: TensorNode, b: TensorNode): TensorNode {
    let outShape: number[];
    if (a.shape.length === 2 && b.shape.length === 2) {
      if (a.shape[1] !== b.shape[0]) {
        throw new Error(`Matmul shape mismatch: [${a.shape}] and [${b.shape}]`);
      }
      outShape = [a.shape[0], b.shape[1]];
    } else if (a.shape.length === 3 && b.shape.length === 3) {
      if (a.shape[0] !== b.shape[0] || a.shape[2] !== b.shape[1]) {
        throw new Error(`Matmul shape mismatch: [${a.shape}] and [${b.shape}]`);
      }
      outShape = [a.shape[0], a.shape[1], b.shape[2]];
    } else {
      throw new Error(`Unsupported shapes for matmul: [${a.shape}] and [${b.shape}]`);
    }

    const out = this.createTempNode(outShape);
    out.requiresGrad = a.requiresGrad || b.requiresGrad;
    out.creatorOp = { op: 'matmul', inputs: [a, b] };
    return out;
  }

  add(a: TensorNode, b: TensorNode): TensorNode {
    const outShape = getBroadcastShape(a.shape, b.shape);
    const out = this.createTempNode(outShape);
    out.requiresGrad = a.requiresGrad || b.requiresGrad;
    out.creatorOp = { op: 'add', inputs: [a, b] };
    return out;
  }

  sub(a: TensorNode, b: TensorNode): TensorNode {
    const outShape = getBroadcastShape(a.shape, b.shape);
    const out = this.createTempNode(outShape);
    out.requiresGrad = a.requiresGrad || b.requiresGrad;
    out.creatorOp = { op: 'sub', inputs: [a, b] };
    return out;
  }

  mul(a: TensorNode, b: TensorNode | number): TensorNode {
    if (typeof b === 'number') {
      const out = this.createTempNode(a.shape);
      out.requiresGrad = a.requiresGrad;
      out.creatorOp = { op: 'mul_scalar', inputs: [a], args: [b] };
      return out;
    } else {
      const outShape = getBroadcastShape(a.shape, b.shape);
      const out = this.createTempNode(outShape);
      out.requiresGrad = a.requiresGrad || b.requiresGrad;
      out.creatorOp = { op: 'mul', inputs: [a, b] };
      return out;
    }
  }

  div(a: TensorNode, b: TensorNode | number): TensorNode {
    if (typeof b === 'number') {
      const out = this.createTempNode(a.shape);
      out.requiresGrad = a.requiresGrad;
      out.creatorOp = { op: 'div_scalar', inputs: [a], args: [b] };
      return out;
    } else {
      const outShape = getBroadcastShape(a.shape, b.shape);
      const out = this.createTempNode(outShape);
      out.requiresGrad = a.requiresGrad || b.requiresGrad;
      out.creatorOp = { op: 'div', inputs: [a, b] };
      return out;
    }
  }

  relu(a: TensorNode): TensorNode {
    const out = this.createTempNode(a.shape);
    out.requiresGrad = a.requiresGrad;
    out.creatorOp = { op: 'relu', inputs: [a] };
    return out;
  }

  gelu(a: TensorNode): TensorNode {
    const out = this.createTempNode(a.shape);
    out.requiresGrad = a.requiresGrad;
    out.creatorOp = { op: 'gelu', inputs: [a] };
    return out;
  }

  softmax(a: TensorNode): TensorNode {
    const out = this.createTempNode(a.shape);
    out.requiresGrad = a.requiresGrad;
    out.creatorOp = { op: 'softmax', inputs: [a] };
    return out;
  }

  // Row-gather embedding lookup. table: [V, d], ids: [...] integer indices.
  // Output shape is ids.shape with the embedding dim appended.
  embedding(table: TensorNode, ids: TensorNode): TensorNode {
    const d = table.shape[table.shape.length - 1];
    const out = this.createTempNode([...ids.shape, d]);
    out.requiresGrad = table.requiresGrad;
    out.creatorOp = { op: 'embedding', inputs: [table, ids] };
    return out;
  }

  // LayerNorm over the last dimension. gamma/beta: [d].
  layernorm(x: TensorNode, gamma: TensorNode, beta: TensorNode): TensorNode {
    const out = this.createTempNode(x.shape);
    out.requiresGrad = x.requiresGrad || gamma.requiresGrad || beta.requiresGrad;
    out.creatorOp = { op: 'layernorm', inputs: [x, gamma, beta] };
    return out;
  }

  transpose(a: TensorNode, dim0 = -2, dim1 = -1): TensorNode {
    const ndim = a.shape.length;
    let d0 = dim0 < 0 ? dim0 + ndim : dim0;
    let d1 = dim1 < 0 ? dim1 + ndim : dim1;
    const outShape = [...a.shape];
    outShape[d0] = a.shape[d1];
    outShape[d1] = a.shape[d0];

    const out = this.createTempNode(outShape);
    out.requiresGrad = a.requiresGrad;
    out.creatorOp = { op: 'transpose', inputs: [a], args: [d0, d1] };
    return out;
  }

  reshape(a: TensorNode, newShape: number[]): TensorNode {
    const out = this.createTempNode(newShape);
    out.requiresGrad = a.requiresGrad;
    out.creatorOp = { op: 'reshape', inputs: [a], args: [newShape] };
    return out;
  }

  cross_entropy(logits: TensorNode, targets: TensorNode): { loss: TensorNode; gradLogits: TensorNode } {
    const loss = this.createTempNode([1]);
    const gradLogits = this.createNode(`g_${logits.name}`, logits.shape);
    
    loss.requiresGrad = logits.requiresGrad;
    // Cross entropy represents a joint node
    loss.creatorOp = { op: 'cross_entropy', inputs: [logits, targets], args: [gradLogits] };
    return { loss, gradLogits };
  }

  // Compile full forward and backward graph
  compile(lossNode: TensorNode, targetsNode: TensorNode): { dsl: string; shapes: Record<string, number[]> } {
    const forwardSorted = topologicalSort(lossNode);

    const forwardInstructions: string[] = [];
    const backwardInstructions: string[] = [];

    // 1. Emit Forward Instructions
    for (const node of forwardSorted) {
      if (!node.creatorOp) continue;
      const op = node.creatorOp.op;
      const inputs = node.creatorOp.inputs.map(n => n.name);
      
      if (op === 'matmul') {
        forwardInstructions.push(`matmul ${node.name}, ${inputs[0]}, ${inputs[1]}`);
      } else if (op === 'add') {
        forwardInstructions.push(`add ${node.name}, ${inputs[0]}, ${inputs[1]}`);
      } else if (op === 'sub') {
        forwardInstructions.push(`sub ${node.name}, ${inputs[0]}, ${inputs[1]}`);
      } else if (op === 'mul') {
        forwardInstructions.push(`mul ${node.name}, ${inputs[0]}, ${inputs[1]}`);
      } else if (op === 'mul_scalar') {
        forwardInstructions.push(`mul ${node.name}, ${inputs[0]}, ${node.creatorOp.args![0]}`);
      } else if (op === 'div') {
        forwardInstructions.push(`div ${node.name}, ${inputs[0]}, ${inputs[1]}`);
      } else if (op === 'div_scalar') {
        forwardInstructions.push(`div ${node.name}, ${inputs[0]}, ${node.creatorOp.args![0]}`);
      } else if (op === 'relu') {
        forwardInstructions.push(`relu ${node.name}, ${inputs[0]}`);
      } else if (op === 'gelu') {
        forwardInstructions.push(`gelu ${node.name}, ${inputs[0]}`);
      } else if (op === 'softmax') {
        forwardInstructions.push(`softmax ${node.name}, ${inputs[0]}`);
      } else if (op === 'embedding') {
        forwardInstructions.push(`embedding ${node.name}, ${inputs[0]}, ${inputs[1]}`);
      } else if (op === 'layernorm') {
        forwardInstructions.push(`layernorm ${node.name}, ${inputs[0]}, ${inputs[1]}, ${inputs[2]}`);
      } else if (op === 'transpose') {
        forwardInstructions.push(`transpose ${node.name}, ${inputs[0]}, ${node.creatorOp.args![0]}, ${node.creatorOp.args![1]}`);
      } else if (op === 'reshape') {
        forwardInstructions.push(`reshape ${node.name}, ${inputs[0]}, ${node.creatorOp.args![0].join(",")}`);
      } else if (op === 'cross_entropy') {
        const gradLogits = node.creatorOp.args![0] as TensorNode;
        forwardInstructions.push(`cross_entropy ${node.name}, ${gradLogits.name}, ${inputs[0]}, ${inputs[1]}`);
      }
    }

    // 2. Perform Backward Pass
    // We traverse forwardSorted in reverse order
    const gradMap = new Map<string, TensorNode>();

    // Root grad initialization
    // For cross_entropy, the logits gradient is already computed in forward pass
    // So we initialize the logits' gradient with the pre-computed gradLogits node!
    for (let i = forwardSorted.length - 1; i >= 0; i--) {
      const node = forwardSorted[i];
      if (node.creatorOp?.op === 'cross_entropy') {
        const logits = node.creatorOp.inputs[0];
        const gradLogits = node.creatorOp.args![0] as TensorNode;
        logits.grad = gradLogits;
      }
    }

    // Traverse in reverse topological order to propagate gradients
    for (let i = forwardSorted.length - 1; i >= 0; i--) {
      const node = forwardSorted[i];
      if (!node.grad || !node.creatorOp) continue;

      const op = node.creatorOp.op;
      const inputs = node.creatorOp.inputs;

      if (op === 'matmul') {
        const A = inputs[0];
        const B = inputs[1];
        // Transposing the LAST TWO dims works for both 2D and 3D (batched)
        // matmul — the previous code assumed 2D and broke inside attention.
        const swapLast2 = (s: number[]) => {
          const r = [...s];
          const n = r.length;
          [r[n - 2], r[n - 1]] = [r[n - 1], r[n - 2]];
          return r;
        };

        // dA = dC @ B^T
        if (A.requiresGrad) {
          const BT = this.createTempNode(swapLast2(B.shape));
          const dA = this.createTempNode(A.shape);
          backwardInstructions.push(`transpose ${BT.name}, ${B.name}`);
          backwardInstructions.push(`matmul ${dA.name}, ${node.grad.name}, ${BT.name}`);
          emitAccumulateGrad(this, A, dA, backwardInstructions);
        }

        // dB = A^T @ dC
        if (B.requiresGrad) {
          const AT = this.createTempNode(swapLast2(A.shape));
          const dB = this.createTempNode(B.shape);
          backwardInstructions.push(`transpose ${AT.name}, ${A.name}`);
          backwardInstructions.push(`matmul ${dB.name}, ${AT.name}, ${node.grad.name}`);
          emitAccumulateGrad(this, B, dB, backwardInstructions);
        }
      } else if (op === 'add') {
        const A = inputs[0];
        const B = inputs[1];
        if (A.requiresGrad) emitAccumulateGrad(this, A, node.grad, backwardInstructions);
        if (B.requiresGrad) emitAccumulateGrad(this, B, node.grad, backwardInstructions);
      } else if (op === 'sub') {
        const A = inputs[0];
        const B = inputs[1];
        if (A.requiresGrad) emitAccumulateGrad(this, A, node.grad, backwardInstructions);
        if (B.requiresGrad) {
          const negGrad = this.createTempNode(node.grad.shape);
          backwardInstructions.push(`mul ${negGrad.name}, ${node.grad.name}, -1`);
          emitAccumulateGrad(this, B, negGrad, backwardInstructions);
        }
      } else if (op === 'mul') {
        const A = inputs[0];
        const B = inputs[1];
        if (A.requiresGrad) {
          const dA = this.createTempNode(A.shape);
          backwardInstructions.push(`mul ${dA.name}, ${node.grad.name}, ${B.name}`);
          emitAccumulateGrad(this, A, dA, backwardInstructions);
        }
        if (B.requiresGrad) {
          const dB = this.createTempNode(B.shape);
          backwardInstructions.push(`mul ${dB.name}, ${node.grad.name}, ${A.name}`);
          emitAccumulateGrad(this, B, dB, backwardInstructions);
        }
      } else if (op === 'mul_scalar') {
        const A = inputs[0];
        const scalar = node.creatorOp.args![0];
        if (A.requiresGrad) {
          const dA = this.createTempNode(A.shape);
          backwardInstructions.push(`mul ${dA.name}, ${node.grad.name}, ${scalar}`);
          emitAccumulateGrad(this, A, dA, backwardInstructions);
        }
      } else if (op === 'div') {
        const A = inputs[0];
        const B = inputs[1];
        if (A.requiresGrad) {
          const dA = this.createTempNode(A.shape);
          backwardInstructions.push(`div ${dA.name}, ${node.grad.name}, ${B.name}`);
          emitAccumulateGrad(this, A, dA, backwardInstructions);
        }
        if (B.requiresGrad) {
          // dL/dB = -dL/dC * A / B^2
          const B2 = this.createTempNode(B.shape);
          const divTerm = this.createTempNode(A.shape);
          const negDivTerm = this.createTempNode(A.shape);
          const dB = this.createTempNode(B.shape);
          backwardInstructions.push(`mul ${B2.name}, ${B.name}, ${B.name}`);
          backwardInstructions.push(`div ${divTerm.name}, ${A.name}, ${B2.name}`);
          backwardInstructions.push(`mul ${negDivTerm.name}, ${divTerm.name}, -1`);
          backwardInstructions.push(`mul ${dB.name}, ${node.grad.name}, ${negDivTerm.name}`);
          emitAccumulateGrad(this, B, dB, backwardInstructions);
        }
      } else if (op === 'div_scalar') {
        const A = inputs[0];
        const scalar = node.creatorOp.args![0];
        if (A.requiresGrad) {
          const dA = this.createTempNode(A.shape);
          backwardInstructions.push(`div ${dA.name}, ${node.grad.name}, ${scalar}`);
          emitAccumulateGrad(this, A, dA, backwardInstructions);
        }
      } else if (op === 'relu') {
        const A = inputs[0];
        if (A.requiresGrad) {
          const dA = this.createTempNode(A.shape);
          backwardInstructions.push(`relu_grad ${dA.name}, ${node.grad.name}, ${node.name}`);
          emitAccumulateGrad(this, A, dA, backwardInstructions);
        }
      } else if (op === 'gelu') {
        const A = inputs[0];
        if (A.requiresGrad) {
          const dA = this.createTempNode(A.shape);
          backwardInstructions.push(`gelu_grad ${dA.name}, ${node.grad.name}, ${A.name}`);
          emitAccumulateGrad(this, A, dA, backwardInstructions);
        }
      } else if (op === 'softmax') {
        const A = inputs[0];
        if (A.requiresGrad) {
          const dA = this.createTempNode(A.shape);
          backwardInstructions.push(`softmax_grad ${dA.name}, ${node.grad.name}, ${node.name}`);
          emitAccumulateGrad(this, A, dA, backwardInstructions);
        }
      } else if (op === 'embedding') {
        // Only the table receives a gradient (ids are integer indices).
        const table = inputs[0];
        const ids = inputs[1];
        if (table.requiresGrad) {
          const dTable = this.createTempNode(table.shape);
          backwardInstructions.push(`embedding_grad ${dTable.name}, ${node.grad.name}, ${ids.name}`);
          emitAccumulateGrad(this, table, dTable, backwardInstructions);
        }
      } else if (op === 'layernorm') {
        // Fused backward emits dx, dgamma, dbeta in one instruction.
        const x = inputs[0];
        const gamma = inputs[1];
        const beta = inputs[2];
        const dX = this.createTempNode(x.shape);
        const dGamma = this.createTempNode(gamma.shape);
        const dBeta = this.createTempNode(beta.shape);
        backwardInstructions.push(
          `layernorm_grad ${dX.name}, ${dGamma.name}, ${dBeta.name}, ${node.grad.name}, ${x.name}, ${gamma.name}`
        );
        if (x.requiresGrad) emitAccumulateGrad(this, x, dX, backwardInstructions);
        if (gamma.requiresGrad) emitAccumulateGrad(this, gamma, dGamma, backwardInstructions);
        if (beta.requiresGrad) emitAccumulateGrad(this, beta, dBeta, backwardInstructions);
      } else if (op === 'transpose') {
        const A = inputs[0];
        const dim0 = node.creatorOp.args![0];
        const dim1 = node.creatorOp.args![1];
        if (A.requiresGrad) {
          const dA = this.createTempNode(A.shape);
          backwardInstructions.push(`transpose ${dA.name}, ${node.grad.name}, ${dim0}, ${dim1}`);
          emitAccumulateGrad(this, A, dA, backwardInstructions);
        }
      } else if (op === 'reshape') {
        const A = inputs[0];
        if (A.requiresGrad) {
          const dA = this.createTempNode(A.shape);
          backwardInstructions.push(`reshape ${dA.name}, ${node.grad.name}, ${A.shape.join(",")}`);
          emitAccumulateGrad(this, A, dA, backwardInstructions);
        }
      }
    }

    const dsl = [
      "# === FORWARD PASS ===",
      ...forwardInstructions,
      "",
      "# === BACKWARD PASS ===",
      ...backwardInstructions
    ].join("\n");

    const shapes: Record<string, number[]> = {};
    for (const node of this.nodes) {
      shapes[node.name] = node.shape;
      if (node.grad) {
        shapes[node.grad.name] = node.grad.shape;
      }
    }

    return { dsl, shapes };
  }
}

// Helpers for topological sorting
function topologicalSort(root: TensorNode): TensorNode[] {
  const visited = new Set<TensorNode>();
  const sorted: TensorNode[] = [];

  function visit(node: TensorNode) {
    if (visited.has(node)) return;
    visited.add(node);
    if (node.creatorOp) {
      for (const input of node.creatorOp.inputs) {
        visit(input);
      }
    }
    sorted.push(node);
  }

  visit(root);
  return sorted;
}

// Broadcasting shape helper
function getBroadcastShape(shapeA: number[], shapeB: number[]): number[] {
  const ndim = Math.max(shapeA.length, shapeB.length);
  const outShape = new Array(ndim);

  for (let i = 0; i < ndim; i++) {
    const aDim = shapeA[shapeA.length - 1 - i] ?? 1;
    const bDim = shapeB[shapeB.length - 1 - i] ?? 1;

    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error(`Broadcasting shape mismatch: [${shapeA}] and [${shapeB}]`);
    }
    outShape[ndim - 1 - i] = Math.max(aDim, bDim);
  }

  return outShape;
}

// Accumulate gradient helper
function emitAccumulateGrad(
  compiler: Compiler,
  target: TensorNode,
  contrib: TensorNode,
  instructions: string[]
) {
  if (!target.requiresGrad) return;

  const targetShape = target.shape;
  const contribShape = contrib.shape;

  const ndim = contribShape.length;
  const targetAligned = new Array(ndim).fill(1);
  const diff = ndim - targetShape.length;
  for (let i = 0; i < targetShape.length; i++) {
    targetAligned[i + diff] = targetShape[i];
  }

  let currentGradNode = contrib;

  for (let d = 0; d < ndim; d++) {
    if (targetAligned[d] === 1 && contribShape[d] > 1) {
      const nextShape = [...currentGradNode.shape];
      nextShape[d] = 1;
      const reducedNode = compiler.createTempNode(nextShape);
      instructions.push(`sum ${reducedNode.name}, ${currentGradNode.name}, ${d}`);
      currentGradNode = reducedNode;
    }
  }

  if (currentGradNode.shape.length !== targetShape.length) {
    const reshapedNode = compiler.createTempNode(targetShape);
    instructions.push(`reshape ${reshapedNode.name}, ${currentGradNode.name}, ${targetShape.join(",")}`);
    currentGradNode = reshapedNode;
  }

  if (!target.grad) {
    target.grad = compiler.createNode(`g_${target.name}`, targetShape);
    instructions.push(`assign ${target.grad.name}, ${currentGradNode.name}`);
  } else {
    instructions.push(`add ${target.grad.name}, ${target.grad.name}, ${currentGradNode.name}`);
  }
}
