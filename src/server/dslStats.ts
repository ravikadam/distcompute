/**
 * Estimate the compute cost of a compiled DSL program.
 *
 * "How much work is left" needs a unit of work. We use FLOPs, dominated by the
 * matmul instructions: for `matmul out, a, b` the cost is 2 · |out| · K where
 * |out| is the number of output elements and K is the shared inner dimension
 * (the last dim of `a`). Element-wise ops contribute |out| and are included for
 * completeness, but matmuls dominate. The DSL already contains both the forward
 * and backward pass, so this is the full per-task cost.
 */

function prod(shape: number[] | undefined): number {
  if (!shape) return 0;
  return shape.reduce((a, b) => a * b, 1);
}

export interface DslCost {
  flops: number;
  matmulFlops: number;
  matmulCount: number;
  instructionCount: number;
}

export function estimateDslFlops(dsl: string, shapes: Record<string, number[]>): DslCost {
  let flops = 0;
  let matmulFlops = 0;
  let matmulCount = 0;
  let instructionCount = 0;

  for (const raw of dsl.split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#') || line.startsWith('//')) continue;
    instructionCount++;
    const parts = line.split(/[\s,]+/).filter(Boolean);
    const op = parts[0];
    const out = parts[1];
    const outSize = prod(shapes[out]);

    if (op === 'matmul') {
      const a = parts[2];
      const aShape = shapes[a];
      const K = aShape ? aShape[aShape.length - 1] : 0;
      const f = 2 * outSize * K;
      matmulFlops += f;
      matmulCount++;
      flops += f;
    } else {
      // Element-wise / reductions / activations: ~one op per output element.
      flops += outSize;
    }
  }

  return { flops, matmulFlops, matmulCount, instructionCount };
}
