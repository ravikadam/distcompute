import { Compiler, TensorNode } from './compiler';

export abstract class Module {
  abstract forward(compiler: Compiler, ...args: any[]): TensorNode;
}

export class Linear extends Module {
  w: TensorNode;
  b: TensorNode;

  constructor(compiler: Compiler, name: string, inFeatures: number, outFeatures: number) {
    super();
    this.w = compiler.registerParameter(`${name}_w`, [inFeatures, outFeatures]);
    this.b = compiler.registerParameter(`${name}_b`, [outFeatures]);
  }

  forward(compiler: Compiler, x: TensorNode): TensorNode {
    const xw = compiler.matmul(x, this.w);
    return compiler.add(xw, this.b);
  }
}

export class MLP extends Module {
  fc1: Linear;
  fc2: Linear;

  constructor(compiler: Compiler, name: string, inDim: number, hDim: number, outDim: number) {
    super();
    this.fc1 = new Linear(compiler, `${name}_fc1`, inDim, hDim);
    this.fc2 = new Linear(compiler, `${name}_fc2`, hDim, outDim);
  }

  forward(compiler: Compiler, x: TensorNode): TensorNode {
    const h = compiler.gelu(this.fc1.forward(compiler, x));
    return this.fc2.forward(compiler, h);
  }
}

export class SelfAttention extends Module {
  wq: Linear;
  wk: Linear;
  wv: Linear;
  wo: Linear;
  scale: number;

  constructor(compiler: Compiler, name: string, dModel: number, dHead: number) {
    super();
    this.wq = new Linear(compiler, `${name}_wq`, dModel, dHead);
    this.wk = new Linear(compiler, `${name}_wk`, dModel, dHead);
    this.wv = new Linear(compiler, `${name}_wv`, dModel, dHead);
    this.wo = new Linear(compiler, `${name}_wo`, dHead, dModel);
    this.scale = 1.0 / Math.sqrt(dHead);
  }

  forward(compiler: Compiler, x: TensorNode): TensorNode {
    // x shape: [B, S, D]
    const Q = this.wq.forward(compiler, x); // [B, S, dHead]
    const K = this.wk.forward(compiler, x); // [B, S, dHead]
    const V = this.wv.forward(compiler, x); // [B, S, dHead]

    const KT = compiler.transpose(K, 1, 2); // [B, dHead, S]
    const QKT = compiler.matmul(Q, KT); // [B, S, S]
    const scaled = compiler.mul(QKT, this.scale);
    const attn = compiler.softmax(scaled); // [B, S, S]

    const out = compiler.matmul(attn, V); // [B, S, dHead]
    return this.wo.forward(compiler, out); // [B, S, dModel]
  }
}
