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

// ===========================================================================
// GPT building blocks. The residual stream is carried as a 2D [N, D] tensor
// (N = B*S), which keeps all the token-wise ops (LayerNorm, Linear, MLP) on the
// VM's 2D matmul path. Attention temporarily reshapes to multi-head form.
// ===========================================================================

export class LayerNorm extends Module {
  gamma: TensorNode;
  beta: TensorNode;
  constructor(compiler: Compiler, name: string, d: number) {
    super();
    this.gamma = compiler.registerParameter(`${name}_g`, [d]);
    this.beta = compiler.registerParameter(`${name}_b`, [d]);
  }
  forward(compiler: Compiler, x: TensorNode): TensorNode {
    return compiler.layernorm(x, this.gamma, this.beta);
  }
}

export class Embedding extends Module {
  table: TensorNode;
  constructor(compiler: Compiler, name: string, numEmbeddings: number, d: number) {
    super();
    this.table = compiler.registerParameter(name, [numEmbeddings, d]);
  }
  forward(compiler: Compiler, ids: TensorNode): TensorNode {
    return compiler.embedding(this.table, ids);
  }
}

export class CausalSelfAttention extends Module {
  wq: Linear; wk: Linear; wv: Linear; wo: Linear;
  nHead: number; dModel: number; dHead: number; scale: number;
  constructor(compiler: Compiler, name: string, dModel: number, nHead: number) {
    super();
    this.wq = new Linear(compiler, `${name}_wq`, dModel, dModel);
    this.wk = new Linear(compiler, `${name}_wk`, dModel, dModel);
    this.wv = new Linear(compiler, `${name}_wv`, dModel, dModel);
    this.wo = new Linear(compiler, `${name}_wo`, dModel, dModel);
    this.nHead = nHead;
    this.dModel = dModel;
    this.dHead = dModel / nHead;
    this.scale = 1.0 / Math.sqrt(this.dHead);
  }
  // x: [N=B*S, D]; mask: [1, S, S] additive causal mask.
  forward(compiler: Compiler, x: TensorNode, B: number, S: number, mask: TensorNode): TensorNode {
    const H = this.nHead, dh = this.dHead, D = this.dModel;
    // [N,D] -> [B,S,H,dh] -> [B,H,S,dh] -> [B*H,S,dh]
    const split = (t: TensorNode) => {
      const a = compiler.reshape(t, [B, S, H, dh]);
      const b = compiler.transpose(a, 1, 2);
      return compiler.reshape(b, [B * H, S, dh]);
    };
    const Qh = split(this.wq.forward(compiler, x));
    const Kh = split(this.wk.forward(compiler, x));
    const Vh = split(this.wv.forward(compiler, x));

    const Kt = compiler.transpose(Kh, -2, -1);          // [B*H, dh, S]
    let scores = compiler.matmul(Qh, Kt);               // [B*H, S, S]
    scores = compiler.mul(scores, this.scale);
    scores = compiler.add(scores, mask);                // causal mask (broadcast)
    const attn = compiler.softmax(scores);              // row-softmax over keys
    const ctx = compiler.matmul(attn, Vh);              // [B*H, S, dh]

    // [B*H,S,dh] -> [B,H,S,dh] -> [B,S,H,dh] -> [N,D]
    const merged = compiler.reshape(
      compiler.transpose(compiler.reshape(ctx, [B, H, S, dh]), 1, 2),
      [B * S, D]
    );
    return this.wo.forward(compiler, merged);
  }
}

export class TransformerBlock extends Module {
  ln1: LayerNorm; attn: CausalSelfAttention; ln2: LayerNorm; mlpFc: Linear; mlpProj: Linear;
  constructor(compiler: Compiler, name: string, dModel: number, nHead: number, dFF: number) {
    super();
    this.ln1 = new LayerNorm(compiler, `${name}_ln1`, dModel);
    this.attn = new CausalSelfAttention(compiler, `${name}_attn`, dModel, nHead);
    this.ln2 = new LayerNorm(compiler, `${name}_ln2`, dModel);
    this.mlpFc = new Linear(compiler, `${name}_mlp_fc`, dModel, dFF);
    this.mlpProj = new Linear(compiler, `${name}_mlp_proj`, dFF, dModel);
  }
  forward(compiler: Compiler, x: TensorNode, B: number, S: number, mask: TensorNode): TensorNode {
    // Pre-norm residual transformer block (GPT style).
    const a = compiler.add(x, this.attn.forward(compiler, this.ln1.forward(compiler, x), B, S, mask));
    const h = compiler.gelu(this.mlpFc.forward(compiler, this.ln2.forward(compiler, a)));
    return compiler.add(a, this.mlpProj.forward(compiler, h));
  }
}

export interface GPTConfig { vocab: number; context: number; dModel: number; nHead: number; dFF: number; nLayer: number; }

export class GPT extends Module {
  wte: Embedding; wpe: Embedding; blocks: TransformerBlock[]; lnf: LayerNorm; cfg: GPTConfig;
  constructor(compiler: Compiler, cfg: GPTConfig) {
    super();
    this.cfg = cfg;
    this.wte = new Embedding(compiler, 'wte', cfg.vocab, cfg.dModel);
    this.wpe = new Embedding(compiler, 'wpe', cfg.context, cfg.dModel);
    this.blocks = [];
    for (let i = 0; i < cfg.nLayer; i++) {
      this.blocks.push(new TransformerBlock(compiler, `h${i}`, cfg.dModel, cfg.nHead, cfg.dFF));
    }
    this.lnf = new LayerNorm(compiler, 'ln_f', cfg.dModel);
  }
  // tokenIds, posIds: flat [N=B*S]; mask: [1,S,S]. Returns logits [N, vocab]
  // using a weight-tied LM head (logits = x @ Wte^T).
  forward(compiler: Compiler, tokenIds: TensorNode, posIds: TensorNode, mask: TensorNode, B: number, S: number): TensorNode {
    let x = compiler.add(this.wte.forward(compiler, tokenIds), this.wpe.forward(compiler, posIds));
    for (const blk of this.blocks) x = blk.forward(compiler, x, B, S, mask);
    x = this.lnf.forward(compiler, x);
    const WteT = compiler.transpose(this.wte.table, -2, -1); // [D, V]
    return compiler.matmul(x, WteT);                          // [N, V]
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
