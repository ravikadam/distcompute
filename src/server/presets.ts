/**
 * Model + training presets.
 *
 * GPT1_CANONICAL is the original GPT-1 configuration from Radford et al. 2018
 * ("Improving Language Understanding by Generative Pre-Training"). GPT1_FEASIBLE
 * is a heavily scaled-down variant whose per-task tensors actually fit in a
 * browser worker (see the dashboard memory estimator for why the full model
 * does not — attention is B·H·S² and logits are B·S·V).
 */

export interface TransformerConfig {
  name: string;
  // Architecture
  nLayer: number;
  nHead: number;
  dModel: number;
  dFF: number;          // feed-forward inner dimension (usually 4 * dModel)
  vocab: number;
  context: number;      // max sequence length / position embeddings
  dropout: number;
  activation: string;
  initStd: number;      // weight init std (normal)
  tokenizer: string;
  // Training
  optimizer: string;
  maxLR: number;
  warmupSteps: number;
  lrSchedule: string;
  weightDecay: number;
  batchSequences: number;  // sequences per optimizer step (global batch)
  seqLen: number;          // tokens per sequence used in training
  epochs: number;
  precision: 'fp16' | 'fp32';
  // Per-task data-parallel slice the distributed workers actually run.
  sliceBatch: number;
  approxParams: number;
  notes: string;
}

/** Compute the exact parameter count for a GPT-style model (tied LM head). */
export function transformerParamCount(c: { dModel: number; dFF: number; nLayer: number; vocab: number; context: number }): number {
  const tokenEmb = c.vocab * c.dModel;
  const posEmb = c.context * c.dModel;
  const perLayer =
    2 * c.dModel +                       // ln_1 (gain+bias)
    (c.dModel * 3 * c.dModel + 3 * c.dModel) + // c_attn (QKV) + bias
    (c.dModel * c.dModel + c.dModel) +   // attn out proj + bias
    2 * c.dModel +                       // ln_2
    (c.dModel * c.dFF + c.dFF) +         // mlp fc + bias
    (c.dFF * c.dModel + c.dModel);       // mlp proj + bias
  return tokenEmb + posEmb + c.nLayer * perLayer + 2 * c.dModel; // + final ln
}

// Canonical GPT-1 (~117M params).
export const GPT1_CANONICAL: TransformerConfig = {
  name: 'GPT-1 (canonical, Radford et al. 2018)',
  nLayer: 12,
  nHead: 12,
  dModel: 768,
  dFF: 3072,
  vocab: 40478,           // BPE, 40,000 merges + base
  context: 512,
  dropout: 0.1,
  activation: 'gelu',
  initStd: 0.02,
  tokenizer: 'bpe',
  optimizer: 'adam',
  maxLR: 2.5e-4,
  warmupSteps: 2000,      // linear warmup, then cosine anneal to 0
  lrSchedule: 'warmup_cosine',
  weightDecay: 0.01,
  batchSequences: 64,     // 64 sequences of 512 tokens
  seqLen: 512,
  epochs: 100,            // over BooksCorpus
  precision: 'fp16',
  sliceBatch: 32,
  approxParams: transformerParamCount({ dModel: 768, dFF: 3072, nLayer: 12, vocab: 40478, context: 512 }),
  notes: 'Original architecture. NOT realistically trainable on browser workers — ~466MB weights shipped per task and tens of GB of activations per slice. Provided as a reference / target config.'
};

// A small GPT that can actually run distributed in a browser tab.
export const GPT1_FEASIBLE: TransformerConfig = {
  name: 'Tiny-GPT (feasible on browser workers)',
  nLayer: 4,
  nHead: 4,
  dModel: 128,
  dFF: 512,
  vocab: 256,             // byte-level — keeps the logits tensor small
  context: 128,
  dropout: 0.1,
  activation: 'gelu',
  initStd: 0.02,
  tokenizer: 'byte',
  optimizer: 'adam',
  maxLR: 3e-4,
  warmupSteps: 200,
  lrSchedule: 'warmup_cosine',
  weightDecay: 0.01,
  batchSequences: 16,
  seqLen: 128,
  epochs: 10,
  precision: 'fp16',
  sliceBatch: 4,
  approxParams: transformerParamCount({ dModel: 128, dFF: 512, nLayer: 4, vocab: 256, context: 128 }),
  notes: 'Same GPT recipe at a scale a phone/laptop browser can hold. Recommended starting point for distributed runs here.'
};

export const PRESETS = {
  gpt1: GPT1_CANONICAL,
  gpt1Feasible: GPT1_FEASIBLE
};
