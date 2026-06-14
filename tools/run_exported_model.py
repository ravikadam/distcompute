#!/usr/bin/env python3
"""
Load DistCompute-exported Tiny-GPT weights into a HuggingFace GPT-2 model and
generate text. The DistCompute GPT is deliberately GPT-2-shaped (learned token +
position embeddings, pre-norm blocks, causal multi-head attention, GELU MLP,
weight-tied LM head, byte-level vocab=256), so it maps onto GPT2LMHeadModel
with no architecture changes — only a weight remap.

Install:
    pip install torch transformers numpy

Run:
    python tools/run_exported_model.py --weights model_weights.json --prompt "To be"

Notes:
  * Tokenizer is byte-level: text -> UTF-8 bytes (ids 0..255) and back. No vocab file.
  * Context is the model's trained window (128 for Tiny-GPT); generation crops to it.
"""
import argparse, json, sys
import numpy as np

def load(weights_path):
    with open(weights_path) as f:
        w = json.load(f)
    if "wte" not in w:
        sys.exit("This file doesn't look like a GPT export (no 'wte'). Was it the char-MLP model?")
    return w

def arr(w, name, shape):
    a = np.array(w[name], dtype=np.float32)
    return a.reshape(shape)

def infer_config(w, n_head):
    # Byte-level model: vocab is 256; derive the rest from weight sizes.
    wte = np.array(w["wte"], dtype=np.float32)
    vocab = 256
    d = wte.size // vocab
    context = np.array(w["wpe"], dtype=np.float32).size // d
    n_layer = len({int(k[1:].split("_")[0]) for k in w if k.startswith("h") and k[1:2].isdigit()})
    d_ff = np.array(w[f"h0_mlp_fc_w"], dtype=np.float32).size // d
    return dict(vocab=vocab, d=d, context=context, n_layer=n_layer, d_ff=d_ff, n_head=n_head)

def build_model(w, cfg):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    conf = GPT2Config(
        vocab_size=cfg["vocab"], n_positions=cfg["context"], n_embd=cfg["d"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_inner=cfg["d_ff"],
        activation_function="gelu_new", layer_norm_epsilon=1e-5,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(conf)
    d, dff = cfg["d"], cfg["d_ff"]
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    sd = {}
    sd["transformer.wte.weight"] = t(arr(w, "wte", (cfg["vocab"], d)))
    sd["transformer.wpe.weight"] = t(arr(w, "wpe", (cfg["context"], d)))
    sd["transformer.ln_f.weight"] = t(arr(w, "ln_f_g", (d,)))
    sd["transformer.ln_f.bias"]   = t(arr(w, "ln_f_b", (d,)))
    for i in range(cfg["n_layer"]):
        p = f"h{i}_"; h = f"transformer.h.{i}."
        sd[h+"ln_1.weight"] = t(arr(w, p+"ln1_g", (d,)))
        sd[h+"ln_1.bias"]   = t(arr(w, p+"ln1_b", (d,)))
        # combined QKV: HF Conv1D weight is [in, out] == our Linear [in, out]; concat on out.
        qkv_w = np.concatenate([arr(w, p+"attn_wq_w", (d, d)),
                                arr(w, p+"attn_wk_w", (d, d)),
                                arr(w, p+"attn_wv_w", (d, d))], axis=1)   # [d, 3d]
        qkv_b = np.concatenate([arr(w, p+"attn_wq_b", (d,)),
                                arr(w, p+"attn_wk_b", (d,)),
                                arr(w, p+"attn_wv_b", (d,))])             # [3d]
        sd[h+"attn.c_attn.weight"] = t(qkv_w)
        sd[h+"attn.c_attn.bias"]   = t(qkv_b)
        sd[h+"attn.c_proj.weight"] = t(arr(w, p+"attn_wo_w", (d, d)))
        sd[h+"attn.c_proj.bias"]   = t(arr(w, p+"attn_wo_b", (d,)))
        sd[h+"ln_2.weight"] = t(arr(w, p+"ln2_g", (d,)))
        sd[h+"ln_2.bias"]   = t(arr(w, p+"ln2_b", (d,)))
        sd[h+"mlp.c_fc.weight"]   = t(arr(w, p+"mlp_fc_w", (d, dff)))
        sd[h+"mlp.c_fc.bias"]     = t(arr(w, p+"mlp_fc_b", (dff,)))
        sd[h+"mlp.c_proj.weight"] = t(arr(w, p+"mlp_proj_w", (dff, d)))
        sd[h+"mlp.c_proj.bias"]   = t(arr(w, p+"mlp_proj_b", (d,)))
    sd["lm_head.weight"] = sd["transformer.wte.weight"]  # tied head
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # HF adds attn.bias / masked_bias buffers (causal mask) — expected to be "missing".
    real_missing = [k for k in missing if not (k.endswith("attn.bias") or k.endswith("attn.masked_bias"))]
    if real_missing:
        print("WARNING: unset params:", real_missing)
    if unexpected:
        print("WARNING: unexpected params:", unexpected)
    model.eval()
    return model

def generate(model, prompt, n_new, temperature, context):
    import torch
    ids = list(prompt.encode("utf-8")) or [ord(" ")]
    with torch.no_grad():
        for _ in range(n_new):
            ctx = ids[-context:]
            logits = model(torch.tensor([ctx])).logits[0, -1]   # [vocab]
            if temperature <= 0:
                nxt = int(torch.argmax(logits))
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            ids.append(nxt)
    return bytes(ids).decode("utf-8", errors="replace")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="exported weights JSON (Download Weights)")
    ap.add_argument("--manifest", help="optional models/tiny-gpt.manifest.json for exact config")
    ap.add_argument("--prompt", default="To be, or not to be")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy/argmax")
    ap.add_argument("--heads", type=int, default=4, help="n_head (used if no manifest)")
    args = ap.parse_args()

    w = load(args.weights)
    if args.manifest:
        mc = json.load(open(args.manifest))["config"]
        cfg = dict(vocab=mc["vocab"], d=mc["dModel"], context=mc["context"],
                   n_layer=mc["nLayer"], d_ff=mc["dFF"], n_head=mc["nHead"])
    else:
        cfg = infer_config(w, args.heads)
    print("Config:", cfg)
    model = build_model(w, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded GPT-2 model: {n_params/1e3:.0f}K params\n")
    print(f"--- prompt: {args.prompt!r} ---")
    print(generate(model, args.prompt, args.tokens, args.temperature, cfg["context"]))

if __name__ == "__main__":
    main()
