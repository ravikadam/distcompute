#!/usr/bin/env python3
"""
Convert DistCompute model_weights.json to a standard PyTorch Hugging Face GPT-2
format and push it to the Hugging Face Hub.

Install dependencies:
    pip install torch transformers huggingface_hub numpy

Run:
    python tools/push_to_huggingface.py --weights ~/Downloads/model_weights.json --repo-id "your-username/your-model-name"
"""
import argparse
import json
import os
import sys
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
        qkv_w = np.concatenate([arr(w, p+"attn_wq_w", (d, d)),
                                arr(w, p+"attn_wk_w", (d, d)),
                                arr(w, p+"attn_wv_w", (d, d))], axis=1)
        qkv_b = np.concatenate([arr(w, p+"attn_wq_b", (d,)),
                                arr(w, p+"attn_wk_b", (d,)),
                                arr(w, p+"attn_wv_b", (d,))])
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
    sd["lm_head.weight"] = sd["transformer.wte.weight"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to model_weights.json")
    ap.add_argument("--manifest", help="Path to models/tiny-gpt.manifest.json")
    ap.add_argument("--repo-id", required=True, help="Hugging Face repo ID (e.g. 'username/model-name')")
    ap.add_argument("--token", help="Hugging Face Write Token (optional if logged in via cli)")
    ap.add_argument("--heads", type=int, default=4, help="n_head (default 4)")
    args = ap.parse_args()

    w = load(args.weights)
    if args.manifest:
        mc = json.load(open(args.manifest))["config"]
        cfg = dict(vocab=mc["vocab"], d=mc["dModel"], context=mc["context"],
                   n_layer=mc["nLayer"], d_ff=mc["dFF"], n_head=mc["nHead"])
    else:
        cfg = infer_config(w, args.heads)
    
    print("Inferred model configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    print("\nBuilding PyTorch model...")
    model = build_model(w, cfg)
    
    print(f"Pushing model to Hugging Face Hub repository: '{args.repo_id}'...")
    model.push_to_hub(args.repo_id, token=args.token)
    print("\nSuccess! Model has been successfully converted and uploaded.")

if __name__ == "__main__":
    main()
