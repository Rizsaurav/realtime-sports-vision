#!/usr/bin/env python3
"""Export the jersey models to ONNX so the live app runs them in the main env.

Runs once in the jersey env (PARSeq's strhub pins pytorch-lightning<2):
  legibility.onnx  whole-player crop (N,3,256,256) ImageNet-normalised -> P(legible) (N,1)
  parseq.onnx      torso crop (N,3,32,128) in [-1, 1] -> logits (N,4,C), greedy autoregressive
                   decoding over 4 steps plus one refinement pass, as in the SoccerNet reader

Only the first three token distributions are used downstream (numbers 0-99), and they equal the
full-length decoder's whenever the read has at most three characters.
Checks the exported models against PyTorch on random crops before exiting.

Usage (jersey env):
  python scripts/export_jersey_onnx.py --repo /data/saurav/repos/jersey-number-pipeline --out models/jersey
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

STEPS = 4


class ParseqFixed(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, images):
        m = self.m
        bs = images.shape[0]
        memory = m.encode(images)
        pos = m.pos_queries[:, :STEPS].expand(bs, -1, -1)
        tgt_mask = torch.triu(torch.full((STEPS, STEPS), float("-inf"), device=images.device), 1)
        query_mask = tgt_mask.clone()
        tokens = [torch.full((bs, 1), m.bos_id, dtype=torch.long, device=images.device)]
        logits = []
        for i in range(STEPS):
            j = i + 1
            tgt_in = torch.cat(tokens, 1)
            out = m.decode(tgt_in, memory, tgt_mask[:j, :j], tgt_query=pos[:, i:j],
                           tgt_query_mask=query_mask[i:j, :j])
            p = m.head(out)
            logits.append(p)
            tokens.append(p[:, 0].argmax(-1, keepdim=True))
        logits = torch.cat(logits, 1)
        cloze = query_mask.masked_fill(torch.triu(torch.ones(STEPS, STEPS, dtype=torch.bool,
                                                             device=images.device), 2), 0.0)
        tgt_in = torch.cat([tokens[0], logits[:, :-1].argmax(-1)], 1)
        pad = (tgt_in == m.eos_id).int().cumsum(-1) > 0
        out = m.decode(tgt_in, memory, tgt_mask, pad, tgt_query=pos, tgt_query_mask=cloze)
        return m.head(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/data/saurav/repos/jersey-number-pipeline")
    ap.add_argument("--parseq-ckpt", default=None)
    ap.add_argument("--out", default="models/jersey")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_jersey_parseq import load_models

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    leg, stm = load_models(args.repo, "cpu", args.parseq_ckpt)
    leg.eval()
    stm.eval()
    torch.manual_seed(0)

    x = torch.randn(4, 3, 256, 256)
    torch.onnx.export(leg, x, out / "legibility.onnx", input_names=["crop"], output_names=["p_legible"],
                      dynamic_axes={"crop": {0: "n"}, "p_legible": {0: "n"}}, opset_version=17)
    h, w = stm.hparams.img_size
    wrap = ParseqFixed(stm).eval()
    y = torch.rand(4, 3, h, w) * 2 - 1
    torch.onnx.export(wrap, y, out / "parseq.onnx", input_names=["crop"], output_names=["logits"],
                      dynamic_axes={"crop": {0: "n"}, "logits": {0: "n"}}, opset_version=17)

    import onnxruntime as ort
    with torch.no_grad():
        ref_leg = leg(x).numpy()
        ref_full = stm(y)[:, :3].softmax(-1).numpy()
        ref_fix = wrap(y)[:, :3].softmax(-1).numpy()
    s1 = ort.InferenceSession(str(out / "legibility.onnx"), providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(str(out / "parseq.onnx"), providers=["CPUExecutionProvider"])
    o1 = s1.run(None, {"crop": x.numpy()})[0]
    o2 = torch.from_numpy(s2.run(None, {"crop": y.numpy()})[0][:, :3]).softmax(-1).numpy()
    print(f"[export] legibility max |diff| {np.abs(o1 - ref_leg).max():.2e}")
    print(f"[export] parseq onnx vs wrapper max |diff| {np.abs(o2 - ref_fix).max():.2e}")
    print(f"[export] wrapper vs full decoder (first 3 tokens) max |diff| {np.abs(ref_fix - ref_full).max():.2e}"
          " (random inputs; equal for reads of <= 3 characters)")
    (out / "charset.txt").write_text(stm.hparams.charset_test)
    print(f"[export] wrote {out}/legibility.onnx, parseq.onnx (img {h}x{w})")


if __name__ == "__main__":
    main()
