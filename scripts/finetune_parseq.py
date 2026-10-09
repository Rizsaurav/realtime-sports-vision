#!/usr/bin/env python3
"""Fine-tune the SoccerNet PARSeq jersey reader on SoccerNet-GSR crops.

Uses PARSeq's own training_step (permutation language modelling loss) and its
RandAugment transform in a plain PyTorch loop; the last `--holdout` clips are
held out for checkpoint selection (exact-match accuracy). The best weights are
saved in the original checkpoint format so load_from_checkpoint still works.
Runs in the jersey env.

Usage:
  python scripts/finetune_parseq.py --data data/processed/jersey_ft \
      --ckpt /data/saurav/repos/jersey-number-pipeline/models/parseq_*.ckpt \
      --out runs/jersey/parseq_gsr_ft.ckpt --epochs 6
"""
import argparse
import glob
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = "/data/saurav/repos/jersey-number-pipeline"
sys.path.insert(0, f"{REPO}/str/parseq")
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


class Crops(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.t = Path(root), rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, lab = self.rows[i][:2]
        return self.t(Image.open(self.root / "images" / name).convert("RGB")), lab


def evaluate(model, loader, device):
    model.eval()
    ok = n = 0
    with torch.no_grad():
        for x, labels in loader:
            probs = model(x.to(device)).softmax(-1)
            preds, _ = model.tokenizer.decode(probs)
            ok += sum(p == l for p, l in zip(preds, labels))
            n += len(labels)
    model.train()
    return ok / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/jersey_ft")
    ap.add_argument("--ckpt", default=f"{REPO}/models/parseq_*.ckpt")
    ap.add_argument("--out", default="runs/jersey/parseq_gsr_ft.ckpt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--holdout", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    dev = "cuda"
    ckpt_path = sorted(glob.glob(args.ckpt))[0]
    model = load_from_checkpoint(ckpt_path).to(dev)
    model.log = lambda *a, **k: None                     # no Lightning trainer here
    hw = tuple(model.hparams.img_size)

    rows = [l.split("\t") for l in Path(args.data, "labels.tsv").read_text().splitlines() if l]
    seqs = sorted({r[2] for r in rows})
    hold = set(seqs[-args.holdout:])
    tr = [r for r in rows if r[2] not in hold]
    ho = [r for r in rows if r[2] in hold]
    print(f"[ft] {len(tr)} train crops, {len(ho)} holdout crops ({len(hold)} clips)", flush=True)
    tl = DataLoader(Crops(args.data, tr, SceneTextDataModule.get_transform(hw, augment=True)),
                    batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    hl = DataLoader(Crops(args.data, ho, SceneTextDataModule.get_transform(hw)),
                    batch_size=256, num_workers=args.workers)

    best = evaluate(model, hl, dev)
    print(f"[ft] epoch -1 (pretrained): holdout exact {100 * best:.2f}%", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(tl),
                                                pct_start=0.1, cycle_momentum=False)
    model.train()
    ck = torch.load(ckpt_path, map_location="cpu")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        tot = 0.0
        for bi, (x, labels) in enumerate(tl):
            loss = model.training_step((x.to(dev), list(labels)), bi)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            opt.step()
            sched.step()
            tot += loss.item()
        acc = evaluate(model, hl, dev)
        print(f"[ft] epoch {ep}: loss {tot / len(tl):.4f}  holdout exact {100 * acc:.2f}%", flush=True)
        if acc > best:
            best = acc
            ck["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(ck, args.out)
            print(f"[ft] saved {args.out}", flush=True)
    print(f"[ft] best holdout exact {100 * best:.2f}%")


if __name__ == "__main__":
    main()
