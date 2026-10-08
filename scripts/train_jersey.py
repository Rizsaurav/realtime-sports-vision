#!/usr/bin/env python3
"""Train a small two-head jersey digit classifier (tens / units, 10 = none).

ResNet18 on 64x64 torso crops, ImageNet init. Real-time by design: a batch of
crops costs well under a millisecond on an H100.

Usage:
  python scripts/train_jersey.py --data data/processed/jersey/train.npz \
      --out runs/jersey/resnet18_64.pt --epochs 8
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision


class JerseyNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Identity()
        self.body = m
        self.h1 = nn.Linear(512, 11)
        self.h2 = nn.Linear(512, 11)

    def forward(self, x):
        f = self.body(x)
        return self.h1(f), self.h2(f)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(X, device):
    x = torch.from_numpy(X[..., ::-1].copy()).to(device).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN.to(device)) / STD.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/jersey/train.npz")
    ap.add_argument("--out", default="runs/jersey/resnet18_64.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    z = np.load(args.data)
    X, y = z["X"], z["y"]
    seqs = z["seq"]
    hold = np.isin(seqs, np.unique(seqs)[-5:])          # last 5 clips = sanity holdout
    Xtr, ytr, Xho, yho = X[~hold], y[~hold], X[hold], y[hold]
    dev = args.device
    net = JerseyNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(Xtr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(len(Xtr))
        t0, tot = time.time(), 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = to_tensor(Xtr[idx], dev)
            if rng.random() < 0.5:
                xb = xb + 0.05 * torch.randn_like(xb)
            yb = torch.from_numpy(ytr[idx]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o1, o2 = net(xb)
                loss = ce(o1.float(), yb[:, 0]) + ce(o2.float(), yb[:, 1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        net.eval()
        correct = n = 0
        with torch.no_grad():
            for i in range(0, len(Xho), 2048):
                o1, o2 = net(to_tensor(Xho[i:i + 2048], dev))
                p = torch.stack([o1.argmax(1), o2.argmax(1)], 1).cpu().numpy()
                correct += (p == yho[i:i + 2048]).all(1).sum()
                n += len(p)
        print(f"epoch {ep}: loss {tot / len(Xtr):.3f}  holdout per-crop exact {100 * correct / max(n, 1):.1f}%  "
              f"({time.time() - t0:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"[jersey] saved {args.out}")


if __name__ == "__main__":
    main()
