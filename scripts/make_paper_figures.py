#!/usr/bin/env python3
"""Vector figures for the IEEE paper (paper/figures/*.pdf).

Reads docs/report/data.json (written by make_report.py), the bootstrap
results, and one test clip's predictions for the qualitative figure.

Usage:
  python scripts/make_paper_figures.py [--clip SNGS-116 --frame 300 --delay L125]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402

COL = 3.5          # IEEE single column width, inches
OUT = Path("paper/figures")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
X = [0, 1, 5, 10, 14]
XT = ["0", "1 s", "5 s", "10 s", "clip"]
GREEN, ORANGE, GREY, BLUE, PURPLE = "#1b6e45", "#c4572e", "#8a8f8b", "#2a5c9a", "#6a4c9c"


def delay_curve(d, bt):
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    names = ["live", "d1s", "d5s", "d10s", "offline"]
    lo = [bt["sets"][n]["ci95"][0] for n in names]
    hi = [bt["sets"][n]["ci95"][1] for n in names]
    ax.fill_between(X, lo, hi, color=GREEN, alpha=0.13, lw=0, label="test 95% CI")
    ax.plot(X, d["e2e_test_final"], "o-", color=GREEN, ms=3.2, label="final, test")
    ax.plot(X, d["e2e_valid_ablation"][5]["gs_hota"], "s-", color=ORANGE, ms=3, label="final, validation")
    ax.plot(X, d["e2e_valid_ablation"][0]["gs_hota"], "o--", color=GREY, ms=2.6, lw=1.0,
            label="ByteTrack baseline, validation")
    ax.set_xticks(X, XT)
    ax.set_xlabel("Output delay")
    ax.set_ylabel("GS-HOTA")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1.4,
              columnspacing=0.8, handletextpad=0.4, fontsize=6.4)
    ax.set_ylim(24, 49)
    fig.savefig(OUT / "delay_curve.pdf")
    plt.close(fig)


def components(d):
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    j = d["jersey_only_valid"]["gated + pose crops"]
    t = d["team_only_valid"]["online re-fit"]
    c = d["calibration_valid"]["keyframe every 10"]["gs_hota_calib_only"]
    jp = [(x, v) for x, v in zip(X, j) if v is not None]
    ax.plot([p[0] for p in jp], [p[1] for p in jp], "o:", color=ORANGE, ms=3.4, label="jersey (measured points)")
    ax.plot(X, t, "s-", color=BLUE, ms=3, label="team")
    ax.plot(X, [c] * 5, ":", color=PURPLE, lw=1.4, label="calibration")
    ax.set_xticks(X, XT)
    ax.set_xlabel("Output delay")
    ax.set_ylabel("Component-only GS-HOTA")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.6)
    ax.set_ylim(55, 96)
    fig.savefig(OUT / "components.pdf")
    plt.close(fig)


def draw_pitch(ax):
    L, W = 105.0, 68.0
    kw = dict(fill=False, lw=0.6, color="#4b5a50")
    ax.add_patch(Rectangle((-L / 2, -W / 2), L, W, **kw))
    ax.plot([0, 0], [-W / 2, W / 2], lw=0.6, color="#4b5a50")
    ax.add_patch(Circle((0, 0), 9.15, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * L / 2 - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((s * L / 2 - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, **kw))
        ax.add_patch(Arc((s * (L / 2 - 11), 0), 18.3, 18.3, angle=0,
                         theta1=(127 if s > 0 else -53), theta2=(233 if s > 0 else 53), lw=0.6, color="#4b5a50"))
    ax.set_xlim(-L / 2 - 3, L / 2 + 3)
    ax.set_ylim(W / 2 + 3, -W / 2 - 3)        # SoccerNet: negative y is the far touchline
    ax.set_aspect("equal")
    ax.axis("off")


def qualitative(clip, frame_idx, delay, split="test"):
    import cv2
    gt_dir = Path("data/raw/soccernet/gsr") / split / clip
    g = json.loads((gt_dir / "Labels-GameState.json").read_text())
    im = g["images"][frame_idx]
    img = cv2.cvtColor(cv2.imread(str(gt_dir / "img1" / im["file_name"])), cv2.COLOR_BGR2RGB)
    pred = json.loads(Path(f"results/_work/gsr_e2e/test_final/SoccerNetGS-{split}/test_final_{delay}/data/"
                           f"{clip}.json").read_text())["predictions"]
    P = [p for p in pred if p["image_id"] == im["image_id"]]
    G = [a for a in g["annotations"] if a["image_id"] == im["image_id"] and a.get("supercategory") == "object"
         and a["attributes"]["role"] != "ball"]
    team_c = {"left": "#D55E00", "right": "#0072B2", None: "#E69F00"}
    fig = plt.figure(figsize=(7.16, 2.35))
    ax1 = fig.add_axes([0.0, 0.0, 0.56, 1.0])
    ax2 = fig.add_axes([0.575, 0.06, 0.425, 0.88])
    ax1.imshow(img)
    for p in P:
        b, a = p["bbox_image"], p["attributes"]
        c = team_c.get(a["team"], "#E69F00") if a["role"] in ("player", "goalkeeper") else "#f2f2f2"
        ax1.add_patch(Rectangle((b["x"], b["y"]), b["w"], b["h"], fill=False, lw=0.8, color=c))
        lab = a["jersey"] if a["jersey"] else ("R" if a["role"] == "referee" else "")
        if lab:
            ax1.text(b["x"] + b["w"] / 2, b["y"] - 6, lab, color="white", fontsize=7, ha="center", va="bottom",
                     bbox=dict(boxstyle="square,pad=0.12", fc=c, ec="none", alpha=0.9))
    ax1.axis("off")
    draw_pitch(ax2)
    for a in G:
        bp = a.get("bbox_pitch")
        if bp:
            ax2.plot(bp["x_bottom_middle"], bp["y_bottom_middle"], "o", ms=5.2, mfc="none", mec="#111111",
                     mew=0.9)
    for p in P:
        a, bp = p["attributes"], p["bbox_pitch"]
        c = team_c.get(a["team"], "#E69F00") if a["role"] in ("player", "goalkeeper") else "#9a9a9a"
        ax2.plot(bp["x_bottom_middle"], bp["y_bottom_middle"], "o", ms=3.2, color=c)
    ax2.plot([], [], "o", ms=5.2, mfc="none", mec="#111111", mew=0.9, label="ground truth")
    ax2.plot([], [], "o", ms=3.2, color="#D55E00", label="ours, left team")
    ax2.plot([], [], "o", ms=3.2, color="#0072B2", label="ours, right team")
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=7,
               handletextpad=0.3, columnspacing=0.9)
    fig.savefig(OUT / "qualitative.pdf", dpi=300)
    plt.close(fig)
    return {"clip": clip, "frame": frame_idx + 1, "n_pred": len(P), "n_gt": len(G)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="SNGS-116")
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--delay", default="L125")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(Path("docs/report/data.json").read_text())
    delay_curve(d, json.loads(Path("results/gsr_bootstrap_test.json").read_text()))
    components(d)
    q = qualitative(args.clip, args.frame, args.delay)
    print(f"[figures] wrote {OUT}: delay_curve.pdf, components.pdf, qualitative.pdf {q}")


if __name__ == "__main__":
    main()
