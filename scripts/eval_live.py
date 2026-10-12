#!/usr/bin/env python3
"""End-to-end check of the live app: run LivePipeline on SoccerNet-GSR clips
and score its outputs with the official GS-HOTA evaluator.

By default frames are fed as fast as the pipeline takes them (no drops), which
measures accuracy; --realtime paces the clip at 25 FPS and drops frames the
pipeline cannot keep up with, which measures what a viewer would get.

Usage:
  python scripts/eval_live.py --config configs/live.yaml --split valid --clips 10 \
      --trackeval /data/saurav/repos/sn-trackeval
"""
import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gsr_assemble import CAT, run_eval  # noqa: E402
from sideline.pipeline import LivePipeline, Models  # noqa: E402


def to_gsr(out, image_ids):
    preds = []
    for j, p in enumerate(out["preds"]):
        if out["frame"] >= len(image_ids):
            continue
        iid = image_ids[out["frame"]]
        x, y, w, h = p["bbox"]
        q = p["pitch"]
        preds.append({"id": f"{iid}_{j}", "image_id": iid, "track_id": int(p["track_id"]),
                      "supercategory": "object", "category_id": CAT[p["attributes"]["role"]],
                      "attributes": p["attributes"],
                      "bbox_image": {"x": x, "y": y, "w": w, "h": h, "x_center": x + w / 2, "y_center": y + h / 2},
                      "bbox_pitch": {"x_bottom_left": q[0], "y_bottom_left": q[1], "x_bottom_middle": q[2],
                                     "y_bottom_middle": q[3], "x_bottom_right": q[4], "y_bottom_right": q[5]}})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live.yaml")
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--clips", type=int, default=0, help="first N clips (0 = all)")
    ap.add_argument("--delays", type=int, nargs="+", default=[0, 25, 125])
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--trackeval", required=True)
    ap.add_argument("--name", default="live_app")
    ap.add_argument("--sync-calib", action="store_true", help="calibrate keyframes inline (as in the paper)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg.setdefault("pipeline", {})["sync_calib"] = args.sync_calib
    models = Models(cfg)
    seqs = sorted(p for p in (Path(args.gt) / args.split).iterdir() if (p / "Labels-GameState.json").exists())
    seqs = seqs[: args.clips or None]
    root = Path("results/_work/live_app").resolve() / args.name
    shutil.rmtree(root, ignore_errors=True)
    stats = {}
    for s in seqs:
        image_ids = [im["image_id"] for im in json.loads((s / "Labels-GameState.json").read_text())["images"]]
        pipe = LivePipeline(models, cfg, delays=args.delays)
        got = {d: [] for d in args.delays}
        done = threading.Event()

        def sink(o):
            if o is None:
                done.set()
            else:
                got[o["delay"]] += to_gsr(o, image_ids)

        pipe.subscribe(sink)
        t0 = time.perf_counter()
        pipe.start(str(s), realtime=args.realtime)
        done.wait()
        pipe.wait()
        if pipe.error:
            raise RuntimeError(pipe.error)
        st = pipe.stats.snapshot()
        st["wall_s"] = round(time.perf_counter() - t0, 1)
        stats[s.name] = st
        for d in args.delays:
            out = root / f"SoccerNetGS-{args.split}" / f"{args.name}_L{d}" / "data"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{s.name}.json").write_text(json.dumps({"predictions": got[d]}))
        print(f"  {s.name}: {st['frames']} frames, dropped {st['dropped']}, {st['fps']} FPS, "
              f"latency p50 {st['latency_ms_p50']} ms, stages {st['stage_ms']}", flush=True)
    names = [s.name for s in seqs]
    res = {}
    for d in args.delays:
        r = run_eval(args.trackeval, Path(args.gt).resolve(), root, args.split, f"{args.name}_L{d}", names)
        res[d] = {"gs_hota": r["HOTA"], "deta": r["DetA"], "assa": r["AssA"]}
        print(f"live app L={d:4d} GS-HOTA={r['HOTA']:6.2f} DetA={r['DetA']:6.2f} AssA={r['AssA']:6.2f}")
    Path(f"results/eval_live_{args.name}.json").write_text(json.dumps({"clips": names, "scores": res,
                                                                        "stats": stats}, indent=1))


if __name__ == "__main__":
    main()
