"""Standard MOT + COCO evaluation via maintained open-source tools.

Tracking: roboflow `trackers eval` (TrackEval-compatible HOTA / CLEAR / Identity).
Detection: torchmetrics MeanAveragePrecision (faster_coco_eval backend, COCO mAP).
Both replace the home-grown metrics in metrics.py (kept only for the unit tests).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def write_mot_file(path, tracks_by_frame):
    """tracks_by_frame: {frame_id: [(x1, y1, x2, y2, track_id, cls)]} -> MOT txt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fid in sorted(tracks_by_frame):
            for x1, y1, x2, y2, tid, _ in tracks_by_frame[fid]:
                f.write(f"{fid},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},"
                        f"{y2 - y1:.2f},1,-1,-1,-1\n")


def write_gt_subset(src_seq_dir, dst_seq_dir, last_frame):
    """Copy gt.txt (frames <= last_frame) + seqinfo.ini so partial runs are scored fairly."""
    src_seq_dir, dst_seq_dir = Path(src_seq_dir), Path(dst_seq_dir)
    (dst_seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    keep = []
    for line in (src_seq_dir / "gt" / "gt.txt").read_text().splitlines():
        if line and int(float(line.split(",")[0])) <= last_frame:
            keep.append(line)
    (dst_seq_dir / "gt" / "gt.txt").write_text("\n".join(keep) + "\n")
    ini = src_seq_dir / "seqinfo.ini"
    if ini.exists():
        shutil.copy(ini, dst_seq_dir / "seqinfo.ini")


def run_trackeval(gt_dir, tracker_dir, seqs, work_dir):
    """Run `trackers eval`; return aggregate HOTA/DetA/AssA/IDF1/MOTA (0-1) + counts."""
    work_dir = Path(work_dir)
    seqmap = work_dir / "seqmap.txt"
    seqmap.write_text("\n".join(seqs) + "\n")
    out_json = work_dir / "trackeval.json"
    cli = Path(sys.executable).parent / "trackers"
    cmd = [str(cli), "eval", "--gt-dir", str(gt_dir),
           "--tracker-dir", str(tracker_dir), "--seqmap", str(seqmap),
           "--metrics", "CLEAR", "HOTA", "Identity", "-o", str(out_json)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    agg = json.loads(out_json.read_text())["aggregate"]
    return {
        "hota": agg["HOTA"]["HOTA"], "deta": agg["HOTA"]["DetA"],
        "assa": agg["HOTA"]["AssA"], "idf1": agg["Identity"]["IDF1"],
        "mota": agg["CLEAR"]["MOTA"], "motp": agg["CLEAR"]["MOTP"],
        "id_switches": int(agg["CLEAR"]["IDSW"]),
        "clr_fp": int(agg["CLEAR"]["CLR_FP"]), "clr_fn": int(agg["CLEAR"]["CLR_FN"]),
    }


def coco_map(pred_by_frame, gt_by_frame):
    """COCO mAP (single 'player' class). Keys are (seq_idx, frame_id) tuples.

    pred_by_frame: {key: [(x1,y1,x2,y2,score,cls)]}; gt_by_frame: {key: [(tid,x1,y1,x2,y2)]}
    Returns map (0.5:0.95), map50, map75 in 0-1.
    """
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  backend="faster_coco_eval")
    keys = sorted(set(gt_by_frame) | set(pred_by_frame))
    for i in range(0, len(keys), 256):
        preds, targets = [], []
        for k in keys[i:i + 256]:
            p, g = pred_by_frame.get(k, []), gt_by_frame.get(k, [])
            preds.append({
                "boxes": torch.tensor([d[:4] for d in p], dtype=torch.float32).reshape(-1, 4),
                "scores": torch.tensor([d[4] for d in p], dtype=torch.float32),
                "labels": torch.zeros(len(p), dtype=torch.int64)})
            targets.append({
                "boxes": torch.tensor([b[1:5] for b in g], dtype=torch.float32).reshape(-1, 4),
                "labels": torch.zeros(len(g), dtype=torch.int64)})
        metric.update(preds, targets)
    r = metric.compute()
    return {"map": float(r["map"]), "map50": float(r["map_50"]),
            "map75": float(r["map_75"])}
