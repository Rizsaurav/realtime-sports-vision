"""Metrics: speed (FPS, p50/p95 latency), accuracy (mAP, MOTA),
and cost (VRAM, watts via pynvml on NVIDIA GPUs).

Accuracy metrics need ground truth (SportsMOT gt.txt); speed metrics
need only the latency trace from StreamingPipeline.
"""

import numpy as np


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compute_detection_metrics(pred_by_frame, gt_by_frame, iou_thr=0.5):
    """Simple mAP@iou_thr over frames.

    pred_by_frame: {fid: [(x1,y1,x2,y2,score,cls)]}
    gt_by_frame:   {fid: [(tid,x1,y1,x2,y2)]}
    Returns {"map": float, "n_gt": int, "n_pred": int}.
    Greedy matching, 11-point interpolation — fine for ablations;
    use pycocotools if you need publication-grade numbers.
    """
    aps, n_gt_total, n_pred_total = [], 0, 0
    for fid, gts in gt_by_frame.items():
        preds = sorted(pred_by_frame.get(fid, []), key=lambda d: -d[4])
        n_gt_total += len(gts)
        n_pred_total += len(preds)
        if not gts or not preds:
            aps.append(0.0)
            continue
        matched = [False] * len(gts)
        tp = np.zeros(len(preds))
        for i, p in enumerate(preds):
            best, bj = iou_thr, -1
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                v = _iou(p[:4], g[1:5])
                if v > best:
                    best, bj = v, j
            if bj >= 0:
                matched[bj] = True
                tp[i] = 1
        fp = 1 - tp
        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        rec = tp_c / len(gts)
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            ap += np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
        aps.append(ap / 11.0)
    return {"map": float(np.mean(aps)) if aps else 0.0,
            "n_gt": n_gt_total, "n_pred": n_pred_total}


def compute_tracking_metrics(tracks_out, ground_truth):
    """MOTA/MOTP via motmetrics.

    tracks_out: [(frame_id, [(x1,y1,x2,y2,track_id,cls)])]
    ground_truth: {fid: [(tid,x1,y1,x2,y2)]}
    Returns {"mota": float, "motp": float, "id_switches": int}.
    Falls back to a simple MOTA computation if motmetrics is missing.
    """
    try:
        import motmetrics as mm
        # motmetrics 1.4.0 predates NumPy 2.0 and calls np.asfarray, which
        # was removed; alias it so the motmetrics path keeps working.
        if not hasattr(np, "asfarray"):
            def _asfarray(a, dtype=float):
                return np.asarray(a, dtype=dtype)
            np.asfarray = _asfarray
        acc = mm.MOTAccumulator(auto_id=True)
        for fid, tracks in tracks_out:
            gt = ground_truth.get(fid, [])
            gt_ids = [g[0] for g in gt]
            gt_boxes = [g[1:5] for g in gt]
            hyp_ids = [t[4] for t in tracks]
            hyp_boxes = [[t[0], t[1], t[2], t[3]] for t in tracks]
            if gt_ids or hyp_ids:
                dist = mm.distances.iou_matrix(
                    gt_boxes, hyp_boxes, max_iou=0.5) if (gt_boxes and hyp_boxes) \
                    else np.empty((len(gt_ids), len(hyp_ids)))
                acc.update(gt_ids, hyp_ids, dist)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=["mota", "motp", "num_switches"],
                             name="acc")
        row = summary.loc["acc"]
        return {"mota": float(row["mota"]), "motp": float(row["motp"]),
                "id_switches": int(row["num_switches"])}
    except ImportError:
        # Simple MOTA fallback: misses + false positives + mismatches over GT
        misses = fp = switches = total_gt = 0
        prev = {}
        for fid, tracks in sorted(tracks_out):
            gt = {g[0]: g[1:5] for g in ground_truth.get(fid, [])}
            total_gt += len(gt)
            matched_gt, matched_tr = set(), set()
            for t in tracks:
                best, bg = 0.5, None
                for gid, gb in gt.items():
                    if gid in matched_gt:
                        continue
                    v = _iou(t[:4], gb)
                    if v > best:
                        best, bg = v, gid
                if bg is None:
                    fp += 1
                else:
                    matched_gt.add(bg)
                    matched_tr.add(t[4])
                    if bg in prev and prev[bg] != t[4]:
                        switches += 1
                    prev[bg] = t[4]
            misses += len(gt) - len(matched_gt)
        mota = 1 - (misses + fp + switches) / max(total_gt, 1)
        return {"mota": float(mota), "motp": None, "id_switches": switches}


def gpu_stats():
    """Peak VRAM MB + current watts via pynvml. {} on CPU-only machines."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        try:
            watts = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            watts = None
        return {"vram_mb": round(mem.used / 1024 ** 2, 1), "watts": watts}
    except Exception:
        return {}
