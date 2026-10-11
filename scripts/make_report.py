#!/usr/bin/env python3
"""Build the report's tables, figures and compact data file from results/*.json.

Every number in docs/REPORT.md comes from docs/report/data.json, which this
script writes; rerun it after new experiments.

Usage:
  python scripts/make_report.py            # reads results/, writes docs/report/
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("results")
OUT = Path("docs/report")
DELAYS = [(0, "0"), (25, "1 s"), (125, "5 s"), (250, "10 s"), (-1, "offline")]
FPS = 25.0


def load(name):
    return json.loads((R / name).read_text())


def by_variant(rows):
    return {r["variant"]: r for r in rows}


def e2e_series(rows, prefix):
    v = by_variant(rows)
    out = []
    for L, _ in DELAYS:
        key = f"{prefix}_L{'inf' if L < 0 else L}"
        if key in v:
            out.append(round(v[key].get("gs_hota", v[key].get("hota")), 2))
        else:
            out.append(None)
    pitch = v.get(f"{prefix}__pitch_hota_no_attrs", {}).get("hota")
    return out, (round(pitch, 2) if pitch is not None else None)


def jersey_series(rows, tag_fmt, delays=(0, 25, 125, 250, -1)):
    v = by_variant(rows)
    out = []
    for L, _ in DELAYS:
        key = tag_fmt.format(L="inf" if L < 0 else L)
        out.append(round(v[key]["gs_hota_jersey_only"], 2) if (L in delays and key in v) else None)
    return out


def team_series(rows, method):
    v = by_variant(rows)
    out = []
    for L, _ in DELAYS:
        key = f"team_{method}_L{'inf' if L < 0 else L}_w25"
        out.append(round(v[key]["gs_hota_team_only"], 2) if key in v else None)
    return out


def fmt(x):
    return "-" if x is None else f"{x:.1f}"


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        s += "| " + " | ".join(r) + " |\n"
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}

    # ---- end-to-end delay curves -----------------------------------------
    ev, et = load("gsr_e2e_valid.json"), load("gsr_e2e_test.json")
    stages = [
        ("ByteTrack, colour team, raw PARSeq jersey", "e2e_rw25"),
        ("+ McByte with shared camera-motion compensation", "e2e_mcbyte_shcmc2"),
        ("+ jersey evidence gating", "e2e_v3_gated"),
        ("+ pitch-space stitching", "e2e_v3_gated_stitch"),
        ("+ (team, jersey) identity linking", "e2e_v4_link"),
        ("+ pose-guided jersey crops (final)", "e2e_v5_pose"),
        ("final, fine-tuned jersey reader (gate 0.9)", "e2e_v7_ft_rc09"),
    ]
    data["e2e_valid_ablation"] = []
    for name, pre in stages:
        s, p = e2e_series(ev, pre)
        data["e2e_valid_ablation"].append({"name": name, "variant": pre, "gs_hota": s, "pitch_hota": p})
    data["e2e_test_final"], data["pitch_hota_test_final"] = e2e_series(et, "test_final")
    data["e2e_test_ft"], _ = e2e_series(et, "test_v7_ft")
    data["delays"] = [d for _, d in DELAYS]

    # ---- component curves --------------------------------------------------
    jr = load("gsr_jersey_valid.json")
    data["jersey_only_valid"] = {
        "raw PARSeq, tau 4": jersey_series(jr, "jersey_parseq_L{L}_tau4.0_r0.5"),
        "gated (read conf >= 0.97)": jersey_series(
            jr, "jersey_g_rc0.97_t4.0_rc0.97_lg0.3_pw1.0_L{L}_tau4.0_r0.6", (0, 125, -1)),
        "gated + pose crops": jersey_series(
            jr, "jersey_gtpose_pre_rc0.97_lg0.3_pw1.0_L{L}_tau4.0_r0.6", (0, 125, -1)),
        "gated + pose, fine-tuned reader": jersey_series(
            jr, "jersey_gtpose_ft_rc0.97_lg0.3_pw1.0_L{L}_tau4.0_r0.6", (0, 125, -1)),
    }
    tr = load("gsr_team_valid.json")
    data["team_only_valid"] = {"fit on first 1 s": team_series(tr, "lab"),
                               "online re-fit": team_series(tr, "refit")}
    calib = {}
    for k, f in (("every frame", "gsr_calib_pnlcalib_valid.json"),
                 ("keyframe every 5", "gsr_calib_pnlcalib_kf5_w640_valid.json"),
                 ("keyframe every 10", "gsr_calib_pnlcalib_kf10_w640_valid.json"),
                 ("keyframe every 25", "gsr_calib_pnlcalib_kf25_w640_valid.json"),
                 ("keyframe every 50", "gsr_calib_pnlcalib_kf50_w640_valid.json"),
                 ("first frame only", "gsr_calib_pnlcalib_kf-1_w640_valid.json")):
        d = load(f)
        calib[k] = {"gs_hota_calib_only": round(d["gs_hota_calib_only"], 2),
                    "err_m_median": d["err_m_median"], "pct_within_1m": d["pct_within_1m"],
                    "keyframe_pct": d["keyframe_pct"], "prop_ms_p50": d.get("prop_ms_p50")}
    data["calibration_valid"] = calib
    rf = load("gsr_calib_rf_kp_kp0.5_valid.json")
    data["calibration_rf_keypoints_2clips"] = {"gs_hota_calib_only": round(rf["gs_hota_calib_only"], 2),
                                               "err_m_median": rf["err_m_median"]}

    # ---- error budget and oracle swaps -------------------------------------
    data["oracle_ladder_valid"] = {r["variant"]: round(r["gs_hota"], 2)
                                   for r in load("gsr_oracle_ladder_valid.json")}
    data["oracle_swap_live_valid"] = {r["variant"]: round(r["gs_hota"], 2)
                                      for r in load("gsr_oracle_swap_e2e_rw25_L0_valid.json")}
    data["oracle_swap_offline_valid"] = {r["variant"]: round(r["gs_hota"], 2)
                                         for r in load("gsr_oracle_swap_e2e_rw25_Linf_valid.json")}

    # ---- tracking ------------------------------------------------------------
    sw = {r["name"]: r for r in json.loads((R / "tracker_sweep/gsr_yolo26s_1280.json").read_text())}
    keep = ["bm_bytetrack_low", "rf_sort", "rf_bytetrack", "rf_cbiou", "rf_ocsort",
            "rf_botsort_nocmc", "rf_botsort_cmc", "rf_mcbyte_nocmc", "rf_mcbyte_cmc",
            "rf_mcbyte_sharedcmc"]
    data["tracker_sweep_gsr_valid"] = [
        {"name": k, "hota": round(sw[k]["hota"], 3), "assa": round(sw[k]["assa"], 3),
         "idf1": round(sw[k]["idf1"], 3), "id_switches": sw[k]["id_switches"],
         "track_ms": sw[k]["track_ms_per_frame"]} for k in keep if k in sw]
    sm = {r["name"]: r for r in json.loads((R / "tracker_sweep/yolo26s_ft.json").read_text())}
    data["sportsmot_val"] = {
        "zero_shot_bytetrack": round(load("baseline_yolo26s_bytetrack.json")["hota"], 3),
        "finetuned_boxmot_bytetrack": round(load("baseline_yolo26s_ft_bytetrack.json")["hota"], 3),
        "finetuned_rf_bytetrack": round(sm["rf_bytetrack"]["hota"], 3),
        "finetuned_rf_mcbyte_cmc_interp": round(sm["rf_mcbyte_cmc+interp20"]["hota"], 3)
        if "rf_mcbyte_cmc+interp20" in sm else None}

    # ---- speed ---------------------------------------------------------------
    st = load("bench_gsr_stages.json")
    data["speed_stages_ms_fp32"] = {k: st[k]["p50"] for k in
                                    ("decode", "detect", "pose", "motion_prop", "track", "team_feat")}
    data["speed_stages_ms_fp32"]["keyframe_calib_amortised"] = st["keyframe_calib"]["amortised_per_frame"]
    data["speed_stages_ms_fp32"]["keyframe_calib_per_call"] = st["keyframe_calib"]["per_keyframe_p50"]
    data["speed_jersey_ms"] = load("bench_jersey_stage.json")["jersey_ms_per_frame_p50"]
    data["speed_tensorrt_fp16_ms"] = {"detect": 5.00, "pose": 5.11}  # from the engine export benchmark
    la, lj = load("live_runner_alone.json"), load("live_runner_with_jersey.json")
    data["live_runner"] = {
        "without_jersey": {k: la[k] for k in ("sustained_fps", "latency_ms_p50", "latency_ms_p95")},
        "with_jersey": {k: lj[k] for k in ("sustained_fps", "latency_ms_p50", "latency_ms_p95")}}

    (OUT / "data.json").write_text(json.dumps(data, indent=1))

    # ---- tables --------------------------------------------------------------
    hdr = ["Configuration"] + data["delays"] + ["Pitch HOTA"]
    t = "## End-to-end GS-HOTA vs output delay, SoccerNet-GSR valid (58 clips)\n\n"
    t += md_table(hdr, [[r["name"]] + [fmt(x) for x in r["gs_hota"]] + [fmt(r["pitch_hota"])]
                        for r in data["e2e_valid_ablation"]])
    t += "\n## End-to-end GS-HOTA vs output delay, SoccerNet-GSR test (49 clips)\n\n"
    t += md_table(hdr, [["final (pretrained jersey reader)"] + [fmt(x) for x in data["e2e_test_final"]]
                        + [fmt(data["pitch_hota_test_final"])],
                        ["final, fine-tuned jersey reader"] + [fmt(x) for x in data["e2e_test_ft"]] + ["-"]])
    t += "\n## Component curves, valid (other attributes from ground truth)\n\n"
    rows = [[f"jersey: {k}"] + [fmt(x) for x in v] for k, v in data["jersey_only_valid"].items()]
    rows += [[f"team: {k}"] + [fmt(x) for x in v] for k, v in data["team_only_valid"].items()]
    t += md_table(["Component"] + data["delays"], rows)
    t += "\n## Calibration-only GS-HOTA, valid (PnLCalib keyframes + optical-flow propagation)\n\n"
    t += md_table(["Policy", "GS-HOTA", "median error (m)", "within 1 m", "calibrator calls", "propagation ms"],
                  [[k, fmt(v["gs_hota_calib_only"]), f"{v['err_m_median']:.2f}", f"{v['pct_within_1m']:.1f}%",
                    f"{v['keyframe_pct']:.1f}%", "-" if v["prop_ms_p50"] is None else f"{v['prop_ms_p50']:.1f}"]
                   for k, v in calib.items()])
    lad = data["oracle_ladder_valid"]
    t += "\n## Error budget: ground truth with one factor degraded, valid\n\n"
    t += md_table(["Degradation", "GS-HOTA"], [[k, fmt(v)] for k, v in lad.items()])
    t += "\n## Oracle swaps on live end-to-end predictions, valid\n\n"
    t += md_table(["Swap in ground truth for", "GS-HOTA live", "GS-HOTA offline"],
                  [[k, fmt(v), fmt(data["oracle_swap_offline_valid"].get(k))]
                   for k, v in data["oracle_swap_live_valid"].items()])
    t += "\n## Image-space tracking on SoccerNet-GSR valid (cached detections)\n\n"
    t += md_table(["Tracker", "HOTA", "AssA", "IDF1", "ID switches", "tracker ms/frame"],
                  [[r["name"], f"{r['hota']:.3f}", f"{r['assa']:.3f}", f"{r['idf1']:.3f}",
                    str(r["id_switches"]), f"{r['track_ms']:.2f}"] for r in data["tracker_sweep_gsr_valid"]])
    (OUT / "tables.md").write_text(t)

    # ---- figures -------------------------------------------------------------
    x = [0, 1, 5, 10, 14]
    xt = ["0", "1 s", "5 s", "10 s", "offline"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    first = data["e2e_valid_ablation"][0]["gs_hota"]
    final = data["e2e_valid_ablation"][5]["gs_hota"]
    ax.plot(x, data["e2e_test_final"], "o-", color="#1b5e20", lw=2.2, label="final system, test")
    ax.plot(x, final, "o-", color="#43a047", lw=2.0, label="final system, valid")
    ax.plot(x, first, "o--", color="#9e9e9e", lw=1.5, label="first version, valid")
    ax.set_xticks(x, xt)
    ax.set_xlabel("output delay (lookahead)")
    ax.set_ylabel("GS-HOTA")
    ax.set_title("Game-state accuracy vs output delay")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_delay_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    comp = {"jersey (gated + pose)": data["jersey_only_valid"]["gated + pose crops"],
            "team (online re-fit)": data["team_only_valid"]["online re-fit"]}
    for (k, v), c in zip(comp.items(), ("#c62828", "#1565c0")):
        pts = [(xi, vi) for xi, vi in zip(x, v) if vi is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=c, lw=2, label=k)
    ax.axhline(calib["keyframe every 10"]["gs_hota_calib_only"], color="#6a1b9a", ls=":", lw=1.8,
               label="calibration (no delay dependence)")
    ax.set_xticks(x, xt)
    ax.set_xlabel("output delay (lookahead)")
    ax.set_ylabel("component-only GS-HOTA")
    ax.set_title("Which component benefits from waiting")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_components.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sp = data["speed_stages_ms_fp32"]
    labels = ["decode", "detector", "pose", "motion+calib", "tracker", "team", "keyframe calib", "jersey"]
    vals = [sp["decode"], sp["detect"], sp["pose"], sp["motion_prop"], sp["track"], sp["team_feat"],
            sp["keyframe_calib_amortised"], data["speed_jersey_ms"]]
    ax.barh(labels[::-1], vals[::-1], color="#546e7a")
    ax.axvline(1000 / FPS, color="#c62828", ls="--", lw=1.5, label="25 FPS budget (40 ms)")
    ax.set_xlabel("ms per frame (PyTorch fp32, idle H100; stages overlap when pipelined)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stage_latency.png", dpi=160)
    plt.close(fig)
    print(f"[report] wrote {OUT}/data.json, tables.md, 3 figures")


if __name__ == "__main__":
    main()
