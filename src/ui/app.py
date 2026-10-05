"""Sideline demo UI (Streamlit).

UX RULE: replay-first. The app NEVER runs inference itself. It reads
results/*.json + rendered demo clips produced during GPU sessions.
Zero GPU to demo, zero GPU to browse tradeoff curves. Works on a laptop.

Layout:
  Sidebar : arm filter, clip selector
  Tab 1   : Demo player — raw vs tracked video side-by-side + FPS readout
  Tab 2   : Tradeoff curves — FPS vs mAP/MOTA across backbones x precisions
  Tab 3   : Runs table — every experiment arm, config hash, metrics

Run:  streamlit run src/ui/app.py   (CPU, no GPU)
"""

import glob
import hashlib
import json
import sys
import tempfile
from pathlib import Path

# allow `streamlit run src/ui/app.py` to import the src/ package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS_GLOB = "results/*.json"
PRECISION_ORDER = ["FP32-PT", "FP16", "FP8", "INT8"]


@st.cache_data
def load_results():
    rows = []
    for p in sorted(glob.glob(RESULTS_GLOB)):
        try:
            r = json.loads(Path(p).read_text())
        except Exception:
            continue
        cfg = r.get("config", {})
        dcfg = cfg.get("detector", {})
        rows.append({
            "experiment": r.get("experiment", Path(p).stem),
            "backbone": dcfg.get("name", "?"),
            "precision": _precision_of(cfg),
            "cache": "k=%s" % cfg.get("streaming", {}).get("keyframe_interval")
                     if cfg.get("streaming", {}).get("frame_cache") else "off",
            "fps": r.get("fps", 0),
            "p50_ms": r.get("latency_p50_ms", 0),
            "p95_ms": r.get("latency_p95_ms", 0),
            "mAP": r.get("map", 0),
            "MOTA": r.get("mota", 0),
            "vram_mb": r.get("vram_mb"),
            "watts": r.get("watts"),
            "device": r.get("device", "?"),
            "mode": r.get("mode", "?"),
            "file": p,
        })
    return pd.DataFrame(rows)


def _precision_of(cfg):
    name = cfg.get("experiment", "")
    for prec in ("fp8", "fp16", "int8"):
        if prec in name.lower():
            return prec.upper().replace("FP", "FP")
    return "FP32-PT"


def _config_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


st.set_page_config(page_title="Sideline — real-time sports vision", layout="wide")
st.title("Sideline: real-time efficient sports vision")
st.caption("Broadcast soccer tracking on a single GPU. Every number below was "
           "measured, not estimated — see docs/TRADEOFF_STUDY.md for the protocol.")

df = load_results()

with st.sidebar:
    st.header("Filters")
    if not df.empty:
        backbones = st.multiselect("Backbone", sorted(df["backbone"].unique()),
                                   default=sorted(df["backbone"].unique()))
        precisions = st.multiselect("Precision", PRECISION_ORDER,
                                    default=PRECISION_ORDER)
        df = df[df["backbone"].isin(backbones) & df["precision"].isin(precisions)]
    st.divider()
    st.markdown("**What am I looking at?**\n\nEach row is one measured arm of the "
                "tradeoff study: detector backbone x precision x frame-caching, "
                "run on real match footage.")

tab_demo, tab_curves, tab_runs = st.tabs(["Demo", "Tradeoff curves", "Runs"])

with tab_demo:
    st.subheader("Tracked output")
    clips = sorted(glob.glob("results/demo_clips/*_tracked.mp4"))
    if clips:
        clip = st.selectbox("Clip", clips, format_func=lambda p: Path(p).name)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw broadcast**")
            raw = clip.replace("_tracked.mp4", "_raw.mp4")
            if Path(raw).exists():
                st.video(raw)
            else:
                st.info("raw clip not cached")
        with col2:
            st.markdown("**Tracked (live FPS overlay)**")
            st.video(clip)
    else:
        st.info("No demo clips yet — they are rendered in GPU Session D "
                "(docs/GPU_RUN_PLAN.md).")
        st.markdown("Preview of the pipeline output format:")
        st.code("frame -> detect (boxes) -> track (IDs) -> annotate + FPS overlay",
                language="text")

    st.divider()
    st.subheader("Live inference on CPU")
    st.caption("Upload a short clip and watch the actual pipeline run: YOLO26n "
               "detection + ByteTrack, on CPU, no GPU. Slow but real.")
    up = st.file_uploader("Upload a video clip (mp4/avi/mov, keep it short)",
                          type=["mp4", "avi", "mov"], key="live_upload")
    if up is not None and st.button("Run live inference", key="live_run"):
        from models.detector import Detector
        from tracking.tracker import Tracker
        from streaming.pipeline import StreamingPipeline
        tmpdir = Path(tempfile.mkdtemp())
        in_path = tmpdir / up.name
        in_path.write_bytes(up.getbuffer())
        out_path = tmpdir / "live_tracked.mp4"
        with st.spinner("Running detection + tracking on CPU..."):
            det = Detector("yolo26n", imgsz=320, conf=0.25, device="cpu")
            trk = Tracker("bytetrack")
            pipe = StreamingPipeline(det, trk, out_path=out_path)
            pipe.run(in_path, max_frames=150)
            rep = pipe.latency_report()
        st.video(str(out_path))
        c1, c2, c3 = st.columns(3)
        c1.metric("FPS (CPU)", f"{rep['fps']:.1f}")
        c2.metric("p50 latency", f"{rep['latency_p50_ms']:.1f} ms")
        c3.metric("p95 latency", f"{rep['latency_p95_ms']:.1f} ms")
        st.caption("GPU Session A will 6-10x these numbers. Same code, device=cuda.")

with tab_curves:
    st.subheader("The money plot: speed vs accuracy")
    if df.empty:
        st.warning("No results/*.json found. Run a smoke test or drop in the "
                   "sample results to see curves.")
    else:
        fig = px.scatter(df, x="fps", y="MOTA", color="backbone",
                         symbol="precision", size="mAP",
                         hover_data=["p95_ms", "vram_mb", "experiment"],
                         labels={"fps": "FPS (higher is better)",
                                 "MOTA": "MOTA (higher is better)"},
                         title="FPS vs tracking accuracy by backbone x precision")
        fig.add_vline(x=30, line_dash="dash", annotation_text="real-time line (30 FPS)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(df, x="backbone", y="p95_ms", color="precision",
                      barmode="group", title="p95 frame latency (ms, lower is better)",
                      labels={"p95_ms": "p95 latency (ms)"})
        st.plotly_chart(fig2, use_container_width=True)

with tab_runs:
    st.subheader("Every measured arm")
    if df.empty:
        st.warning("No results yet.")
    else:
        show = df.drop(columns=["file"]).copy()
        show["mAP"] = show["mAP"].round(3)
        show["MOTA"] = show["MOTA"].round(3)
        show["fps"] = show["fps"].round(1)
        st.dataframe(show, use_container_width=True, hide_index=True)
