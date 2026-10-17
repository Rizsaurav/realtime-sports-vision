"""Command line: `python -m sideline serve` (web viewer) or `python -m sideline run` (headless)."""
import argparse
import json
import sys
import threading
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(prog="sideline", description="Live soccer minimap from a broadcast feed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the web viewer")
    s.add_argument("--config", default="configs/live.yaml")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    r = sub.add_parser("run", help="process one source headless and write per-frame JSON lines")
    r.add_argument("source", help="video file, image folder, stream or page URL")
    r.add_argument("--config", default="configs/live.yaml")
    r.add_argument("--delay", type=float, default=0.0, help="output delay in seconds")
    r.add_argument("--out", default="-", help="JSON lines file (default stdout)")
    r.add_argument("--realtime", action="store_true", help="pace a file at 25 FPS and drop late frames")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    if args.cmd == "serve":
        import uvicorn
        from .server import create_app
        srv = cfg.get("server", {})
        host, port = args.host or srv.get("host", "127.0.0.1"), args.port or srv.get("port", 8000)
        print(f"[sideline] viewer at http://{host}:{port}  (models load in the background)", flush=True)
        uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
        return

    from .pipeline import LivePipeline, Models
    from .server import compact
    delay = int(round(args.delay * 25))
    pipe = LivePipeline(Models(cfg), cfg, delays=(delay,))
    out = sys.stdout if args.out == "-" else open(args.out, "w")
    done = threading.Event()

    def sink(o):
        if o is None:
            done.set()
        else:
            out.write(json.dumps(compact(o)) + "\n")

    pipe.subscribe(sink)
    pipe.start(args.source, realtime=args.realtime)
    done.wait()
    pipe.wait()
    st = pipe.stats.snapshot()
    print(f"[sideline] {st['frames']} frames, {st['dropped']} dropped, {st['fps']} FPS, "
          f"latency p50 {st['latency_ms_p50']} ms", file=sys.stderr)
    if pipe.error:
        sys.exit(f"[sideline] error: {pipe.error}")


if __name__ == "__main__":
    main()
