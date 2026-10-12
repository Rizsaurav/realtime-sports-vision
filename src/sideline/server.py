"""Web server for the live minimap.

  GET  /                 the viewer (src/sideline/web)
  GET  /api/sources      clips and videos found in the configured media folders
  POST /api/start        {"source": path or URL, "loop": bool}
  POST /api/stop
  GET  /api/status       pipeline statistics and the current source
  WS   /ws               per frame: one binary message = 4-byte JSON length, JSON state, JPEG preview.
                         The client sends {"delay": frames} to switch output delay.

The pipeline computes every configured delay at once, so switching is instant;
a viewer watching delay L sees frame t once frame t + L has been processed.
"""
import asyncio
import json
import struct
import threading
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import LivePipeline, Models

WEB = Path(__file__).resolve().parent / "web"
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts"}


def compact(out):
    """Engine output -> what the viewer draws."""
    people = []
    for p in out["preds"]:
        a = p["attributes"]
        x, y, w, h = p["bbox"]
        people.append({"id": int(p["track_id"]), "role": a["role"], "team": a["team"], "jersey": a["jersey"],
                       "box": [round(x), round(y), round(w), round(h)],
                       "xy": [round(p["pitch"][2], 2), round(p["pitch"][3], 2)]})
    ball = out.get("ball")
    return {"frame": out["frame"], "delay": out["delay"], "people": people,
            "ball": None if ball is None else {"box": [round(v) for v in ball["bbox"]],
                                               "xy": None if ball["pitch"] is None
                                               else [round(ball["pitch"][0], 2), round(ball["pitch"][1], 2)]},
            "colors": out.get("team_colors")}


class Hub:
    """Fans pipeline outputs out to websocket clients; each client keeps only the newest frame."""

    def __init__(self):
        self.clients = {}                        # websocket -> {"delay": int, "queue": asyncio.Queue}
        self.loop = None
        self.lock = threading.Lock()

    def attach(self, pipe):
        pipe.subscribe(lambda out: self.publish(pipe, out))

    def publish(self, pipe, out):
        if self.loop is None:
            return
        if out is None:
            msg = ("end", None)
            targets = list(self.clients.values())
        else:
            with self.lock:
                targets = [c for c in self.clients.values() if c["delay"] == out["delay"]]
            if not targets:
                return
            jpg = pipe.preview(out["frame"])
            if jpg is None:
                return
            state = compact(out)
            state["stats"] = pipe.stats.snapshot()
            head = json.dumps(state, separators=(",", ":")).encode()
            msg = ("frame", struct.pack("<I", len(head)) + head + jpg)
        for c in targets:
            self.loop.call_soon_threadsafe(self._offer, c["queue"], msg)

    @staticmethod
    def _offer(q, msg):
        while q.full():
            q.get_nowait()
        q.put_nowait(msg)


def create_app(cfg, root="."):
    root = Path(root)
    app = FastAPI(title="Sideline")
    state = {"models": None, "pipe": None, "source": None, "loading": True, "error": None, "started": None}
    hub = Hub()
    delays = cfg.get("decisions", {}).get("delays", [0, 25, 125, 250])

    def load():
        try:
            m = Models(cfg)
            pipe = LivePipeline(m, cfg, delays=delays)
            hub.attach(pipe)
            state.update(models=m, pipe=pipe)
        except Exception as e:
            import traceback
            traceback.print_exc()
            state["error"] = f"model loading failed: {e}"
        state["loading"] = False

    @app.on_event("startup")
    async def startup():
        hub.loop = asyncio.get_running_loop()
        threading.Thread(target=load, daemon=True).start()

    @app.on_event("shutdown")
    async def shutdown():
        if state["pipe"] is not None:
            await asyncio.to_thread(state["pipe"].stop)
        if state["models"] is not None and state["models"].calib is not None:
            state["models"].calib.close()

    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/api/sources")
    async def sources():
        found = []
        for d in cfg.get("server", {}).get("media_dirs", []):
            d = (root / d) if not Path(d).is_absolute() else Path(d)
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if (p / "img1").is_dir():
                    found.append({"label": f"{d.name}/{p.name}", "path": str(p), "kind": "clip"})
                elif p.suffix.lower() in VIDEO_EXT:
                    found.append({"label": p.name, "path": str(p), "kind": "video"})
        return found

    @app.post("/api/start")
    async def start(body: dict):
        pipe = state["pipe"]
        if pipe is None:
            return JSONResponse({"error": state["error"] or "models are still loading"}, status_code=503)
        src = str(body.get("source", "")).strip()
        if not src:
            return JSONResponse({"error": "no source given"}, status_code=400)
        try:
            await asyncio.to_thread(pipe.start, src, True, bool(body.get("loop", False)))   # real-time pacing
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        state.update(source=src, started=time.time(), error=None)
        return {"ok": True, "source": src}

    @app.post("/api/stop")
    async def stop():
        if state["pipe"] is not None:
            await asyncio.to_thread(state["pipe"].stop)
        state["source"] = None
        return {"ok": True}

    @app.get("/api/status")
    async def status():
        pipe = state["pipe"]
        running = bool(pipe and pipe.alive)
        return {"loading": state["loading"], "error": state["error"] or (pipe.error if pipe else None),
                "running": running, "source": state["source"] if running else None,
                "delays": delays, "fps_target": 25,
                "stats": pipe.stats.snapshot() if pipe else None}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        client = {"delay": delays[0], "queue": asyncio.Queue(maxsize=2)}
        with hub.lock:
            hub.clients[sock] = client

        async def receive():
            while True:
                msg = json.loads(await sock.receive_text())
                if "delay" in msg and int(msg["delay"]) in delays:
                    client["delay"] = int(msg["delay"])
                    while not client["queue"].empty():
                        client["queue"].get_nowait()

        rx = asyncio.create_task(receive())
        try:
            while not rx.done():
                try:
                    kind, data = await asyncio.wait_for(client["queue"].get(), 1.0)
                except asyncio.TimeoutError:
                    continue
                if kind == "end":
                    await sock.send_text(json.dumps({"event": "end"}))
                else:
                    await sock.send_bytes(data)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            rx.cancel()
            with hub.lock:
                hub.clients.pop(sock, None)

    return app
