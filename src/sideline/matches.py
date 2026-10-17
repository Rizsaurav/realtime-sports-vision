"""Match library: upload, store and serve recorded matches.

Implements the ``/api/matches`` endpoints from ``docs/WEBAPP_CONTRACT.md``.
Only the standard library plus FastAPI/Starlette is used. The ``analytics``
module is imported lazily, inside the method that needs it, so this module
stays importable without the heavy pipeline dependencies.
"""

import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

_ID_RE = re.compile(r"m_[0-9a-f]{10}\Z")
_NUM_RE = re.compile(r"[0-9]+\Z")
_DEFAULT_MAX_BYTES = 100 * 1024 * 1024
_FPS = 25
_log = logging.getLogger(__name__)
_PLAYERS_CSV_KEYS = ("id", "team", "jersey", "role", "frames", "seconds",
                     "distance_m", "avg_speed_ms", "max_speed_ms", "sprints")


class _RequestError(Exception):
    """A request rejected with a 400/404/413 and a client-facing message."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _display_name(name, filename):
    if name is not None and name.strip():
        return name.strip()
    stem = Path(filename).stem if filename else ""
    return stem if stem else "match"


def _parse_frames(raw):
    """Validate the uploaded bytes and return the list of frame objects."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise _RequestError(400, "file is not valid UTF-8")
    frames = []
    prev = None
    for lineno, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            raise _RequestError(400, "line %d: invalid JSON" % lineno)
        if not isinstance(obj, dict):
            raise _RequestError(400, "line %d: not a JSON object" % lineno)
        if not _is_int(obj.get("frame")):
            raise _RequestError(400, "line %d: frame must be an integer" % lineno)
        if not isinstance(obj.get("people"), list):
            raise _RequestError(400, "line %d: people must be a list" % lineno)
        if prev is not None and obj["frame"] <= prev:
            raise _RequestError(400, "line %d: frame must be strictly increasing" % lineno)
        prev = obj["frame"]
        frames.append(obj)
    if not frames:
        raise _RequestError(400, "file contains no frames")
    return frames


class MatchStore:
    def __init__(self, root, max_bytes=_DEFAULT_MAX_BYTES, smooth=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        # smooth: when True, the cached analytics summary is computed with
        # analytics.summarize's jitter smoothing switched on (player
        # positions); when False (the default) the summary uses the
        # recorded positions.
        self.smooth = smooth
        self._lock = threading.RLock()

    # -- paths -------------------------------------------------------------

    def _dir(self, match_id):
        return self.root / match_id

    def _meta_path(self, match_id):
        return self._dir(match_id) / "meta.json"

    @staticmethod
    def _atomic_write(path, data):
        path = Path(path)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
        try:
            with open(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            try:
                Path(tmp).unlink()
            except OSError as exc:
                _log.warning("could not remove temp file %s: %s", tmp, exc)
            raise

    # -- basic access ------------------------------------------------------

    def get(self, match_id):
        """Meta dict for a stored match, or None for unknown/malformed ids."""
        if not isinstance(match_id, str) or not _ID_RE.match(match_id):
            return None
        path = self._meta_path(match_id)
        if not path.is_file():
            return None
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return meta if isinstance(meta, dict) else None

    def list(self):
        """All stored meta objects, newest first (ties by id descending)."""
        items = []
        if self.root.is_dir():
            for entry in self.root.iterdir():
                if entry.is_dir() and _ID_RE.match(entry.name):
                    meta = self.get(entry.name)
                    if meta is not None:
                        items.append(meta)
        items.sort(key=lambda m: (m.get("created_at", ""), m.get("id", "")), reverse=True)
        return items

    # -- upload ------------------------------------------------------------

    def add(self, raw, name, source=None):
        """Store a match from the uploaded bytes.

        Returns ``(meta, created)`` where ``created`` is False when a match
        with the same id was already stored (nothing is changed then).
        Raises _RequestError(413/400) for rejected uploads; rejected uploads
        store nothing.
        """
        with self._lock:
            if len(raw) > self.max_bytes:
                raise _RequestError(413, "file exceeds the maximum size of %d bytes" % self.max_bytes)
            frames = _parse_frames(raw)
            meta = self._build_meta(raw, frames, name, source)
            meta_path = self._meta_path(meta["id"])
            if meta_path.is_file():
                return json.loads(meta_path.read_text(encoding="utf-8")), False
            self._dir(meta["id"]).mkdir(parents=True, exist_ok=True)
            self._atomic_write(self._dir(meta["id"]) / "frames.jsonl", raw)
            self._atomic_write(meta_path, json.dumps(meta, indent=2).encode("utf-8") + b"\n")
            return meta, True

    @staticmethod
    def _build_meta(raw, frames, name, source):
        first, last = frames[0], frames[-1]
        delay = first.get("delay")
        if not _is_int(delay):
            delay = 0
        colors = None
        for frame in frames:
            if frame.get("colors") is not None:
                colors = frame.get("colors")
        meta = {
            "id": "m_" + hashlib.sha256(raw).hexdigest()[:10],
            "name": name,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "frame_count": len(frames),
            "first_frame": first["frame"],
            "last_frame": last["frame"],
            "duration_s": round((last["frame"] - first["frame"] + 1) / _FPS, 2),
            "delay": delay,
            "colors": colors,
            "size_bytes": len(raw),
        }
        if source is not None and source.strip():
            meta["source"] = source.strip()
        return meta

    # -- reading -----------------------------------------------------------

    def load_frames(self, match_id):
        path = self._dir(match_id) / "frames.jsonl"
        text = path.read_text(encoding="utf-8")
        frames = []
        for line in text.split("\n"):
            if line.strip():
                frames.append(json.loads(line))
        return frames

    def frames(self, match_id, start, end, step):
        """Frames whose number lies in [start, end], thinned to every step-th."""
        selected = [f for f in self.load_frames(match_id) if start <= f["frame"] <= end]
        return {"frames": selected[::step], "total": len(selected)}

    def analytics(self, match_id):
        """Cached analytics summary; computed on the first request only.

        The summary is stored in ``analytics.json`` next to a
        ``analytics.json.mode`` sidecar that records the ``smooth`` flag it
        was produced with.  A cached summary is served only when that sidecar
        exists and records this store's flag; a cache from a store of the
        other mode (a plain matches router sharing a smoothed root, or a
        smoothed store over a pre-upgrade library) - or one with no recorded
        mode at all - is recomputed, so a store never serves a summary
        computed in the other mode.

        Computed with the jitter smoothing of analytics.summarize when the
        store was created with smooth=True.
        """
        with self._lock:
            path = self._dir(match_id) / "analytics.json"
            if path.is_file():
                cached = self._cached_summary(path)
                if cached is not None:
                    return cached
            from . import analytics
            frames = self.load_frames(match_id)
            if self.smooth:
                summary = analytics.summarize(frames, fps=_FPS, smooth=True)
            else:
                summary = analytics.summarize(frames, fps=_FPS)
            self._atomic_write(path, json.dumps(summary, indent=2).encode("utf-8") + b"\n")
            self._atomic_write(path.parent / "analytics.json.mode",
                               (json.dumps({"smooth": self.smooth}) + "\n").encode("utf-8"))
            return summary

    def _cached_summary(self, path):
        """The cached summary for a match, or None when the cache is not valid
        for this store: the document or the ``analytics.json.mode`` sidecar is
        missing, unreadable or not a JSON object, or the recorded ``smooth``
        flag is not a bool equal to ``self.smooth``.
        """
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            mode = json.loads(path.parent.joinpath("analytics.json.mode").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(summary, dict) or not isinstance(mode, dict):
            return None
        smooth = mode.get("smooth")
        if not isinstance(smooth, bool) or smooth is not self.smooth:
            return None
        return summary

    def positions_csv(self, match_id):
        rows = [["frame", "id", "team", "jersey", "role", "x", "y"]]
        for frame in self.load_frames(match_id):
            for person in frame.get("people") or []:
                if not isinstance(person, dict):   # a non-object has no xy, so no row
                    continue
                xy = person.get("xy")
                if not isinstance(xy, list) or len(xy) != 2 or xy[0] is None or xy[1] is None:
                    continue
                rows.append([frame["frame"], person.get("id"), person.get("team"),
                             person.get("jersey"), person.get("role"), xy[0], xy[1]])
        return self._csv_text(rows)

    def players_csv(self, match_id):
        rows = [list(_PLAYERS_CSV_KEYS)]
        for player in self.analytics(match_id).get("players") or []:
            rows.append([player.get(key) for key in _PLAYERS_CSV_KEYS])
        return self._csv_text(rows)

    @staticmethod
    def _csv_text(rows):
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerows(rows)
        return buf.getvalue()

    # -- delete ------------------------------------------------------------

    def delete(self, match_id):
        """Remove the match folder; True when a match was removed."""
        if not isinstance(match_id, str) or not _ID_RE.match(match_id):
            return False
        with self._lock:
            if not self._meta_path(match_id).is_file():
                return False
            shutil.rmtree(self._dir(match_id))
            return True


def _int_param(value, what, default):
    if value is None:
        return default
    if not _NUM_RE.match(value):
        raise _RequestError(400, "%s must be a non-negative integer" % what)
    return int(value)


def _not_found():
    return JSONResponse({"error": "match not found"}, status_code=404)


def create_router(store):
    router = APIRouter()

    @router.post("/api/matches")
    async def create_match(request: Request):
        form = await request.form()
        try:
            file = form.get("file")
            name = form.get("name")
            source = form.get("source")
            if not isinstance(file, UploadFile):
                return JSONResponse({"error": "file field is required"}, status_code=400)
            raw = await file.read(store.max_bytes + 1)
        finally:
            await form.close()
        try:
            meta, created = store.add(raw, _display_name(name, file.filename), source)
        except _RequestError as exc:
            return JSONResponse({"error": exc.message}, status_code=exc.status)
        return JSONResponse(meta, status_code=201 if created else 200)

    @router.get("/api/matches")
    def list_matches():
        return {"items": store.list()}

    @router.get("/api/matches/{match_id}")
    def get_match(match_id: str):
        meta = store.get(match_id)
        if meta is None:
            return _not_found()
        return meta

    @router.get("/api/matches/{match_id}/frames")
    def get_frames(match_id: str, start: Optional[str] = None,
                   end: Optional[str] = None, step: Optional[str] = None):
        meta = store.get(match_id)
        if meta is None:
            return _not_found()
        try:
            start_v = _int_param(start, "start", meta["first_frame"])
            end_v = _int_param(end, "end", meta["last_frame"])
            step_v = _int_param(step, "step", 1)
            if step_v < 1:
                raise _RequestError(400, "step must be at least 1")
            if start_v > end_v:
                raise _RequestError(400, "start must not be greater than end")
        except _RequestError as exc:
            return JSONResponse({"error": exc.message}, status_code=exc.status)
        return store.frames(match_id, start_v, end_v, step_v)

    @router.get("/api/matches/{match_id}/analytics")
    def get_analytics(match_id: str):
        if store.get(match_id) is None:
            return _not_found()
        return store.analytics(match_id)

    @router.get("/api/matches/{match_id}/export.csv")
    def export_csv(match_id: str, kind: Optional[str] = None):
        if store.get(match_id) is None:
            return _not_found()
        kind = "positions" if kind is None else kind
        if kind not in ("positions", "players"):
            return JSONResponse({"error": "kind must be 'positions' or 'players'"}, status_code=400)
        text = store.positions_csv(match_id) if kind == "positions" else store.players_csv(match_id)
        return Response(content=text.encode("utf-8"), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="%s-%s.csv"' % (match_id, kind)})

    @router.delete("/api/matches/{match_id}")
    def delete_match(match_id: str):
        if not store.delete(match_id):
            return _not_found()
        return Response(status_code=204)

    return router
