"""Shared helpers and the FeatureContext for plug-in features.

Standard library only (plus FastAPI's HTTPException for ``require``), so
the feature package stays importable without the heavy pipeline
dependencies (torch, cv2, numpy). Frame objects have the format from
``docs/WEBAPP_CONTRACT.md`` section 1.
"""

import json
import logging
import os
import re
import tempfile
from collections import OrderedDict

from fastapi import HTTPException

_log = logging.getLogger(__name__)

# 25 frames per second (docs/WEBAPP_CONTRACT.md).
FPS = 25
# Pitch extents in metres: x along the pitch from -PITCH_X to PITCH_X,
# y across from -PITCH_Y to PITCH_Y (contract section 1).
PITCH_X = 52.5
PITCH_Y = 34.0

# The match id format (matches.MatchStore._ID_RE); ids that do not match
# it can be rejected without any file access.
_MATCH_ID_RE = re.compile(r"m_[0-9a-f]{10}\Z")
# A feature cache name: it is interpolated into a file name, so it must
# stay a single plain path component (no '/', no leading dot, no '..').
_CACHE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# How many matches' parsed frames the context keeps in its LRU cache.
_FRAMES_CACHE_LIMIT = 4


def by_frame(frames):
    """Mapping of frame number to the frame dict."""
    return {frame["frame"]: frame for frame in frames}


def _xy(xy):
    """(x, y) when xy is a two-element list with both numbers present,
    else None (the contract: xy is null when not placed on the pitch)."""
    if isinstance(xy, (list, tuple)) and len(xy) == 2 and xy[0] is not None and xy[1] is not None:
        return xy[0], xy[1]
    return None


def people_xy(frame, team=None, roles=("player", "goalkeeper")):
    """List of (id, x, y) for the people with a pitch position.

    team: keep only that team ("left"/"right"), or None for both.
    roles: the roles to keep; referees are excluded by default. People
    whose xy is missing or null have no position and are skipped.
    """
    out = []
    for person in frame.get("people") or []:
        if not isinstance(person, dict):
            continue
        if team is not None and person.get("team") != team:
            continue
        if person.get("role") not in roles:
            continue
        pos = _xy(person.get("xy"))
        if pos is None:
            continue
        out.append((person.get("id"), pos[0], pos[1]))
    return out


def ball_xy(frame):
    """(x, y) of the ball, or None when it was not placed on the pitch
    (ball null, or ball.xy null)."""
    ball = frame.get("ball")
    if not isinstance(ball, dict):
        return None
    return _xy(ball.get("xy"))


def frame_seconds(frame_number, first_frame):
    """Seconds between two frame numbers (25 frames per second)."""
    return (frame_number - first_frame) / FPS


def rounded(obj, ndigits=2):
    """obj with every float rounded to ndigits, dicts and lists in depth.

    Integers, strings, None and booleans pass through unchanged.
    """
    if isinstance(obj, dict):
        return {key: rounded(value, ndigits) for key, value in obj.items()}
    if isinstance(obj, list):
        return [rounded(value, ndigits) for value in obj]
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj


def window_ranges(first, last, size):
    """Inclusive (start, end) frame ranges covering first..last in order.

    Every range holds at most size frames; the final range may be
    shorter. Empty when last < first or size < 1.
    """
    if last < first or size < 1:
        return []
    ranges = []
    start = first
    while start <= last:
        end = min(start + size - 1, last)
        ranges.append((start, end))
        start = end + 1
    return ranges


class FeatureContext:
    """What a feature receives in register(): the store plus accessors.

    frames(match_id) is cached in an LRU of _FRAMES_CACHE_LIMIT matches.
    An entry is dropped when its match is deleted, or when the stored
    frames file was replaced (detected from the file's inode, mtime and
    size), so the cache never serves stale frames.
    """

    def __init__(self, store):
        self.store = store
        self._frames_cache = OrderedDict()

    # -- match access ------------------------------------------------------

    def meta(self, match_id):
        """The stored meta dict, or None for unknown or malformed ids."""
        return self.store.get(match_id)

    def require(self, match_id):
        """The stored meta dict, or HTTPException(404, {"error": ...}).

        Ids not matching the match id format answer 404 without any file
        access.
        """
        meta = self.store.get(match_id)
        if meta is None:
            raise HTTPException(404, {"error": "match not found"})
        return meta

    def frames(self, match_id):
        """The parsed frame dicts of a match, or None for unknown or
        malformed ids.

        Cached in the LRU of _FRAMES_CACHE_LIMIT matches; the entry is
        dropped when the match is deleted or its frames file changed.
        """
        if not isinstance(match_id, str) or not _MATCH_ID_RE.match(match_id):
            self._frames_cache.pop(match_id, None)
            return None
        if self.store.get(match_id) is None:
            self._frames_cache.pop(match_id, None)
            return None
        try:
            st = os.stat(self.store.root / match_id / "frames.jsonl")
            stamp = (st.st_ino, st.st_mtime_ns, st.st_size)
        except OSError:
            self._frames_cache.pop(match_id, None)
            return None
        entry = self._frames_cache.get(match_id)
        if entry is not None and entry[0] == stamp:
            self._frames_cache.move_to_end(match_id)
            return entry[1]
        frames = self.store.load_frames(match_id)
        self._frames_cache[match_id] = (stamp, frames)
        self._frames_cache.move_to_end(match_id)
        while len(self._frames_cache) > _FRAMES_CACHE_LIMIT:
            self._frames_cache.popitem(last=False)
        return frames

    def analytics(self, match_id):
        """The analytics summary of a match (disk-cached by the store)."""
        return self.store.analytics(match_id)

    # -- per-feature result cache -------------------------------------------

    def cache_json(self, match_id, name, compute):
        """The JSON stored in <match>/features/<name>.json, or a fresh value.

        The stored value is served while it is not older than the
        match's frames.jsonl; otherwise compute() runs, its result is
        stored atomically (temporary file, rename) and returned. A
        stored file that cannot be read or parsed is treated as stale.
        Both match_id and name are validated before any file access:
        they are interpolated into paths, so anything that is not a
        match id or a single plain file name raises ValueError.
        """
        if not isinstance(match_id, str) or not _MATCH_ID_RE.match(match_id):
            raise ValueError("not a match id: %r" % (match_id,))
        if not isinstance(name, str) or not _CACHE_NAME_RE.match(name):
            raise ValueError("not a feature cache name: %r" % (name,))
        match_dir = self.store.root / match_id
        cache_dir = match_dir / "features"
        cache_path = cache_dir / (name + ".json")
        frames_path = match_dir / "frames.jsonl"
        try:
            if cache_path.is_file() and cache_path.stat().st_mtime_ns >= frames_path.stat().st_mtime_ns:
                return json.loads(cache_path.read_text(encoding="utf-8"))
        except OSError:
            _log.debug("the feature cache %s is unreadable; recomputing",
                       cache_path)
        except ValueError:
            _log.debug("the feature cache %s is not valid JSON; recomputing",
                       cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(cache_dir), prefix=".feature-cache-")
        try:
            value = compute()
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(value, fh)
            os.replace(tmp, cache_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError as exc:
                _log.debug("could not remove the temporary feature cache %s: %s",
                           tmp, exc)
            raise
        return value
