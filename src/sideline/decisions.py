"""Streaming attribute decisions with a bounded output delay.

The engine takes one frame of perception at a time (tracked boxes with pitch
positions, torso colour features and optional jersey evidence) and, once
frame t + L has arrived, emits the final state of frame t. Every decision uses
only evidence up to the newest frame:

  stitching  re-link a new track id to a recently lost one that is reachable
             at running speed in pitch space and has a similar kit colour
  role       majority detector class of the track over (t - window, t + L]
  team       two-way torso colour clusters re-fitted every `refit_every`
             frames; side from the clusters' mean pitch x; goalkeepers by half
  jersey     legibility-weighted per-track number evidence, committed once it
             passes tau and the ratio test; shown from the commit frame on
  linking    a track that outputs (team, jersey) owned by an identity not
             visible in this frame takes over that identity

This is the incremental form of scripts/gsr_assemble.py. One difference is
deliberate: a row enters the team clusters with the role known when it arrives
(evidence up to that frame), where the batch code used its delayed role.
"""
from collections import Counter, deque

import numpy as np

from .geometry import kmeans2, kmeans2_init

PEOPLE = ("player", "goalkeeper")


class Stitcher:
    def __init__(self, max_gap=150, v_max=0.4, d0=2.0, feat_thr=3.0):
        self.max_gap, self.v_max, self.d0, self.feat_thr = max_gap, v_max, d0, feat_thr
        self.canon, self.last, self.first_seen = {}, {}, set()
        self.n_rows = 0

    @staticmethod
    def _group(cls):
        return "ref" if cls == "referee" else "pl"

    def update(self, f, rows):
        """rows of one frame -> canonical track id per row."""
        canon, last = self.canon, self.last
        visible = {canon.get(r["track"], r["track"]) for r in rows if r["track"] in self.first_seen}
        out = []
        for r in rows:
            t = r["track"]
            if t not in self.first_seen:
                self.first_seen.add(t)
                best, bc = None, 1e9
                if r["pitch"] is not None:
                    x, y = r["pitch"][2], r["pitch"][3]
                    for c, (lf, lx, ly, lg, lfeat) in last.items():
                        gap = f - lf
                        if c in visible or gap <= 0 or gap > self.max_gap or lg != self._group(r["cls"]):
                            continue
                        dist = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
                        reach = self.d0 + self.v_max * gap
                        if dist > reach:
                            continue
                        fd = (np.linalg.norm((np.asarray(r["feat"]) - lfeat) / (np.abs(lfeat) + 10))
                              if r["feat"] is not None and lfeat is not None else 0.0)
                        if fd > self.feat_thr:
                            continue
                        cost = dist / reach + fd / self.feat_thr + gap / self.max_gap
                        if cost < bc:
                            best, bc = c, cost
                if best is not None:
                    canon[t] = best
                    visible.add(best)
            out.append(canon.get(t, t))
        # a revived id that was already handed to a new fragment keeps its own id
        used = {}
        for j in sorted(range(len(rows)), key=lambda j: rows[j]["track"] != out[j]):
            c = out[j]
            if c in used:
                t = rows[j]["track"]
                canon[t] = t
                c = t if t not in used else 5_000_000 + self.n_rows + j
                out[j] = c
            used[c] = j
        for r, c in zip(rows, out):
            if r["pitch"] is not None:
                prev = last.get(c)
                feat = np.asarray(r["feat"]) if r["feat"] is not None else None
                if prev is not None and prev[4] is not None and feat is not None:
                    feat = 0.9 * prev[4] + 0.1 * feat
                elif prev is not None and feat is None:
                    feat = prev[4]
                last[c] = (f, r["pitch"][2], r["pitch"][3], self._group(r["cls"]), feat)
        self.n_rows += len(rows)
        if len(last) > 512:                     # forget identities lost for longer than max_gap
            for c in [c for c, v in last.items() if f - v[0] > self.max_gap]:
                del last[c]
        return out


class RoleVote:
    """Per track: counter over the classes of rows in (t - window, t + L]."""

    def __init__(self, window=25):
        self.window = window
        self.hist = {}                          # track -> deque of (frame, cls) not yet removed
        self.cnt = {}

    def add(self, track, f, cls):
        self.hist.setdefault(track, deque()).append((f, cls))
        self.cnt.setdefault(track, Counter())[cls] += 1

    def now(self, track, f):
        """Role from (f - window, f] only, for rows entering the team clusters at arrival."""
        c = Counter(cls for g, cls in self.hist.get(track, ()) if g > f - self.window) if self.window \
            else self.cnt.get(track, Counter())
        return c.most_common(1)[0][0] if c else None

    def at(self, track, t, cls):
        """Role of the track's row at frame t, with every row up to t + L already added."""
        cnt, h = self.cnt[track], self.hist[track]
        if self.window:
            while h and h[0][0] <= t - self.window:
                cnt[h.popleft()[1]] -= 1
        return cnt.most_common(1)[0][0] if sum(cnt.values()) > 0 else cls


class TeamModel:
    """Two-way kit colour clusters; the cluster whose players have the lower mean pitch x
    over the last `side_window` frames is "left". The window (3 minutes by default) is
    longer than any benchmark clip, where this equals the all-history batch rule, and lets
    a long stream follow the half-time switch of ends."""

    def __init__(self, warmup=25, refit_every=25, max_fit=4000, seed=0, side_window=4500,
                 max_seen=50000):
        self.warmup, self.refit_every, self.max_fit = warmup, refit_every, max_fit
        self.side_window, self.max_seen = side_window, max_seen
        self.cap = 1 << 17                      # recent (frame, track row, x) for the side rule
        self.buf_f = np.zeros(self.cap, np.int64)
        self.buf_r = np.zeros(self.cap, np.int64)
        self.buf_x = np.zeros(self.cap)
        self.n_buf = 0
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.seen = []
        self.row = {}                           # track -> row in the per-track arrays
        self.fsum = np.zeros((64, 6))           # per track: sum of features, count
        self.fcnt = np.zeros(64)
        self.first = None
        self.C = self.mu = self.sd = None
        self.last_fit = None
        self.left = None

    def colors(self):
        """Kit colour of each side as '#rrggbb', from the cluster centres (CIELAB means)."""
        if self.C is None or self.left is None:
            return None
        import cv2
        out = {}
        for k in (0, 1):
            lab = (self.C[k] * self.sd + self.mu)[:3]
            rgb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB)[0, 0]
            out["left" if k == self.left else "right"] = "#%02x%02x%02x" % tuple(int(v) for v in rgb)
        return out

    def add(self, f, track, role, x, feat):
        if self.first is None:
            self.first = f
        if role == "player":
            self.seen.append(feat)
            if len(self.seen) > 2 * self.max_seen:      # long streams: keep the recent features
                del self.seen[: len(self.seen) - self.max_seen]
            i = self.row.get(track)
            if i is None:
                i = self.row[track] = len(self.row)
                if i == len(self.fcnt):
                    self.fsum = np.concatenate([self.fsum, np.zeros_like(self.fsum)])
                    self.fcnt = np.concatenate([self.fcnt, np.zeros_like(self.fcnt)])
            self.fsum[i] += feat
            self.fcnt[i] += 1
            j = self.n_buf % self.cap
            self.buf_f[j], self.buf_r[j], self.buf_x[j] = f, i, x
            self.n_buf += 1

    def assign(self, t, horizon, rows):
        """rows: (key, track, role, x, feat) of frame t -> {key: 'left'|'right'}."""
        if self.first is None or horizon < self.first + self.warmup or len(self.seen) < 4:
            return {}
        if self.C is None or t - self.last_fit >= self.refit_every:
            X = np.asarray(self.seen)
            if len(X) > self.max_fit:
                X = X[self.rng.choice(len(X), self.max_fit, replace=False)]
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
            Z = (X - self.mu) / self.sd
            self.C = kmeans2(Z, seed=self.seed) if self.C is None else kmeans2_init(Z, self.C)
            self.last_fit = t
        cl = lambda v: int(np.argmin((((v - self.mu) / self.sd - self.C) ** 2).sum(-1)))
        n = len(self.row)
        Z = (self.fsum[:n] / self.fcnt[:n, None] - self.mu) / self.sd
        lab = np.argmin(((Z[:, None] - self.C[None]) ** 2).sum(-1), 1)
        m = min(self.n_buf, self.cap)
        keep = self.buf_f[:m] > horizon - self.side_window
        k_rows, x_rows = lab[self.buf_r[:m][keep]], self.buf_x[:m][keep]
        mx = [x_rows[k_rows == k].mean() if (k_rows == k).any() else 0.0 for k in (0, 1)]
        left = int(np.argmin(mx))
        self.left = left
        out = {}
        for key, track, role, x, feat in rows:
            if role == "player":
                k = int(lab[self.row[track]]) if track in self.row else cl(feat)
                out[key] = "left" if k == left else "right"
            else:
                out[key] = "left" if x < 0 else "right"
        return out


class JerseyVote:
    def __init__(self, tau=4.0, ratio=0.6):
        self.tau, self.ratio = tau, ratio
        self.ev, self.committed = {}, {}

    def add(self, track, P, Pnone):
        self.ev[track] = self.ev.get(track, 0) + (1 - Pnone) * P

    def commit(self, t):
        for tr, s in self.ev.items():
            if tr not in self.committed:
                k = int(s.argmax())
                if s[k] >= self.tau and s[k] >= self.ratio * s.sum():
                    self.committed[tr] = (t, str(k))

    def get(self, track, t):
        c = self.committed.get(track)
        return c[1] if c and t >= c[0] else None

    def evidence(self, track):
        s = self.ev.get(track)
        if s is None:
            return None
        k = int(s.argmax())
        return str(k), float(s[k])


class NumberLink:
    def __init__(self):
        self.owner, self.remap = {}, {}

    def update(self, preds):
        active = {self.remap.get(p["track_id"], p["track_id"]) for p in preds}
        for p in preds:
            a, t = p["attributes"], p["track_id"]
            cur = self.remap.get(t, t)
            if a["jersey"] is not None and a["team"] is not None and a["role"] in PEOPLE:
                key = (a["team"], a["jersey"])
                o = self.owner.get(key)
                if o is None:
                    self.owner[key] = cur
                elif o != cur and o not in active:
                    self.remap[t] = o
                    active.discard(cur)
                    active.add(o)
                    cur = o
            p["track_id"], p["_raw"] = cur, t
        used = {}
        for p in sorted(preds, key=lambda p: p["_raw"] != p["track_id"]):
            if p["track_id"] in used:
                self.remap.pop(p["_raw"], None)
                p["track_id"] = p["_raw"] if p["_raw"] not in used else 6_000_000 + len(used)
            used[p["track_id"]] = 1
        for p in preds:
            p.pop("_raw")
        return preds


class DecisionEngine:
    """push(frame) -> list of finished frames; flush() at the end of the stream.

    A frame is a dict: {"frame": int, "image_id": any, "rows": [row, ...],
    "ball": optional (x, y) pitch position or None}. A row is a dict with
    track, cls, bbox (x, y, w, h), pitch (6 floats or None), feat (6 floats or
    None) and optionally jersey = (P[100], Pnone) evidence for that box.
    """

    def __init__(self, delay=0, tau=4.0, ratio=0.6, role_window=25, team_warmup=25,
                 stitch=None, link_numbers=True, side_window=4500):
        self.L = int(delay)
        self.stitch = Stitcher(**(stitch or {})) if stitch is not False else None
        self.roles = RoleVote(role_window)
        self.team = TeamModel(warmup=team_warmup, side_window=side_window)
        self.jersey = JerseyVote(tau, ratio)
        self.link = NumberLink() if link_numbers else None
        self.pending = deque()                  # frames waiting for their delay to pass
        self.newest = None

    def push(self, fr):
        f = fr["frame"]
        rows = [dict(r) for r in fr["rows"] if r["cls"] != "ball"]
        ids = self.stitch.update(f, rows) if self.stitch else [r["track"] for r in rows]
        has_jersey = False
        for r, c in zip(rows, ids):
            r["raw_track"], r["track"] = r["track"], c
            self.roles.add(c, f, r["cls"])
            if r.get("jersey") is not None:
                self.jersey.add(c, *r["jersey"])
                has_jersey = True
        for r in rows:
            if r["feat"] is not None:
                role_now = self.roles.now(r["track"], f)
                if role_now in PEOPLE:
                    self.team.add(f, r["track"], role_now, r["pitch"][2] if r["pitch"] else 0.0,
                                  np.asarray(r["feat"]))
        self.pending.append({**fr, "rows": rows, "has_jersey": has_jersey})
        self.newest = f
        out = []
        while self.pending and self.pending[0]["frame"] <= f - self.L:
            out.append(self._emit(self.pending.popleft(), f))
        return out

    def flush(self):
        out = []
        while self.pending:
            out.append(self._emit(self.pending.popleft(), self.newest))
        return out

    def _emit(self, fr, horizon):
        t = fr["frame"]
        rows = fr["rows"]
        roles = [self.roles.at(r["track"], t, r["cls"]) for r in rows]
        team_rows = [(j, r["track"], roles[j], r["pitch"][2] if r["pitch"] else 0.0, np.asarray(r["feat"]))
                     for j, r in enumerate(rows) if roles[j] in PEOPLE and r["feat"] is not None]
        teams = self.team.assign(t, horizon, team_rows) if team_rows else {}
        if fr["has_jersey"]:
            self.jersey.commit(t)
        preds = []
        for j, r in enumerate(rows):
            if r["pitch"] is None:
                continue
            role = roles[j]
            jersey = self.jersey.get(r["track"], t) if role in PEOPLE else None
            preds.append({"track_id": r["track"], "raw_track": r["raw_track"], "bbox": r["bbox"],
                          "pitch": r["pitch"], "score": r.get("score"),
                          "attributes": {"role": role, "jersey": jersey,
                                         "team": teams.get(j) if role in PEOPLE else None}})
        if self.link is not None:
            self.link.update(preds)
        return {"frame": t, "image_id": fr.get("image_id"), "ball": fr.get("ball"),
                "delay": self.L, "preds": preds, "team_colors": self.team.colors()}
