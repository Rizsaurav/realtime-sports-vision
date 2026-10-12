"""Streaming decision engine: delay semantics, jersey commits, teams, stitching, linking."""
import numpy as np

from sideline.decisions import DecisionEngine, NumberLink, Stitcher

RED = [120.0, 170.0, 150.0, 10.0, 5.0, 5.0]      # CIELAB mean + std of a torso crop
WHITE = [240.0, 128.0, 128.0, 8.0, 3.0, 3.0]


def row(track, x, y=0.0, cls="player", feat=None, jersey=None):
    return {"track": track, "cls": cls, "score": 0.9, "bbox": [100.0 * track, 100.0, 40.0, 90.0],
            "pitch": [x - 0.3, y, x, y, x + 0.3, y], "feat": feat, "jersey": jersey}


def number(n, p=0.99):
    P = np.full(100, (1 - p) / 99, np.float32)
    P[n] = p
    return P


def frames(n, make):
    return [{"frame": f, "image_id": f, "rows": make(f)} for f in range(n)]


def run(engine, frs):
    out = []
    for fr in frs:
        out += engine.push(fr)
    return out, engine.flush()


def test_frame_t_is_emitted_once_t_plus_delay_arrives():
    eng = DecisionEngine(delay=5, stitch=False)
    emitted = []
    for fr in frames(20, lambda f: [row(1, 0.0, feat=RED)]):
        emitted.append([o["frame"] for o in eng.push(fr)])
    assert emitted[4] == [] and emitted[5] == [0] and emitted[19] == [14]
    assert [o["frame"] for o in eng.flush()] == list(range(15, 20))


def test_jersey_shown_from_commit_frame_and_never_before():
    # legible from frame 10 on; tau = 4 needs about 4 confident reads
    make = lambda f: [row(1, 0.0, feat=RED, jersey=(number(10), 0.0) if f >= 10 else None)]
    live, rest = run(DecisionEngine(delay=0, stitch=False), frames(40, make))
    seq = [o["preds"][0]["attributes"]["jersey"] for o in live + rest]
    first = seq.index("10")
    assert all(j is None for j in seq[:first]) and all(j == "10" for j in seq[first:])
    assert 12 <= first <= 14
    delayed, rest = run(DecisionEngine(delay=25, stitch=False), frames(40, make))
    seq_d = [o["preds"][0]["attributes"]["jersey"] for o in delayed + rest]
    assert seq_d.index("10") < first                      # waiting lets earlier frames carry the number


def test_illegible_reads_carry_no_evidence():
    make = lambda f: [row(1, 0.0, feat=RED, jersey=(number(7), 1.0))]   # Pnone = 1
    live, rest = run(DecisionEngine(delay=0, stitch=False), frames(60, make))
    assert all(o["preds"][0]["attributes"]["jersey"] is None for o in live + rest)


def test_teams_follow_kit_colour_and_side_of_pitch():
    def make(f):
        rs = [row(i, -20.0 - i, feat=RED) for i in range(1, 6)]
        rs += [row(10 + i, 20.0 + i, feat=WHITE) for i in range(1, 6)]
        rs.append(row(99, -50.0, cls="goalkeeper", feat=[60.0, 128, 128, 5, 2, 2]))
        rs.append(row(98, 0.0, cls="referee", feat=[200.0, 120, 190, 5, 2, 2]))
        return rs
    live, rest = run(DecisionEngine(delay=0, stitch=False), frames(60, make))
    last = {p["raw_track"]: p["attributes"] for p in rest[-1]["preds"]} if rest else \
        {p["raw_track"]: p["attributes"] for p in live[-1]["preds"]}
    assert all(last[i]["team"] == "left" for i in range(1, 6))
    assert all(last[10 + i]["team"] == "right" for i in range(1, 6))
    assert last[99]["team"] == "left" and last[99]["role"] == "goalkeeper"
    assert last[98]["team"] is None and last[98]["role"] == "referee"
    colors = live[-1]["team_colors"]
    assert colors["left"] != colors["right"]


def test_no_team_before_warmup():
    make = lambda f: [row(1, -10.0, feat=RED), row(2, 10.0, feat=WHITE), row(3, -12.0, feat=RED),
                      row(4, 12.0, feat=WHITE)]
    live, _ = run(DecisionEngine(delay=0, team_warmup=25, stitch=False), frames(40, make))
    assert all(p["attributes"]["team"] is None for o in live[:25] for p in o["preds"])
    assert all(p["attributes"]["team"] is not None for o in live[26:] for p in o["preds"])


def test_stitching_relinks_a_broken_track_within_reach():
    st = Stitcher(max_gap=150, v_max=0.4, d0=2.0, feat_thr=3.0)
    assert st.update(0, [row(1, 0.0, feat=RED)]) == [1]
    assert st.update(1, [row(1, 0.5, feat=RED)]) == [1]
    # track 1 lost for 10 frames, new id 2 appears 3 m away (reach = 2 + 0.4 * 10 = 6 m)
    assert st.update(11, [row(2, 3.5, feat=RED)]) == [1]
    # an id far away at running speed stays separate
    assert st.update(12, [row(2, 3.6, feat=RED), row(3, 40.0, feat=RED)]) == [1, 3]


def test_stitching_rejects_a_strongly_different_kit():
    # the colour distance is relative to the lost track's own feature, so the gate
    # (feat_thr = 3) separates contrasting kits such as dark versus white
    dark = [12.0, 128.0, 128.0, 2.0, 2.0, 2.0]
    st = Stitcher()
    st.update(0, [row(1, 0.0, feat=dark)])
    assert st.update(5, [row(2, 0.5, feat=WHITE)]) == [2]


def test_number_link_hands_identity_to_new_track():
    link = NumberLink()
    a = {"track_id": 1, "attributes": {"role": "player", "team": "left", "jersey": "9"}}
    link.update([a])
    b = {"track_id": 7, "attributes": {"role": "player", "team": "left", "jersey": "9"}}
    link.update([b])                                       # track 1 not visible: 7 becomes identity 1
    assert b["track_id"] == 1
    c = {"track_id": 1, "attributes": {"role": "player", "team": "left", "jersey": "9"}}
    d = {"track_id": 7, "attributes": {"role": "player", "team": "left", "jersey": "9"}}
    link.update([c, d])                                    # both visible: ids stay unique
    assert len({c["track_id"], d["track_id"]}) == 2


def test_rows_without_calibration_are_not_emitted():
    r = row(1, 0.0, feat=RED)
    r["pitch"] = None
    out, rest = run(DecisionEngine(delay=0, stitch=False), [{"frame": 0, "rows": [r]}])
    assert (out + rest)[0]["preds"] == []


def test_sides_follow_a_half_time_switch():
    # red defends the left goal for 300 frames, then the teams change ends
    def make(f):
        s = -1.0 if f < 300 else 1.0
        rs = [row(i, s * (20.0 + i), feat=RED) for i in range(1, 6)]
        rs += [row(10 + i, -s * (20.0 + i), feat=WHITE) for i in range(1, 6)]
        return rs
    eng = DecisionEngine(delay=0, stitch=False, side_window=100)
    out, rest = run(eng, frames(700, make))
    red_side = [next(p["attributes"]["team"] for p in o["preds"] if p["raw_track"] == 1) for o in out + rest]
    assert red_side[250] == "left" and red_side[690] == "right"
    switch = red_side.index("right", 300)
    assert 300 <= switch <= 400
