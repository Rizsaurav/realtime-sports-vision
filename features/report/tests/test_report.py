"""Tests for the report feature (spec: features/report/spec.md).

Every expected value below is computed by hand from the spec; the tests only
exercise the HTTP surface of the library app, and the recordings are built in
code.  The moving tracks use displacements that are exact in binary (0.25 m and
0.125 m per frame) and are steady, so the library's jitter smoothing leaves
them untouched.
"""
import html
import json
import os
import re
import sys
import time

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest  # noqa: E402
from fastapi import APIRouter, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from sideline.library import create_library_app  # noqa: E402


# -- fixture builders --------------------------------------------------------

def person(pid, team, x, y, role="player", jersey="1", xy=True):
    return {"id": pid, "role": role, "team": team, "jersey": jersey,
            "box": [0, 0, 10, 10], "xy": [x, y] if xy else None}


def frame(n, people, ball=None):
    return {"frame": n, "delay": 0, "people": people, "ball": ball,
            "colors": {"left": "#d03030", "right": "#2050c0"}}


def recording(frames):
    return ("\n".join(json.dumps(f) for f in frames) + "\n").encode("utf-8")


def canonical_frames(ball=True):
    """50 frames (0..49, 2.0 s), match name 'Canonical match'.

      left  id 1  (-10, -10)  standing   (jersey 1)
      left  id 2  (-10,   0)  standing   (jersey 2)
      left  id 3  (-10,  10)  standing   (jersey 3)
      right id 10 (10 + 0.25f, 0)  49 steps of 0.25 m at 6.25 m/s = 12.25 m
      right id 11 (20, 10)  standing   (jersey 11)
      ball  (0.125f, 0)   49 steps of 0.125 m = 6.125 m -> 6.12 in the summary

    Summary (verified against the contract): 5 players; left distance 0.0,
    right 12.25; top speed 6.25; sprints 0; ball 6.12 m over 50 frames.
    Team shape: left (50 frames, centroid (-10.0, 0.0), width 20.0, length
    0.0); right (0 frames, everything null).
    Heatmaps (row, col, count): left (4,8,50) (7,8,50) (9,8,50); right
    (7,12,10) (7,13,20) (7,14,20) (9,14,50); ball (7,10,20) (7,11,30).
    """
    frames = []
    for f in range(50):
        people = [person(1, "left", -10.0, -10.0, jersey="1"),
                  person(2, "left", -10.0, 0.0, jersey="2"),
                  person(3, "left", -10.0, 10.0, jersey="3"),
                  person(10, "right", 10.0 + 0.25 * f, 0.0, jersey="10"),
                  person(11, "right", 20.0, 10.0, jersey="11")]
        ball = {"box": [0, 0, 4, 4], "xy": [0.125 * f, 0.0]} if ball else None
        frames.append(frame(f, people, ball=ball))
    return frames


def one_frame_frames():
    """A single frame with one standing player per team."""
    return [frame(0, [person(1, "left", 0.0, 0.0, jersey="1"),
                      person(2, "right", 1.0, 1.0, jersey="2")])]


def empty_people_frames():
    """25 frames, 1.0 s, nobody on the pitch, no ball."""
    return [frame(i, []) for i in range(25)]


def gap_frames():
    """Frames 0, 10, 20 (span 21 frames -> 0.84 s).  id 1 (left) at (0,0),
    (1,0), (2,0): the two steps span 10 frames (> 5), so none is counted:
    distance 0.0, but the heatmap still holds the 3 observed positions, all
    in cell (row 7, col 10) -> count 3."""
    return [frame(0, [person(1, "left", 0.0, 0.0, jersey="1")]),
            frame(10, [person(1, "left", 1.0, 0.0, jersey="1")]),
            frame(20, [person(1, "left", 2.0, 0.0, jersey="1")])]


def jersey_null_frames():
    """6 frames.  id 1 (left, jersey null) stands at (0,0) in frames 0..4 and
    moves to (0.2, 0) at frame 5: one step of 0.2 m in 1 frame, v = 5.0.
    distance 0.2, avg 1.0, max 5.0, sprints 0; the jersey cell must be '-'.
    Heatmap: cell (7, 10) count 6."""
    out = [frame(f, [person(1, "left", 0.0, 0.0, jersey=None)]) for f in range(5)]
    out.append(frame(5, [person(1, "left", 0.2, 0.0, jersey=None)]))
    return out


def duplicate_frames():
    """6 frames.  id 70 (left) has two entries in frame 0: (0,0) first and
    (100,0) second; it stays at (0,0) through frame 4 and moves to (0.2, 0)
    at frame 5.  First entry wins: the step 4 -> 5 is 0.2 m (v = 5.0), total
    0.2 m, heatmap cell (7, 10) count 6.  If the duplicate won, the 99.8 m
    jump would be discarded (v > 12) and the distance would read 0.0."""
    out = [frame(0, [person(70, "left", 0.0, 0.0, jersey="70"),
                     person(70, "left", 100.0, 0.0, jersey="70")])]
    out.extend(frame(f, [person(70, "left", 0.0, 0.0, jersey="70")])
               for f in range(1, 5))
    out.append(frame(5, [person(70, "left", 0.2, 0.0, jersey="70")]))
    return out


def referee_frames():
    """25 frames.  id 1 (left) stands at (0,0); id 50 is a left-team referee
    moving 0.3 m per frame (v = 7.5 m/s, 7.2 m, top speed 7.5).  The referee
    must not appear anywhere: players 1, left distance 0.0, top speed 0.0."""
    frames = []
    for f in range(25):
        frames.append(frame(f, [person(1, "left", 0.0, 0.0, jersey="1"),
                                person(50, "left", 0.3 * f, 0.0,
                                       role="referee", jersey="50")]))
    return frames


def top_six_frames():
    """6 frames.  ids 1..6 (left) each step 0.25 m once (frames 0 -> 5):
    0.25 m at 1.25 m/s each; left total 1.5 m.  All six tie at 0.25 m, so the
    top table holds ids 1..5 (id 6 dropped) in id order."""
    out = [frame(0, [person(i, "left", 10.0 * i, 0.0, jersey=str(i))
                     for i in range(1, 7)])]
    out.extend(frame(f, []) for f in range(1, 5))
    out.append(frame(5, [person(i, "left", 10.0 * i + 0.25, 0.0, jersey=str(i))
                         for i in range(1, 7)]))
    return out


def small_frames():
    """6 frames, one standing left player; used for the name-escaping test."""
    return [frame(f, [person(1, "left", 0.0, 0.0, jersey="1")]) for f in range(6)]


def perf_recording():
    """25,000 frames (1000.0 s) with 22 players (11 per team) moving on a
    fixed modulo pattern, the ball resting at (0,0) in every even frame.
    60,708,202 bytes.  Summary: 22 players, left and right each 61486.92 m,
    top speed 5.59 m/s, sprints 0, ball 0.0 m over 12500 frames; ids 1..5 tie
    at 5589.72 m (avg 5.59, max 5.59, 0 sprints)."""
    lines = []
    for f in range(25000):
        people = []
        for i in range(22):
            team = "left" if i < 11 else "right"
            x = round(-52.0 + (i * 7.3 + 0.2 * f) % 105.0, 2)
            y = round(-34.0 + (i * 3.1 + 0.1 * f) % 68.0, 2)
            people.append({"id": i + 1, "role": "player", "team": team,
                           "jersey": str(i % 11 + 1), "box": [0, 0, 10, 10],
                           "xy": [x, y]})
        ball = {"box": [0, 0, 4, 4], "xy": [0.0, 0.0]} if f % 2 == 0 else None
        lines.append(json.dumps({"frame": f, "delay": 0, "people": people,
                                 "ball": ball,
                                 "colors": {"left": "#d03030", "right": "#2050c0"}}))
    return ("\n".join(lines) + "\n").encode("utf-8")


# -- app / request helpers ---------------------------------------------------

def add_feature_routes(app, routes):
    """Register fake feature routes on the app after creation.

    routes: name -> value, where value is the JSON-able return value of the
    endpoint or an Exception instance the endpoint must raise.  Endpoints take
    the single keyword argument match_id, like registry features.
    """
    sub = APIRouter()
    for name, value in routes.items():
        def make_endpoint(value):
            def endpoint(match_id: str):
                if isinstance(value, Exception):
                    raise value
                return value
            return endpoint
        sub.add_api_route("/api/matches/{match_id}/" + name, make_endpoint(value),
                          methods=["GET"], name="fake_" + name)
    app.include_router(sub)


@pytest.fixture()
def app(tmp_path):
    application = create_library_app(tmp_path)
    return TestClient(application), tmp_path


def build_app(tmp_path, feature_routes=None):
    application = create_library_app(tmp_path)
    if feature_routes:
        add_feature_routes(application, feature_routes)
    return TestClient(application), tmp_path


def upload(client, frames, name="game"):
    r = client.post("/api/matches",
                    files={"file": (name + ".jsonl", recording(frames),
                                    "application/json")},
                    data={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def report(client, mid, **params):
    return client.get("/api/matches/%s/report.html" % mid, params=params or None)


def figure(body, fig_id):
    """The <figure> block of one heatmap (or None)."""
    m = re.search(r'<figure class="heat" id="%s">.*?</figure>' % fig_id,
                  body, re.S)
    return m.group(0) if m else None


def rect_count(fig):
    return len(re.findall(r"<rect x=", fig))


def pre_of(body, name):
    """The unescaped <pre class="data"> text of one feature section, or None
    when the section is absent."""
    marker = 'id="section-%s"' % name
    if marker not in body:
        return None
    segment = body.split(marker, 1)[1].split("</section>", 1)[0]
    m = re.search(r'<pre class="data">(.*?)</pre>', segment, re.S)
    return html.unescape(m.group(1)) if m else None


def canonical_upload(app):
    client, root = app
    return client, root, upload(client, canonical_frames(), "Canonical match")


# -- status codes and content type ---------------------------------------------

def test_report_is_200_html(app):
    client, root, mid = canonical_upload(app)
    r = report(client, mid)
    assert r.status_code == 200
    ct = r.headers.get("content-type", "")
    assert "text/html" in ct and "charset=utf-8" in ct
    assert r.text.startswith("<!DOCTYPE html>")
    assert r.text.strip().endswith("</html>")


def test_unknown_match_is_404(app):
    client, root = app
    assert report(client, "m_0000000000").status_code == 404


def test_malformed_ids_are_404(app):
    client, root = app
    for mid in ("m_123", "M_0123456789", "m_01234567890", "report", "m_ggghhhiiij"):
        assert report(client, mid).status_code == 404


def test_query_parameters_are_ignored(app):
    client, root, mid = canonical_upload(app)
    r1 = report(client, mid)
    r2 = report(client, mid, team="left", start="0", end="49", what="x")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.text == r2.text


# -- header ---------------------------------------------------------------------

def test_header_title_and_meta_line(app):
    client, root, mid = canonical_upload(app)
    body = report(client, mid).text
    assert "<title>Canonical match - match report</title>" in body
    assert "<h1>Canonical match</h1>" in body
    assert "<p class=\"meta\">50 frames | 2.0 s | 25 fps</p>" in body


def test_name_is_html_escaped(app):
    client, root = app
    mid = upload(client, small_frames(), "A & B <C>")
    body = report(client, mid).text
    assert "<h1>A &amp; B &lt;C&gt;</h1>" in body
    assert "<title>A &amp; B &lt;C&gt; - match report</title>" in body
    assert "A & B <C>" not in body


# -- headline -------------------------------------------------------------------

def test_headline_stats(app):
    client, root, mid = canonical_upload(app)
    body = report(client, mid).text
    assert '<td data-stat="players">5</td>' in body
    assert '<td data-stat="left-distance">0.0</td>' in body
    assert '<td data-stat="right-distance">12.25</td>' in body
    assert '<td data-stat="top-speed">6.25</td>' in body
    assert '<td data-stat="sprints">0</td>' in body
    assert '<td data-stat="ball-distance">6.12</td>' in body
    assert '<td data-stat="ball-frames">50</td>' in body


# -- team shape -----------------------------------------------------------------

def test_shape_row_left(app):
    client, root, mid = canonical_upload(app)
    body = report(client, mid).text
    assert ('<tr><td>left</td><td data-cell="left-players">3</td>'
            '<td data-cell="left-frames">50</td><td data-cell="left-cx">-10.0</td>'
            '<td data-cell="left-cy">0.0</td><td data-cell="left-width">20.0</td>'
            '<td data-cell="left-length">0.0</td></tr>') in body


def test_shape_row_right_missing(app):
    """The right team has only 2 positioned players, so no frame qualifies
    (fewer than 3): frames 0 and every metric cell is a dash."""
    client, root, mid = canonical_upload(app)
    body = report(client, mid).text
    assert ('<tr><td>right</td><td data-cell="right-players">2</td>'
            '<td data-cell="right-frames">0</td><td data-cell="right-cx">-</td>'
            '<td data-cell="right-cy">-</td><td data-cell="right-width">-</td>'
            '<td data-cell="right-length">-</td></tr>') in body


# -- top players -----------------------------------------------------------------

def test_top_players_rows(app):
    client, root, mid = canonical_upload(app)
    body = report(client, mid).text
    expected = [
        ('<tr data-player="10"><td>1</td><td>10</td><td>right</td><td>10</td>'
         '<td>12.25</td><td>6.25</td><td>6.25</td><td>0</td></tr>'),
        ('<tr data-player="1"><td>2</td><td>1</td><td>left</td><td>1</td>'
         '<td>0.0</td><td>0.0</td><td>0.0</td><td>0</td></tr>'),
        ('<tr data-player="2"><td>3</td><td>2</td><td>left</td><td>2</td>'
         '<td>0.0</td><td>0.0</td><td>0.0</td><td>0</td></tr>'),
        ('<tr data-player="3"><td>4</td><td>3</td><td>left</td><td>3</td>'
         '<td>0.0</td><td>0.0</td><td>0.0</td><td>0</td></tr>'),
        ('<tr data-player="11"><td>5</td><td>11</td><td>right</td><td>11</td>'
         '<td>0.0</td><td>0.0</td><td>0.0</td><td>0</td></tr>'),
    ]
    for row in expected:
        assert row in body
    assert body.count("<tr data-player=") == 5
    # list order: rank 1 first, then the 0.0 m players in id order
    assert body.index('data-player="10"') < body.index('data-player="1"') \
        < body.index('data-player="2"') < body.index('data-player="3"') \
        < body.index('data-player="11"')


def test_top_players_truncated_to_five(app):
    client, root = app
    mid = upload(client, top_six_frames())
    body = report(client, mid).text
    assert body.count("<tr data-player=") == 5
    assert '<td data-stat="left-distance">1.5</td>' in body
    assert 'data-player="6"' not in body
    assert ('<tr data-player="1"><td>1</td><td>1</td><td>left</td><td>1</td>'
            '<td>0.25</td><td>1.25</td><td>1.25</td><td>0</td></tr>') in body


def test_jersey_null_shows_dash(app):
    client, root = app
    mid = upload(client, jersey_null_frames())
    body = report(client, mid).text
    assert ('<tr data-player="1"><td>1</td><td>1</td><td>left</td><td>-</td>'
            '<td>0.2</td><td>1.0</td><td>5.0</td><td>0</td></tr>') in body


def test_referee_is_excluded(app):
    client, root = app
    mid = upload(client, referee_frames())
    body = report(client, mid).text
    # the referee moved 7.2 m at 7.5 m/s: if it leaked in, players would be 2,
    # left distance 7.2 and top speed 7.5
    assert '<td data-stat="players">1</td>' in body
    assert '<td data-stat="left-distance">0.0</td>' in body
    assert '<td data-stat="top-speed">0.0</td>' in body
    assert 'data-player="50"' not in body


# -- heatmaps --------------------------------------------------------------------

def test_heatmap_left_cells(app):
    client, root, mid = canonical_upload(app)
    fig = figure(report(client, mid).text, "heat-left")
    assert fig is not None
    assert rect_count(fig) == 294
    assert 'viewBox="0 0 21 14"' in fig and 'width="210"' in fig and 'height="140"' in fig
    # rows 4, 7, 9 in column 8, count 50, scale 50 -> opacity 1.0
    assert '<rect x="8" y="4" width="1" height="1" fill="#d03030" opacity="1.0"/>' in fig
    assert '<rect x="8" y="7" width="1" height="1" fill="#d03030" opacity="1.0"/>' in fig
    assert '<rect x="8" y="9" width="1" height="1" fill="#d03030" opacity="1.0"/>' in fig
    # every other cell is empty
    assert fig.count('opacity="1.0"/>') == 3
    assert fig.count('opacity="0.0"/>') == 291


def test_heatmap_right_cells(app):
    client, root, mid = canonical_upload(app)
    fig = figure(report(client, mid).text, "heat-right")
    assert fig is not None
    assert rect_count(fig) == 294
    # (7,12,10) (7,13,20) (7,14,20) (9,14,50); scale 50 -> 0.2, 0.4, 0.4, 1.0
    assert '<rect x="12" y="7" width="1" height="1" fill="#2050c0" opacity="0.2"/>' in fig
    assert '<rect x="13" y="7" width="1" height="1" fill="#2050c0" opacity="0.4"/>' in fig
    assert '<rect x="14" y="7" width="1" height="1" fill="#2050c0" opacity="0.4"/>' in fig
    assert '<rect x="14" y="9" width="1" height="1" fill="#2050c0" opacity="1.0"/>' in fig
    assert fig.count('opacity="0.0"/>') == 290


def test_heatmap_ball_cells(app):
    client, root, mid = canonical_upload(app)
    fig = figure(report(client, mid).text, "heat-ball")
    assert fig is not None
    assert rect_count(fig) == 294
    # (7,10,20) (7,11,30); scale 30 -> 20/30 rounds to 0.67, 30/30 is 1.0
    assert '<rect x="10" y="7" width="1" height="1" fill="#404040" opacity="0.67"/>' in fig
    assert '<rect x="11" y="7" width="1" height="1" fill="#404040" opacity="1.0"/>' in fig
    assert fig.count('opacity="0.0"/>') == 292


# -- self-contained document and print CSS ----------------------------------------

def test_no_external_resources(app):
    client, root, mid = canonical_upload(app)
    body = report(client, mid).text
    for forbidden in ("<script", "<link", "<img", "<iframe", "<object",
                      "<embed", "src=", "@import", "url(", "http://",
                      "https://", "xmlns"):
        assert forbidden not in body, forbidden
    assert body.count("<!DOCTYPE html>") == 1
    assert body.count("<main class=\"report\">") == 1
    assert body.count("<svg") == 3


def test_print_css(app):
    client, root, mid = canonical_upload(app)
    body = report(client, mid).text
    assert "<style>" in body and "</style>" in body
    assert "@page { size: A4; margin: 12mm; }" in body
    assert "@media print" in body
    assert "break-inside: avoid" in body


# -- optional feature sections ------------------------------------------------------

def test_feature_sections_absent_without_routes(app):
    client, root, mid = canonical_upload(app)
    body = report(client, mid).text
    assert 'id="feature-sections"' in body
    for name in ("possession", "formation", "events", "quality"):
        assert 'id="section-%s"' % name not in body
    assert body.count('<section class="feature"') == 0
    assert '<pre class="data">' not in body


def test_feature_section_possession_content(tmp_path):
    client, root = build_app(tmp_path, {"possession": {"share": 0.6}})
    mid = upload(client, canonical_frames(), "Canonical match")
    body = report(client, mid).text
    assert 'id="section-possession"' in body
    assert 'id="section-formation"' not in body
    assert json.loads(pre_of(body, "possession")) == {"share": 0.6}


def test_feature_sections_fixed_order(tmp_path):
    """All four present, registered in a shuffled order: the document must
    still carry them in the fixed order possession, formation, events,
    quality.  The events fake returns a list (dict or list both render)."""
    client, root = build_app(tmp_path, {
        "formation": {"width": 30.0},
        "quality": {"score": 0.9},
        "possession": {"share": 0.5},
        "events": [1, 2, 3],
    })
    mid = upload(client, small_frames())
    body = report(client, mid).text
    positions = [body.index('id="section-%s"' % name)
                 for name in ("possession", "formation", "events", "quality")]
    assert positions == sorted(positions)
    assert json.loads(pre_of(body, "events")) == [1, 2, 3]
    assert json.loads(pre_of(body, "quality")) == {"score": 0.9}


def test_feature_section_omitted_when_endpoint_fails(tmp_path):
    """A failing sibling is dropped silently; the working sibling and the
    200 status survive."""
    client, root = build_app(tmp_path, {
        "possession": HTTPException(status_code=404, detail="match not found"),
        "formation": {"width": 30.0},
    })
    mid = upload(client, small_frames())
    r = report(client, mid)
    assert r.status_code == 200
    body = r.text
    assert 'id="section-possession"' not in body
    assert 'id="section-formation"' in body
    assert json.loads(pre_of(body, "formation")) == {"width": 30.0}


def test_feature_section_omitted_when_not_dict(tmp_path):
    client, root = build_app(tmp_path, {"possession": "hello",
                                        "quality": None})
    mid = upload(client, small_frames())
    r = report(client, mid)
    assert r.status_code == 200
    body = r.text
    assert 'id="section-possession"' not in body
    assert 'id="section-quality"' not in body
    assert body.count('<section class="feature"') == 0


def test_feature_route_matching_is_exact(tmp_path):
    """A route that only contains the name in a longer path must not be
    picked up: '/api/matches/{match_id}/possession/extra' is not the
    possession route."""
    application = create_library_app(tmp_path)
    sub = APIRouter()

    @sub.get("/api/matches/{match_id}/possession/extra")
    def decoy(match_id: str):
        return {"decoy": True}

    application.include_router(sub)
    client = TestClient(application)
    mid = upload(client, small_frames())
    r = report(client, mid)
    assert r.status_code == 200
    body = r.text
    assert 'id="section-possession"' not in body
    assert client.get("/api/matches/%s/possession/extra" % mid).json() == {"decoy": True}


# -- degenerate matches ---------------------------------------------------------------

def test_one_frame_match(app):
    client, root = app
    mid = upload(client, one_frame_frames())
    body = report(client, mid).text
    assert "<p class=\"meta\">1 frames | 0.04 s | 25 fps</p>" in body
    assert '<td data-stat="players">2</td>' in body
    assert '<td data-stat="top-speed">0.0</td>' in body
    assert '<td data-stat="ball-frames">0</td>' in body
    assert ('<tr data-player="1"><td>1</td><td>1</td><td>left</td><td>1</td>'
            '<td>0.0</td><td>0.0</td><td>0.0</td><td>0</td></tr>') in body
    assert ('<tr data-player="2"><td>2</td><td>2</td><td>right</td><td>2</td>'
            '<td>0.0</td><td>0.0</td><td>0.0</td><td>0</td></tr>') in body
    assert '<td data-cell="left-cx">-</td>' in body
    assert '<td data-cell="right-cx">-</td>' in body


def test_empty_people_match(app):
    client, root = app
    mid = upload(client, empty_people_frames())
    body = report(client, mid).text
    assert "<p class=\"meta\">25 frames | 1.0 s | 25 fps</p>" in body
    assert '<td data-stat="players">0</td>' in body
    assert '<td data-stat="left-distance">0.0</td>' in body
    assert '<td data-stat="right-distance">0.0</td>' in body
    assert '<td data-stat="top-speed">0.0</td>' in body
    assert '<td data-stat="ball-distance">0.0</td>' in body
    assert '<td data-stat="ball-frames">0</td>' in body
    assert body.count("<tr data-player=") == 0
    assert '<td data-cell="left-players">0</td>' in body
    assert '<td data-cell="left-frames">0</td>' in body
    assert '<td data-cell="right-length">-</td>' in body
    for fig_id in ("heat-left", "heat-right", "heat-ball"):
        fig = figure(body, fig_id)
        assert rect_count(fig) == 294
        assert fig.count('opacity="0.0"/>') == 294


def test_no_ball_match(app):
    client, root = app
    mid = upload(client, canonical_frames(ball=False))
    body = report(client, mid).text
    assert '<td data-stat="ball-distance">0.0</td>' in body
    assert '<td data-stat="ball-frames">0</td>' in body
    fig = figure(body, "heat-ball")
    assert rect_count(fig) == 294 and fig.count('opacity="0.0"/>') == 294
    # the rest of the report is unaffected
    assert '<td data-stat="right-distance">12.25</td>' in body
    assert '<td data-stat="players">5</td>' in body


def test_frame_gaps(app):
    client, root = app
    mid = upload(client, gap_frames())
    body = report(client, mid).text
    # span (20 - 0 + 1) / 25 = 0.84 s, 3 recorded frames
    assert "<p class=\"meta\">3 frames | 0.84 s | 25 fps</p>" in body
    # both steps span 10 frames (> 5) and are discarded
    assert '<td data-stat="left-distance">0.0</td>' in body
    assert '<td data-stat="top-speed">0.0</td>' in body
    # the three observed positions all fall in cell (7, 10), scale 3
    fig = figure(body, "heat-left")
    assert '<rect x="10" y="7" width="1" height="1" fill="#d03030" opacity="1.0"/>' in fig
    assert fig.count('opacity="0.0"/>') == 293


def test_duplicate_id_first_entry_wins(app):
    client, root = app
    mid = upload(client, duplicate_frames())
    body = report(client, mid).text
    # if the (100, 0) duplicate won, the jump would be discarded: 0.0 m
    assert '<td data-stat="left-distance">0.2</td>' in body
    assert ('<tr data-player="70"><td>1</td><td>70</td><td>left</td><td>70</td>'
            '<td>0.2</td><td>1.0</td><td>5.0</td><td>0</td></tr>') in body
    fig = figure(body, "heat-left")
    assert '<rect x="10" y="7" width="1" height="1" fill="#d03030" opacity="1.0"/>' in fig
    assert fig.count('opacity="0.0"/>') == 293


# -- idempotence and mutation ---------------------------------------------------------

def test_idempotent_byte_identical(tmp_path):
    client, root = build_app(tmp_path, {"possession": {"share": 0.6}})
    mid = upload(client, canonical_frames(), "Canonical match")
    r1 = report(client, mid)
    r2 = report(client, mid, cache="no")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.text == r2.text
    assert '<td data-stat="right-distance">12.25</td>' in r1.text
    assert 'id="section-possession"' in r1.text
    assert json.loads(pre_of(r1.text, "possession")) == {"share": 0.6}


def test_frames_not_mutated(app):
    client, root, mid = canonical_upload(app)
    before = client.get("/api/matches/%s/frames" % mid).json()
    assert report(client, mid).status_code == 200
    assert report(client, mid, team="left").status_code == 200
    after = client.get("/api/matches/%s/frames" % mid).json()
    assert before == after


def test_feature_listed(app):
    client, root = app
    r = client.get("/api/features")
    assert r.status_code == 200
    feature = next(f for f in r.json()["features"] if f["name"] == "report")
    assert feature["ok"] is True


# -- performance ------------------------------------------------------------------------

def test_performance_25000_frames(app):
    client, root = app
    r = client.post("/api/matches",
                    files={"file": ("big.jsonl", perf_recording(),
                                    "application/json")},
                    data={"name": "big"})
    assert r.status_code == 201, r.text
    mid = r.json()["id"]
    t0 = time.monotonic()
    r = report(client, mid)
    first_s = time.monotonic() - t0
    assert r.status_code == 200, r.text
    assert first_s < 20.0, "first report took %.1f s (budget 20 s)" % first_s
    body = r.text
    assert "<p class=\"meta\">25000 frames | 1000.0 s | 25 fps</p>" in body
    assert '<td data-stat="players">22</td>' in body
    assert '<td data-stat="left-distance">61486.92</td>' in body
    assert '<td data-stat="right-distance">61486.92</td>' in body
    assert '<td data-stat="top-speed">5.59</td>' in body
    assert '<td data-stat="sprints">0</td>' in body
    assert '<td data-stat="ball-distance">0.0</td>' in body
    assert '<td data-stat="ball-frames">12500</td>' in body
    assert ('<tr data-player="1"><td>1</td><td>1</td><td>left</td><td>1</td>'
            '<td>5589.72</td><td>5.59</td><td>5.59</td><td>0</td></tr>') in body
    assert ('<tr data-player="2"><td>2</td><td>2</td><td>left</td><td>2</td>'
            '<td>5589.72</td><td>5.59</td><td>5.59</td><td>0</td></tr>') in body
    for fig_id in ("heat-left", "heat-right", "heat-ball"):
        assert rect_count(figure(body, fig_id)) == 294
    t0 = time.monotonic()
    r2 = report(client, mid)
    second_s = time.monotonic() - t0
    assert r2.status_code == 200
    assert second_s < 5.0, "cached report took %.1f s (budget 5 s)" % second_s
    assert r2.text == body
