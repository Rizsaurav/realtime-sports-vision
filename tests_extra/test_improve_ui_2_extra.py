"""Extra adversarial tests for the UI pages task (tasks/ui_pages.md).

The page half of the task is plain HTML (src/sideline/web/library.html and
match.html) plus an inline module script; the spec's "Pages" section fixes
concrete, mechanically checkable properties of those pages.  The other half
is the contract the pages consume from the match library app
(src/sideline/library.py over src/sideline/matches.py): the meta objects the
library list renders ({id, name, created_at, frame_count, duration_s,
size_bytes} per tasks/ui_pages.md, with the exact keys fixed by
tasks/matches_api.md) and the ``colors`` field the match page reads.

These tests cover the page requirements and that contract surface that the
existing suites (tests/test_library_app.py, web_tests/*.test.mjs,
tests_extra/test_improve_ui_extra.py) do not check.
"""

import json
import os
import re
import sys
from html.parser import HTMLParser

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("SIDELINE_SRC", ROOT / "src")).resolve()
sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient  # noqa: E402

from sideline import library  # noqa: E402

WEB = SRC / "sideline" / "web"


# ---------------------------------------------------------------------------
# small HTML parsing helpers (standard library only)
# ---------------------------------------------------------------------------

class PageParser(HTMLParser):
    """Collects tags, anchors (with text and header scope) and scripts."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []       # (tag, attrs)
        self.anchors = []    # {"href", "text", "in_header"}
        self.scripts = []    # {"attrs", "text"}
        self._stack = []
        self._anchor = None
        self._script = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        self._stack.append(tag)
        if tag == "a":
            self._anchor = {"href": attrs.get("href"), "text": [], "in_header": "header" in self._stack}
        elif tag == "script":
            self._script = {"attrs": attrs, "text": []}

    def handle_endtag(self, tag):
        if tag in self._stack:
            while self._stack and self._stack[-1] != tag:
                self._stack.pop()
            if self._stack:
                self._stack.pop()
        if tag == "a" and self._anchor is not None:
            self._anchor["text"] = " ".join("".join(self._anchor["text"]).split())
            self.anchors.append(self._anchor)
            self._anchor = None
        if tag == "script" and self._script is not None:
            self._script["text"] = "".join(self._script["text"])
            self.scripts.append(self._script)
            self._script = None

    def handle_data(self, data):
        if self._anchor is not None:
            self._anchor["text"].append(data)
        if self._script is not None:
            self._script["text"].append(data)


def parse(name):
    page = (WEB / name).read_text(encoding="utf-8")
    p = PageParser()
    p.feed(page)
    p.close()
    return p


def module_script(p):
    scripts = [s for s in p.scripts if s["attrs"].get("type") == "module"]
    assert scripts, "expected a <script type='module'>"
    return "".join(s["text"] for s in scripts)


def anchor(p, href, text=None, in_header=None):
    for a in p.anchors:
        if a["href"] != href:
            continue
        if text is not None and a["text"] != text:
            continue
        if in_header is not None and a["in_header"] != in_header:
            continue
        return a
    return None


# ---------------------------------------------------------------------------
# library.html
# ---------------------------------------------------------------------------

def test_library_page_header_title_and_live_viewer_link():
    p = parse("library.html")
    text = (WEB / "library.html").read_text(encoding="utf-8")
    assert "SIDELINE" in text
    m = re.search(r"<h1[^>]*>(.*?)</h1>", text, re.S)
    assert m and "Match library" in m.group(1), "header title 'Match library'"
    a = anchor(p, "/", text="Live viewer")
    assert a is not None, "expected a link to / with the text 'Live viewer'"


def test_library_page_script_implements_the_page_behaviour():
    s = module_script(parse("library.html"))
    assert 'from "/static/lib/render.js"' in s, "absolute /static import of render.js"
    assert 'fetch("/api/matches"' in s, "on load GET /api/matches"
    assert "renderLibrary(" in s
    assert "new FormData()" in s and 'body.append("file", file)' in s \
        and 'body.append("name", name)' in s, "multipart with file and (optional) name"
    assert '"/api/matches"' in s and "method: \"POST\"" in s, "POST /api/matches"
    assert "confirm(" in s, "onDelete asks confirm(...)"
    assert "method: \"DELETE\"" in s and '"/api/matches/"' in s, "DELETE /api/matches/<id>"
    assert "disabled = true" in s, "button disabled while uploading"
    assert ".reset()" in s, "form cleared on success"
    assert "Loading" in s, "a loading text is shown"


# ---------------------------------------------------------------------------
# match.html
# ---------------------------------------------------------------------------

def test_match_page_module_imports_and_reads_the_id_from_the_query_string():
    s = module_script(parse("match.html"))
    m = re.search(r"import\s*\{([^}]*)\}\s*from\s*\"/static/lib/render\.js\"", s)
    assert m, "module import from /static/lib/render.js"
    for name in ("renderPlayerTable", "renderTeamCards", "renderHeatmap"):
        assert name in m.group(1), "%s imported for the match page" % name
    assert "location.search" in s and '.get("id")' in s, "id read from location.search"


def test_match_page_loads_metadata_and_analytics_in_parallel():
    s = module_script(parse("match.html"))
    assert "Promise.all" in s, "metadata and analytics fetched in parallel"
    assert "/analytics" in s, "GET /api/matches/<id>/analytics"
    assert '"/api/matches/"' in s, "GET /api/matches/<id>"


def test_match_page_export_links_and_default_sort():
    p = parse("match.html")
    for text in ("Export positions CSV", "Export players CSV"):
        m = re.search(r"<a[^>]*download[^>]*>\s*%s\s*</a>" % re.escape(text),
                      (WEB / "match.html").read_text(encoding="utf-8"))
        assert m, "plain <a download> '%s'" % text
    s = module_script(p)
    assert "/export.csv?kind=positions" in s
    assert "/export.csv?kind=players" in s
    # default sort: distance_m descending
    assert 'key: "distance_m"' in s and "descending: true" in s


def test_match_page_renders_team_cards_table_and_three_captioned_heatmaps():
    p = parse("match.html")
    s = module_script(p)
    assert "renderTeamCards(" in s and "renderPlayerTable(" in s
    assert s.count("renderHeatmap(") >= 3, "three heatmaps (left, right, ball)"
    figcaptions = [a for t, a in p.tags if t == "figcaption"]
    assert len(figcaptions) >= 3, "each heatmap has a caption"
    # colours come from GET /api/matches/<id> when it has a colors field
    assert "colors" in s


# ---------------------------------------------------------------------------
# the API contract the pages consume
# ---------------------------------------------------------------------------

def frame_line(n, people=None, colors="__omit__"):
    d = {"frame": n, "delay": 0, "people": people or []}
    if colors != "__omit__":
        d["colors"] = colors
    return json.dumps(d)


def person(pid, team="left", jersey="1"):
    return {"id": pid, "role": "player", "team": team, "jersey": jersey,
            "box": [0, 0, 10, 20], "xy": [0.0, 0.0]}


def upload(client, lines, name=None):
    data = ("\n".join(lines) + "\n").encode("utf-8")
    return client.post("/api/matches",
                       files={"file": ("rec.jsonl", data, "application/json")},
                       data={"name": name} if name is not None else {}), data


@pytest.fixture()
def client(tmp_path):
    return TestClient(library.create_library_app(tmp_path / "matches"))


def test_meta_objects_carry_the_fields_the_library_page_renders(client):
    # tasks/ui_pages.md: renderLibrary consumes {id, name, created_at,
    # frame_count, duration_s, size_bytes}; tasks/matches_api.md fixes the
    # meta object to exactly those keys (plus first/last frame, delay, colors).
    lines = [frame_line(i, [person(1)]) for i in range(30)]
    (r, data) = upload(client, lines, name="rec")
    assert r.status_code == 201
    mid = r.json()["id"]
    items = client.get("/api/matches").json()["items"]
    assert [m["id"] for m in items] == [mid]
    for key in ("id", "name", "created_at", "frame_count", "duration_s", "size_bytes"):
        assert key in items[0], "list item must carry %r" % key
    meta = client.get("/api/matches/%s" % mid).json()
    assert meta["id"] == mid and meta["name"] == "rec"
    assert meta["frame_count"] == 30
    assert meta["size_bytes"] == len(data)
    # duration_s = round((last_frame - first_frame + 1) / 25, 2) = 30/25
    assert meta["duration_s"] == 1.2
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", meta["created_at"])


def test_meta_colors_field_follows_the_ui_contract(client):
    # tasks/ui_pages.md (match.html): "colours come from GET /api/matches/<id>
    # if it has a colors field, else null"; tasks/matches_api.md: the colors
    # value of the LAST frame whose colors is not null, null if none.
    (r1, _) = upload(client, [
        frame_line(0, [person(1)], {"left": "#111111"}),
        frame_line(1, [person(1)], {"left": "#222222", "right": "#333333"}),
        frame_line(2, [person(1)], None),
    ])
    meta1 = client.get("/api/matches/%s" % r1.json()["id"]).json()
    assert meta1["colors"] == {"left": "#222222", "right": "#333333"}

    (r2, _) = upload(client, [frame_line(i, [person(1)], None) for i in range(3)])
    meta2 = client.get("/api/matches/%s" % r2.json()["id"]).json()
    assert meta2["colors"] is None

    (r3, _) = upload(client, [frame_line(i, [person(1)]) for i in range(3)])  # key missing
    meta3 = client.get("/api/matches/%s" % r3.json()["id"]).json()
    assert meta3["colors"] is None
