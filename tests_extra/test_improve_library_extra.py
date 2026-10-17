"""Extra adversarial tests for the match library app (tasks/library_server.md).

The acceptance suite (tests/test_library_app.py) only exercises the happy
path.  These tests attack the spec rules it does not check, through the real
``create_library_app`` and its endpoints:

* pages served byte-for-byte from the package ``web`` folder, independent of
  the working directory; the ``/static`` mount (other folder files, traversal);
* upload validation details: integral floats as ``frame``, line numbering
  with leading blank lines, the first failing line, a ``file`` field sent as
  a plain form field, and ``people`` entries that are not objects;
* meta edge values: ``delay`` (string / float / bool / null / negative),
  ``colors`` with a missing key, blank vs stripped ``source``;
* duplicates, hand-edited ``meta.json`` / ``analytics.json`` (read from disk
  on every request, no recompute), list entries that are not matches;
* frames parameter grammar (leading zeros, embedded spaces), huge bounds,
  and the 404-before-400 check order on frames and export.csv;
* CSV cell rules: ``str()`` of the parsed value, empty cells for missing
  keys, and every ``xy`` variant that must not produce a row;
* concurrency on the first analytics request (written once, guarded by the
  store lock) and on concurrent uploads of the same bytes;
* a 1500-frame recording (large input);
* the CLI: ``build_parser()`` imports nothing heavy in a fresh interpreter,
  and argument parsing works while uvicorn cannot be imported (it is only
  needed when the command actually runs).

Every assertion is required by tasks/library_server.md,
tasks/matches_api.md or docs/WEBAPP_CONTRACT.md.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from fastapi.testclient import TestClient  # noqa: E402

from sideline import __main__ as cli  # noqa: E402
from sideline import analytics  # noqa: E402
from sideline import library  # noqa: E402
from sideline import matches  # noqa: E402

WEB_DIR = Path(os.environ.get("SIDELINE_SRC", "src")).resolve() / "sideline" / "web"


def recording(frames):
    """Serialize frame dicts as an uploaded recording (one JSON object per line)."""
    return ("\n".join(json.dumps(f) for f in frames) + "\n").encode("utf-8")


def simple(n, start=0):
    return [{"frame": start + i, "delay": 0, "people": [], "colors": None, "ball": None}
            for i in range(n)]


@pytest.fixture()
def app(tmp_path):
    return library.create_library_app(tmp_path / "store")


@pytest.fixture()
def client(app):
    # A 500 is a legitimate response to assert against; do not re-raise it.
    return TestClient(app, raise_server_exceptions=False)


def upload(c, data, filename="rec.jsonl", **form):
    return c.post("/api/matches", files={"file": (filename, data, "application/json")},
                  data=form)


# ------------------------------------------------------------- pages / static

@pytest.mark.parametrize("page", ["library.html", "match.html"])
def test_pages_serve_the_exact_files_from_the_web_folder(client, page):
    r = client.get("/" + page)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.text == (WEB_DIR / page).read_text(encoding="utf-8")


def test_pages_and_static_are_package_relative_not_cwd_relative(app, tmp_path, monkeypatch):
    # The web folder is located relative to the sideline package, not the CWD.
    monkeypatch.chdir(tmp_path)
    c = TestClient(app, raise_server_exceptions=False)
    for path in ("/library.html", "/match.html", "/static/lib/render.js", "/static/lib/replay.js"):
        assert c.get(path).status_code == 200, path


@pytest.mark.parametrize("path", ["/static/style.css", "/static/pages.css", "/static/lib/pitch.js"])
def test_static_serves_other_files_of_the_web_folder(client, path):
    r = client.get(path)
    assert r.status_code == 200, path
    # /static is the mount point of the web folder itself.
    assert r.content == WEB_DIR.joinpath(path[len("/static"):].lstrip("/")).read_bytes()


@pytest.mark.parametrize("path", ["/static/%2E%2E/lib", "/static/%2e%2e%2fstyle.css"])
def test_static_path_traversal_is_404(client, path):
    assert client.get(path).status_code == 404


def test_following_the_root_redirect_lands_on_the_library_page(client):
    r = client.get("/", follow_redirects=True)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.text == (WEB_DIR / "library.html").read_text(encoding="utf-8")


# ------------------------------------------------------------ upload validation

def test_upload_frame_must_not_be_an_integral_float(client):
    data = recording([{"frame": 0, "people": []}, {"frame": 3.0, "people": []}])
    r = upload(client, data)
    assert r.status_code == 400
    assert re.search(r"\b2\b", r.json()["error"])


def test_upload_error_reports_the_first_failing_line(client):
    good = json.dumps({"frame": 0, "people": []})
    data = ("\n".join([good, '{"frame": "x", "people": []}', '{"frame": "y", "people": []}'])
            + "\n").encode()
    r = upload(client, data)
    assert r.status_code == 400
    assert re.search(r"\b2\b", r.json()["error"])


def test_upload_line_numbers_count_leading_blank_lines(client):
    data = ("\n\n" + '{"frame": "x", "people": []}' + "\n").encode()
    r = upload(client, data)
    assert r.status_code == 400
    assert re.search(r"\b3\b", r.json()["error"])


def test_upload_file_sent_as_a_plain_form_field_is_400(client):
    r = client.post("/api/matches", data={"file": "some text", "name": "x"})
    assert r.status_code == 400 and r.json()["error"]


def test_upload_accepts_people_entries_that_are_not_objects(client):
    # Rule 4 of the upload validation only requires `people` to be a list.
    data = recording([{"frame": 0, "people": [1, None, "x",
                                              {"id": 5, "role": "player", "xy": [1.0, 2.0]}]}])
    r = upload(client, data)
    assert r.status_code == 201, r.text


def test_positions_export_skips_people_entries_that_are_not_objects(client):
    # People without an xy (missing or null) produce no row; a non-object
    # person has no xy, so it must produce no row either - not a server error.
    data = recording([{"frame": 0, "people": [1, None, "x",
                                              {"id": 5, "role": "player",
                                               "team": None, "jersey": None, "xy": [1.0, 2.0]}]}])
    mid = upload(client, data).json()["id"]
    r = client.get("/api/matches/%s/export.csv?kind=positions" % mid)
    assert r.status_code == 200, r.text
    assert r.text == "frame,id,team,jersey,role,x,y\n0,5,,,player,1.0,2.0\n"


def test_duplicate_upload_keeps_the_stored_meta_untouched(client):
    data = recording(simple(3))
    first = upload(client, data, filename="a.jsonl", name="one").json()
    second = upload(client, data, filename="b.jsonl", name="two")
    assert second.status_code == 200
    assert second.json() == first
    assert second.json()["name"] == "one"


# ------------------------------------------------------------------------ meta

@pytest.mark.parametrize("delay", ["5", 5.0, True, None])
def test_meta_delay_is_zero_when_not_an_integer(client, delay):
    data = recording([{"frame": 0, "people": [], "delay": delay}])
    assert upload(client, data).json()["delay"] == 0


def test_meta_delay_is_zero_when_missing(client):
    data = recording([{"frame": 0, "people": []}])
    assert upload(client, data).json()["delay"] == 0


@pytest.mark.parametrize("delay", [-5, 7])
def test_meta_delay_keeps_plain_integers(client, delay):
    start = abs(delay) + 100  # distinct bytes per case
    data = recording([{"frame": start, "people": [], "delay": delay},
                       {"frame": start + 1, "people": [], "delay": 0}])
    assert upload(client, data).json()["delay"] == delay


def test_meta_colors_missing_key_counts_as_null(client):
    c1, c2 = {"left": "#111111"}, {"left": "#222222"}
    # a frame without the key does not reset the value ...
    data = recording([{"frame": 0, "people": [], "colors": c1},
                      {"frame": 1, "people": []},
                      {"frame": 2, "people": [], "colors": None}])
    assert upload(client, data).json()["colors"] == c1
    # ... and the last frame whose colors is not null wins
    data = recording([{"frame": 0, "people": [], "colors": c1},
                      {"frame": 1, "people": []},
                      {"frame": 2, "people": [], "colors": c2}])
    assert upload(client, data).json()["colors"] == c2
    # never known -> null
    data = recording([{"frame": 0, "people": []}, {"frame": 1, "people": []}])
    assert upload(client, data).json()["colors"] is None


@pytest.mark.parametrize("source, expected, start",
                         [(None, "absent", 100), ("   ", "absent", 200), ("  clip  ", "clip", 300)])
def test_meta_source_absent_when_blank_and_stripped(client, source, expected, start):
    data = recording(simple(2, start=start))  # distinct bytes per case
    form = {} if source is None else {"source": source}
    meta = upload(client, data, **form).json()
    if expected == "absent":
        assert "source" not in meta
    else:
        assert meta["source"] == expected


def test_hand_edited_meta_json_is_served_and_listed(client, tmp_path, app):
    mid = upload(client, recording(simple(2))).json()["id"]
    meta = client.get("/api/matches/%s" % mid).json()
    meta["name"] = "edited by hand"
    (tmp_path / "store" / mid / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert client.get("/api/matches/%s" % mid).json() == meta
    assert client.get("/api/matches").json()["items"] == [meta]


def test_list_ignores_non_match_entries(client, tmp_path, app):
    store = tmp_path / "store"
    real = upload(client, recording(simple(2))).json()
    (store / "notes.txt").write_text("not a match", encoding="utf-8")
    (store / "notamatch").mkdir()
    (store / "m_ZZZZ").mkdir()                       # name is not a valid id
    (store / "m_ZZZZ" / "meta.json").write_text("{}", encoding="utf-8")
    (store / "m_0123456789").mkdir()                 # valid name, no meta.json
    assert client.get("/api/matches").json()["items"] == [real]


# ------------------------------------------------------------------- analytics

def test_hand_edited_analytics_json_is_served_without_recomputing(app, tmp_path, monkeypatch):
    c1 = TestClient(app, raise_server_exceptions=False)
    mid = upload(c1, recording(simple(3))).json()["id"]
    c1.get("/api/matches/%s/analytics" % mid)        # computes and caches
    calls = []
    monkeypatch.setattr(analytics, "summarize",
                        lambda frames, fps=25: calls.append(fps) or {"fps": 25, "edited": True})
    (tmp_path / "store" / mid / "analytics.json").write_text(
        json.dumps({"fps": 25, "edited": True}), encoding="utf-8")
    # A brand new app (new MatchStore) returns the file contents, no summarize.
    c2 = TestClient(library.create_library_app(tmp_path / "store"), raise_server_exceptions=False)
    assert c2.get("/api/matches/%s/analytics" % mid).json() == {"fps": 25, "edited": True}
    assert calls == []


def test_real_analytics_on_a_peopleless_recording(client):
    mid = upload(client, recording(simple(3))).json()["id"]
    a = client.get("/api/matches/%s/analytics" % mid).json()
    assert a["frame_count"] == 3 and a["fps"] == 25 and a["players"] == []
    r = client.get("/api/matches/%s/export.csv?kind=players" % mid)
    assert r.status_code == 200
    assert r.text == "id,team,jersey,role,frames,seconds,distance_m,avg_speed_ms,max_speed_ms,sprints\n"


def test_concurrent_first_analytics_requests_compute_once(app, tmp_path, monkeypatch):
    store = matches.MatchStore(tmp_path / "store")
    c = TestClient(app, raise_server_exceptions=False)
    mid = upload(c, recording(simple(5))).json()["id"]

    calls = []

    def fake(frames, fps=25):
        calls.append(fps)
        return {"fps": fps, "tag": "fake"}

    monkeypatch.setattr(analytics, "summarize", fake)
    results, errors = [], []
    barrier = threading.Barrier(8)

    def go():
        try:
            barrier.wait(timeout=5)
            results.append(store.analytics(mid))
        except Exception as exc:  # pragma: no cover - diagnostics only
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []
    assert calls == [25]                                # written on the first request only
    assert all(r == {"fps": 25, "tag": "fake"} for r in results)
    assert (tmp_path / "store" / mid / "analytics.json").is_file()


# ---------------------------------------------------------------------- frames

def test_frames_params_accept_digit_strings_with_leading_zeros(client):
    mid = upload(client, recording([{"frame": n, "people": []} for n in (5, 6, 7, 8)])).json()["id"]
    r = client.get("/api/matches/%s/frames" % mid, params={"start": "007", "end": "008"})
    assert r.status_code == 200
    assert [f["frame"] for f in r.json()["frames"]] == [7, 8]
    assert r.json()["total"] == 2


@pytest.mark.parametrize("param, value", [("start", " 1"), ("end", "1 "), ("step", " 2")])
def test_frames_params_with_spaces_are_400(client, param, value):
    mid = upload(client, recording(simple(3))).json()["id"]
    r = client.get("/api/matches/%s/frames" % mid, params={param: value})
    assert r.status_code == 400 and r.json()["error"]


def test_frames_unknown_match_is_404_before_param_errors(client):
    r = client.get("/api/matches/m_0123456789/frames", params={"start": "abc"})
    assert r.status_code == 404 and r.json()["error"]


def test_frames_huge_bounds_select_by_frame_number(client):
    mid = upload(client, recording(simple(10))).json()["id"]
    r = client.get("/api/matches/%s/frames" % mid, params={"start": "0", "end": "1000000000000"})
    assert r.status_code == 200
    assert r.json()["total"] == 10
    assert [f["frame"] for f in r.json()["frames"]] == list(range(10))


# ------------------------------------------------------------------------- csv

def test_positions_csv_cells_and_skips(client):
    frame0 = {"frame": 0, "people": [
        {"id": 2, "team": "left", "jersey": "10", "role": "player", "xy": [5.0, -3]},
        {"id": 1, "team": None, "jersey": None, "role": "player", "xy": [0, 1]},
        {"id": 3, "xy": [1.5, 2.5]},
        {"id": 44, "xy": None},
        {"id": 45},
    ]}
    frame1 = {"frame": 1, "people": [
        {"id": 4, "xy": [1, 2, 3]},
        {"id": 5, "xy": [None, 5]},
        {"id": 6, "xy": [5]},
        {"id": 7, "xy": 5},
        {"id": 8, "xy": "1,2"},
        {"id": 9, "team": "right", "jersey": "7", "role": "referee", "xy": [-12.5, 0]},
    ]}
    mid = upload(client, recording([frame0, frame1])).json()["id"]
    r = client.get("/api/matches/%s/export.csv?kind=positions" % mid)
    assert r.status_code == 200
    # str() of the parsed JSON value: 5.0 stays "5.0", -3 stays "-3";
    # missing keys and nulls are empty cells.
    assert r.text == (
        "frame,id,team,jersey,role,x,y\n"
        "0,2,left,10,player,5.0,-3\n"
        "0,1,,,player,0,1\n"
        "0,3,,,,1.5,2.5\n"
        "1,9,right,7,referee,-12.5,0\n"
    )


def test_export_csv_unknown_match_is_404_before_kind_errors(client):
    r = client.get("/api/matches/m_0123456789/export.csv", params={"kind": "bogus"})
    assert r.status_code == 404 and r.json()["error"]


# ------------------------------------------------------------------ concurrency

def test_concurrent_uploads_of_the_same_bytes_create_exactly_one_match(app, tmp_path):
    data = recording(simple(4))
    mid = "m_" + hashlib.sha256(data).hexdigest()[:10]
    results, errors = [], []
    barrier = threading.Barrier(4)

    def go(i):
        try:
            barrier.wait(timeout=5)
            c = TestClient(app, raise_server_exceptions=False)
            results.append(upload(c, data, filename="g%d.jsonl" % i, name="g%d" % i).status_code)
        except Exception as exc:  # pragma: no cover - diagnostics only
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []
    assert sorted(results) == [200, 200, 200, 201]      # one new, the rest the stored meta
    c = TestClient(app, raise_server_exceptions=False)
    items = c.get("/api/matches").json()["items"]
    assert [m["id"] for m in items if m["id"] == mid] == [mid]
    assert len(items) == 1
    assert items[0]["frame_count"] == 4
    assert (tmp_path / "store" / mid / "frames.jsonl").read_bytes() == data


# ----------------------------------------------------------------------- large

def test_large_recording_round_trip(client):
    n = 1500
    frames = [{"frame": f, "delay": 0, "colors": None, "ball": None, "people": [
        {"id": 1, "role": "player", "team": "left", "jersey": "9", "xy": [0.1 * f, 5.0]},
        {"id": 2, "role": "player", "team": "right", "jersey": "4", "xy": [52.4 - 0.1 * f, -5.0]},
    ]} for f in range(n)]
    meta = upload(client, recording(frames)).json()
    assert meta["frame_count"] == n
    assert meta["first_frame"] == 0 and meta["last_frame"] == n - 1
    assert meta["duration_s"] == 60.0
    mid = meta["id"]
    r = client.get("/api/matches/%s/frames" % mid, params={"step": "250"})
    assert [f["frame"] for f in r.json()["frames"]] == [0, 250, 500, 750, 1000, 1250]
    assert r.json()["total"] == n
    r = client.get("/api/matches/%s/frames" % mid, params={"start": "1499", "end": "1499"})
    assert [f["frame"] for f in r.json()["frames"]] == [1499]
    r = client.get("/api/matches/%s/export.csv?kind=positions" % mid)
    assert len(r.text.splitlines()) == 1 + 2 * n
    a = client.get("/api/matches/%s/analytics" % mid).json()
    assert a["frame_count"] == n and len(a["players"]) == 2


# ------------------------------------------------------------------------ cli

def test_build_parser_imports_nothing_heavy_in_a_fresh_interpreter():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from sideline import __main__ as cli\n"
        "ns = cli.build_parser().parse_args(['library', '--root', 'r'])\n"
        "assert ns.cmd == 'library' and ns.root == 'r'\n"
        "bad = [m for m in ('uvicorn', 'yaml', 'torch', 'cv2', 'numpy',\n"
        "                   'sideline.pipeline', 'sideline.server') if m in sys.modules]\n"
        "sys.exit(1 if bad else 0)"
    ) % os.environ.get("SIDELINE_SRC", "src")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_argument_parsing_works_without_uvicorn_until_the_command_runs(monkeypatch, tmp_path):
    # Blocking the uvicorn import: parsing must still work, and main() may
    # only fail when the command actually runs (i.e. when it imports uvicorn).
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    parser = cli.build_parser()
    ns = parser.parse_args(["library", "--root", str(tmp_path / "store")])
    assert ns.cmd == "library" and ns.root == str(tmp_path / "store")
    assert ns.host == "127.0.0.1" and ns.port == 8100
    monkeypatch.setattr(sys, "argv",
                        ["sideline", "library", "--root", str(tmp_path / "store")])
    with pytest.raises(ImportError) as exc:
        cli.main()
    assert "uvicorn" in str(exc.value)


def test_cli_library_port_must_be_an_integer():
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["library", "--root", "r", "--port", "not-a-port"])
    assert exc.value.code == 2
