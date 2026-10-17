"""Extra adversarial integration tests for the match library app.

Same target as tests_integration (``create_library_app`` plus the
``/api/matches`` contract from docs/WEBAPP_CONTRACT.md), but covering the
boundaries, malformed inputs, large inputs, repeated calls, ordering and
concurrency that the existing suite does not check.
"""
import json
import math
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sideline import library  # noqa: E402
from sideline.matches import MatchStore, create_router  # noqa: E402

SAMPLE = REPO / "tests" / "fixtures" / "sample_match.jsonl"
WEB = REPO / "src" / "sideline" / "web"


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def sample_bytes():
    return SAMPLE.read_bytes()


@pytest.fixture(scope="module")
def sample_frames():
    return [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.fixture()
def app_root(tmp_path):
    return tmp_path / "matches"


@pytest.fixture()
def client(app_root):
    return TestClient(library.create_library_app(app_root))


@pytest.fixture()
def sample(client, sample_bytes):
    return post(client, sample_bytes, "sample.jsonl", "sample").json()


def post(client, data, filename="t.jsonl", name=None):
    return client.post(
        "/api/matches",
        files={"file": (filename, data, "application/octet-stream")},
        data={"name": name} if name is not None else {},
    )


def _valid_xy(value):
    return (
        isinstance(value, list) and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in value)
    )


def _in_pitch(xy):
    return -52.5 <= xy[0] <= 52.5 and -34.0 <= xy[1] <= 34.0


# ------------------------------------------------- frames endpoint: slices


def test_frames_slice_at_the_boundaries_of_the_match(client, sample):
    """Catches a frames endpoint that drops or invents frames at first/last."""
    mid = sample["id"]
    whole = client.get("/api/matches/%s/frames?start=0&end=749" % mid).json()
    assert [f["frame"] for f in whole["frames"]] == list(range(750))
    assert whole["total"] == 750
    last = client.get("/api/matches/%s/frames?start=749&end=749" % mid).json()
    assert [f["frame"] for f in last["frames"]] == [749] and last["total"] == 1
    first = client.get("/api/matches/%s/frames?start=0&end=0" % mid).json()
    assert [f["frame"] for f in first["frames"]] == [0] and first["total"] == 1


def test_frames_slice_beyond_the_last_frame_only_sees_stored_frames(client, sample):
    """Catches a frames endpoint that fabricates frames past the last recorded one."""
    tail = client.get("/api/matches/%s/frames?start=748&end=100000" % sample["id"]).json()
    assert [f["frame"] for f in tail["frames"]] == [748, 749] and tail["total"] == 2
    rest = client.get("/api/matches/%s/frames?start=1&end=749" % sample["id"]).json()
    assert rest["total"] == 749 and len(rest["frames"]) == 749


def test_frames_step_larger_than_the_span_returns_the_first_frame(client, sample):
    """Catches a step that skips past the end instead of thinning to frame 0."""
    body = client.get("/api/matches/%s/frames?start=0&end=749&step=1000" % sample["id"]).json()
    assert [f["frame"] for f in body["frames"]] == [0]
    assert body["total"] == 750


def test_frames_without_params_return_the_whole_match(client, sample):
    """Catches default start/end that do not cover the whole stored match."""
    body = client.get("/api/matches/%s/frames" % sample["id"]).json()
    assert body["total"] == 750 and len(body["frames"]) == 750


@pytest.mark.parametrize(
    "query",
    ["start=10&end=5", "step=0", "start=-1", "start=abc", "end=abc", "step=1.5"],
)
def test_frames_params_outside_the_documented_domain_are_rejected(client, sample, query):
    """Catches frames parameters the documented API rejects being served anyway."""
    response = client.get("/api/matches/%s/frames?%s" % (sample["id"], query))
    assert response.status_code == 400
    assert "error" in response.json()


# ------------------------------------------------------- upload: content


def test_small_upload_meta_reflects_the_recorded_content(client):
    """Catches meta fields (duration, delay, colors, size) computed from the wrong thing."""
    payload = (
        json.dumps({"frame": 10, "delay": 7, "people": [],
                    "colors": {"left": "#112233", "right": "#445566"}}) + "\n"
        + json.dumps({"frame": 11, "people": []}) + "\n"
        + json.dumps({"frame": 12, "people": []}) + "\n"
    ).encode("utf-8")
    response = post(client, payload, "clip.jsonl", "clip")
    assert response.status_code == 201
    meta = response.json()
    assert (meta["frame_count"], meta["first_frame"], meta["last_frame"]) == (3, 10, 12)
    assert meta["duration_s"] == 0.12
    assert meta["delay"] == 7
    assert meta["colors"] == {"left": "#112233", "right": "#445566"}
    assert meta["size_bytes"] == len(payload)


def test_upload_name_falls_back_to_the_filename_stem(client):
    """Catches an upload that ignores the optional name or the file stem."""
    a = post(client, b'{"frame": 0, "people": []}\n', "clip.jsonl")
    assert a.status_code == 201 and a.json()["name"] == "clip"
    b = post(client, b'{"frame": 1, "people": []}\n', "other.jsonl", name="   ")
    assert b.status_code == 201 and b.json()["name"] == "other"


def test_reupload_with_a_different_name_keeps_the_stored_meta(client, sample_bytes):
    """Catches a re-upload that overwrites the stored match with new metadata."""
    meta = post(client, sample_bytes, "sample.jsonl", "alpha").json()
    again = post(client, sample_bytes, "sample.jsonl", "beta")
    assert again.status_code == 200
    assert again.json()["id"] == meta["id"]
    assert again.json()["name"] == "alpha"
    assert [item["name"] for item in client.get("/api/matches").json()["items"]] == ["alpha"]


def test_reupload_after_delete_stores_a_fresh_match(client, app_root):
    """Catches a delete that leaves the store believing the match still exists."""
    payload = b'{"frame": 0, "people": []}\n'
    meta = post(client, payload, "gone.jsonl", "gone").json()
    assert client.delete("/api/matches/%s" % meta["id"]).status_code == 204
    again = post(client, payload, "gone.jsonl", "gone")
    assert again.status_code == 201
    assert [item["id"] for item in client.get("/api/matches").json()["items"]] == [meta["id"]]
    assert len(list(app_root.iterdir())) == 1


def test_deleting_twice_gives_404_the_second_time(client, app_root):
    """Catches a delete that reports success for a match that is already gone."""
    meta = post(client, b'{"frame": 0, "people": []}\n', "x.jsonl", "x").json()
    assert client.delete("/api/matches/%s" % meta["id"]).status_code == 204
    assert client.delete("/api/matches/%s" % meta["id"]).status_code == 404
    assert list(app_root.iterdir()) == []
    assert client.get("/api/matches").json()["items"] == []


def test_distinct_uploads_coexist(client, sample_bytes):
    """Catches a store where a second upload clobbers or confuses the first."""
    a = post(client, sample_bytes, "a.jsonl", "a").json()
    b = post(client, b'{"frame": 0, "people": []}\n', "b.jsonl", "b").json()
    assert a["id"] != b["id"]
    items = client.get("/api/matches").json()["items"]
    assert sorted(item["id"] for item in items) == sorted([a["id"], b["id"]])
    assert client.get("/api/matches/%s" % a["id"]).json() == a
    assert client.get("/api/matches/%s" % b["id"]).json() == b


def test_missing_frames_are_kept_as_gaps_not_fabricated(client):
    """Catches a store or frames endpoint that fills in frame numbers that were never recorded."""
    payload = b'{"frame": 0, "people": []}\n{"frame": 1, "people": []}\n{"frame": 3, "people": []}\n'
    meta = post(client, payload, "gap.jsonl", "gap").json()
    assert (meta["frame_count"], meta["first_frame"], meta["last_frame"]) == (3, 0, 3)
    body = client.get("/api/matches/%s/frames?start=0&end=10" % meta["id"]).json()
    assert [f["frame"] for f in body["frames"]] == [0, 1, 3]
    assert body["total"] == 3


def test_large_upload_roundtrip(client):
    """Catches a store or reader that breaks on a match far bigger than the sample."""
    lines = []
    for i in range(2000):
        lines.append(json.dumps({
            "frame": i, "delay": 0,
            "people": [
                {"id": 1, "role": "player", "team": "left", "jersey": "1",
                 "xy": [i % 100 - 50, 0.0]},
                {"id": 2, "role": "player", "team": "right", "jersey": "2",
                 "xy": [50 - i % 100, 1.0]},
            ],
            "ball": {"xy": [i % 105 - 52, 0.5]},
        }))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    meta = post(client, payload, "big.jsonl", "big").json()
    assert (meta["frame_count"], meta["first_frame"], meta["last_frame"]) == (2000, 0, 1999)
    assert meta["duration_s"] == 80.0
    # The match page fetches frames in 250-frame chunks at these offsets.
    first_chunk = client.get("/api/matches/%s/frames?start=0&end=249" % meta["id"]).json()
    assert len(first_chunk["frames"]) == 250 and first_chunk["total"] == 250
    last_chunk = client.get("/api/matches/%s/frames?start=1750&end=1999" % meta["id"]).json()
    assert [f["frame"] for f in last_chunk["frames"]][-2:] == [1998, 1999]
    summary = client.get("/api/matches/%s/analytics" % meta["id"]).json()
    assert summary["frame_count"] == 2000


# ------------------------------------------------ upload: size boundary


def test_upload_at_and_over_the_size_limit(tmp_path):
    """Catches a size limit that rejects exactly-at-limit files or accepts over-limit ones."""
    limit = 100
    root_at = tmp_path / "at"
    app_at = FastAPI()
    app_at.include_router(create_router(MatchStore(root_at, max_bytes=limit)))
    client_at = TestClient(app_at)
    prefix = b'{"frame": 0, "people": [], "pad": "'
    suffix = b'"}\n'
    exact = prefix + b"x" * (limit - len(prefix) - len(suffix)) + suffix
    assert len(exact) == limit
    assert client_at.post(
        "/api/matches", files={"file": ("a.jsonl", exact, "application/octet-stream")},
    ).status_code == 201

    root_over = tmp_path / "over"
    app_over = FastAPI()
    app_over.include_router(create_router(MatchStore(root_over, max_bytes=limit)))
    client_over = TestClient(app_over)
    response = client_over.post(
        "/api/matches", files={"file": ("a.jsonl", exact + b"y", "application/octet-stream")},
    )
    assert response.status_code == 413
    assert list(root_over.iterdir()) == []


# ----------------------------------------------- upload: malformed bytes


BAD_UPLOADS = [
    ("empty file", b""),
    ("whitespace lines only", b"   \n\n"),
    ("array line", b"[1, 2, 3]\n"),
    ("string line", b'"hello"\n'),
    ("people not a list", b'{"frame": 1, "people": {}}\n'),
    ("frame a float", b'{"frame": 1.5, "people": []}\n'),
    ("frame a bool", b'{"frame": true, "people": []}\n'),
    ("frame missing", b'{"people": []}\n'),
    ("decreasing frames", b'{"frame": 5, "people": []}\n{"frame": 4, "people": []}\n'),
]


@pytest.mark.parametrize(("label", "payload"), BAD_UPLOADS,
                         ids=[label for label, _ in BAD_UPLOADS])
def test_malformed_upload_is_rejected_and_leaves_no_folder(tmp_path, label, payload):
    """Catches a rejected upload that still leaves a match folder or file behind."""
    root = tmp_path / "matches"
    client = TestClient(library.create_library_app(root))
    response = post(client, payload, "bad.jsonl", "bad")
    assert response.status_code == 400, label
    assert list(root.iterdir()) == []


# ------------------------------------------------------ upload: concurrent


def test_concurrent_uploads_of_the_same_bytes_store_one_match(tmp_path):
    """Catches a race where concurrent identical uploads double-store a match."""
    root = tmp_path / "matches"
    app = library.create_library_app(root)
    payload = b"".join(b'{"frame": %d, "people": []}\n' % i for i in range(5))
    results, lock = [], threading.Lock()

    def worker():
        with TestClient(app) as worker_client:
            response = post(worker_client, payload, "same.jsonl", "same")
            with lock:
                results.append((response.status_code, response.json()["id"]))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ids = {match_id for _, match_id in results}
    assert len(ids) == 1
    assert sorted(status for status, _ in results) == [200, 200, 200, 200, 201]
    assert sorted(entry.name for entry in root.iterdir()) == list(ids)


def test_concurrent_analytics_calls_all_agree(tmp_path, sample_bytes):
    """Catches a first-time analytics computation that is torn or non-deterministic."""
    app = library.create_library_app(tmp_path / "matches")
    with TestClient(app) as client:
        meta = post(client, sample_bytes, "sample.jsonl", "sample").json()
    url = "/api/matches/%s/analytics" % meta["id"]
    results, lock = [], threading.Lock()

    def worker():
        with TestClient(app) as worker_client:
            response = worker_client.get(url)
            with lock:
                results.append((response.status_code, response.json()))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(status == 200 for status, _ in results)
    assert all(body == results[0][1] for _, body in results)


# ------------------------------------------------------------- store list


def test_list_orders_newest_first_and_ties_by_id_descending(tmp_path):
    """Catches a list order that is not newest-first with ties by id descending."""
    root = tmp_path / "matches"
    store = MatchStore(root)

    def put(match_id, created_at):
        folder = root / match_id
        folder.mkdir(exist_ok=True)
        (folder / "meta.json").write_text(
            json.dumps({"id": match_id, "created_at": created_at}), encoding="utf-8")

    put("m_0000000001", "2026-01-01T00:00:00Z")
    put("m_0000000002", "2026-01-01T00:00:00Z")
    put("m_0000000003", "2026-01-01T00:00:00Z")
    # A folder that is not a match id, and a corrupt meta, are not stored matches.
    strays = root / "notes"
    strays.mkdir()
    (strays / "meta.json").write_text("{}", encoding="utf-8")
    (root / "m_0000000003" / "meta.json").write_text("corrupt", encoding="utf-8")
    assert [item["id"] for item in store.list()] == ["m_0000000002", "m_0000000001"]

    put("m_0000000001", "2026-01-02T00:00:00Z")  # now the newest
    assert [item["id"] for item in store.list()] == ["m_0000000001", "m_0000000002"]


@pytest.mark.parametrize(
    "bad_id",
    ["m_12345678901", "M_1234567890", "m_123456789g", "m_123456789", "m_123456789012",
     "m_12345678_0"],
)
def test_malformed_ids_are_404_and_leave_the_store_empty(client, app_root, bad_id):
    """Catches an id endpoint that treats a malformed id as a real match or writes for it."""
    for method in ("get", "delete"):
        assert getattr(client, method)("/api/matches/%s" % bad_id).status_code == 404, (
            method, bad_id)
    assert list(app_root.iterdir()) == []


# ------------------------------------------------------------- analytics


PLAYER_KEYS = {"id", "role", "team", "jersey", "frames", "seconds", "distance_m",
               "avg_speed_ms", "max_speed_ms", "sprints"}


def test_analytics_summary_has_the_contract_shape(client, sample):
    """Catches an analytics summary that drifts from the contract's fixed shape."""
    summary = client.get("/api/matches/%s/analytics" % sample["id"]).json()
    assert summary["fps"] == 25
    assert (summary["first_frame"], summary["last_frame"]) == (0, 749)
    assert summary["heatmaps"]["grid"] == {
        "nx": 21, "ny": 14, "x_range": [-52.5, 52.5], "y_range": [-34.0, 34.0]}
    assert isinstance(summary["ball"]["frames_with_ball"], int)
    assert 0 <= summary["ball"]["frames_with_ball"] <= 750
    assert isinstance(summary["ball"]["distance_m"], (int, float))
    for player in summary["players"]:
        assert set(player) == PLAYER_KEYS
        assert player["role"] in ("player", "goalkeeper", "referee")
        assert player["team"] in ("left", "right", None)
        assert isinstance(player["frames"], int) and player["frames"] > 0
        assert abs(player["seconds"] - round(player["frames"] / 25, 2)) < 1e-9
        assert player["distance_m"] >= 0
        assert player["avg_speed_ms"] >= 0
        assert player["max_speed_ms"] >= 0
        assert isinstance(player["sprints"], int) and player["sprints"] >= 0
    for side in ("left", "right"):
        team = summary["teams"][side]
        assert set(team) == {"frames", "avg_centroid", "avg_width", "avg_length"}
        assert isinstance(team["frames"], int) and team["frames"] >= 0
        if team["frames"]:
            assert len(team["avg_centroid"]) == 2
            assert team["avg_width"] >= 0 and team["avg_length"] >= 0
        else:
            assert team["avg_centroid"] is None


def test_analytics_heatmaps_count_the_recorded_positions(client, sample, sample_frames):
    """Catches heatmaps (or the ball summary) that do not count the recorded positions."""
    summary = client.get("/api/matches/%s/analytics" % sample["id"]).json()
    expected = {"left": 0, "right": 0, "ball": 0}
    frames_with_ball = 0
    for frame in sample_frames:
        for person in frame.get("people") or []:
            xy = person.get("xy")
            if (person.get("role") in ("player", "goalkeeper")
                    and person.get("team") in ("left", "right")
                    and _valid_xy(xy) and _in_pitch(xy)):
                expected[person["team"]] += 1
        ball = frame.get("ball")
        if isinstance(ball, dict) and _valid_xy(ball.get("xy")):
            frames_with_ball += 1
            if _in_pitch(ball["xy"]):
                expected["ball"] += 1
    for name in ("left", "right", "ball"):
        assert sum(map(sum, summary["heatmaps"][name])) == expected[name], name
    assert summary["ball"]["frames_with_ball"] == frames_with_ball


def test_heatmap_places_known_positions_in_the_contract_cells(client):
    """Catches a heatmap that puts a recorded position in the wrong cell or skips it."""
    payload = ("\n".join(json.dumps(frame) for frame in [
        {"frame": 0,
         "people": [{"id": 1, "role": "player", "team": "left", "xy": [0.0, 0.0]}],
         "ball": {"xy": [-52.5, -34.0]}},
        {"frame": 1,
         "people": [{"id": 1, "role": "player", "team": "left", "xy": [0.0, 0.0]}],
         "ball": {"xy": None}},
        {"frame": 2,
         "people": [{"id": 2, "role": "player", "team": "left", "xy": [52.5, 34.0]}],
         "ball": None},
        {"frame": 3,
         "people": [
             {"id": 3, "role": "referee", "team": None, "xy": [0.0, 0.0]},
             {"id": 4, "role": "player", "team": "left", "xy": [60.0, 0.0]},
         ],
         "ball": {"xy": [52.6, 0.0]}},
    ]) + "\n").encode("utf-8")
    meta = post(client, payload, "hm.jsonl", "hm").json()
    heatmaps = client.get("/api/matches/%s/analytics" % meta["id"]).json()["heatmaps"]
    left = heatmaps["left"]
    # Centre spot (row 7, column 10) twice, far corner (row 13, column 20) once;
    # the off-pitch player and the referee are not counted.
    assert sum(map(sum, left)) == 3
    assert left[7][10] == 2
    assert left[13][20] == 1
    assert heatmaps["ball"][0][0] == 1
    assert sum(map(sum, heatmaps["ball"])) == 1  # 52.6 is outside the pitch x range


def test_analytics_cache_survives_an_app_restart(tmp_path):
    """Catches analytics that differ between a live app and a restart on the same folder."""
    root = tmp_path / "matches"
    payload = (b'{"frame": 0, "people": [{"id": 1, "role": "player", "team": "left",'
               b' "xy": [1.0, 2.0]}]}\n{"frame": 1, "people": []}\n')
    with TestClient(library.create_library_app(root)) as first:
        meta = post(first, payload, "restart.jsonl", "restart").json()
        before = first.get("/api/matches/%s/analytics" % meta["id"]).json()
    with TestClient(library.create_library_app(root)) as second:
        assert second.get("/api/matches/%s/analytics" % meta["id"]).json() == before
        assert second.get("/api/matches/%s" % meta["id"]).json() == meta


# -------------------------------------------------------------- exports


def test_positions_csv_rows_are_exactly_the_recorded_positions(client):
    """Catches a positions CSV that includes unplaced people or drops placed ones."""
    payload = (
        '{"frame": 0, "people": ['
        '{"id": 1, "role": "player", "team": "left", "jersey": "7", "xy": [-10.5, 0]},'
        '{"id": 2, "role": "player", "team": "right", "jersey": "8", "xy": [0, null]},'
        '{"id": 3, "role": "player", "team": "right", "jersey": "9"},'
        '{"id": 4, "role": "player", "team": "left", "jersey": "4", "xy": [1.0]}'
        ']}\n'
        '{"frame": 1, "people": [{"id": 1, "role": "player", "team": "left", "jersey": "7",'
        ' "xy": [-11, 1]}]}\n'
    ).encode("utf-8")
    meta = post(client, payload, "pos.jsonl", "pos").json()
    text = client.get("/api/matches/%s/export.csv?kind=positions" % meta["id"]).text
    assert text == (
        "frame,id,team,jersey,role,x,y\n"
        "0,1,left,7,player,-10.5,0\n"
        "1,1,left,7,player,-11,1\n"
    )


def test_export_csv_kind_defaults_and_validation(client):
    """Catches an export whose default kind or invalid kind handling is wrong."""
    payload = b'{"frame": 0, "people": [{"id": 1, "role": "player", "team": "left",' \
              b'"jersey": "7", "xy": [1.0, 2.0]}]}\n'
    meta = post(client, payload, "exp.jsonl", "exp").json()
    base = "/api/matches/%s/export.csv" % meta["id"]
    default = client.get(base)
    positions = client.get(base + "?kind=positions")
    assert default.status_code == 200 and default.text == positions.text
    players = client.get(base + "?kind=players")
    assert players.status_code == 200
    assert players.text.splitlines()[0] == (
        "id,team,jersey,role,frames,seconds,distance_m,avg_speed_ms,max_speed_ms,sprints")
    assert client.get(base + "?kind=nope").status_code == 400


def test_players_csv_rows_equal_the_analytics_players(client, sample):
    """Catches a players CSV that drifts from the analytics summary it is built from."""
    summary = client.get("/api/matches/%s/analytics" % sample["id"]).json()
    lines = client.get("/api/matches/%s/export.csv?kind=players" % sample["id"]).text.splitlines()
    header = lines[0].split(",")
    assert header == ["id", "team", "jersey", "role", "frames", "seconds", "distance_m",
                      "avg_speed_ms", "max_speed_ms", "sprints"]
    assert len(lines) - 1 == len(summary["players"])
    for line, player in zip(lines[1:], summary["players"]):
        expected = ",".join("" if player[key] is None else str(player[key])
                            for key in header)
        assert line == expected


# ----------------------------------------------- page to API contract


def test_open_meta_has_every_field_the_match_page_reads(client, sample):
    """Catches the match page reading a meta field the open endpoint does not return."""
    match_text = (WEB / "match.html").read_text(encoding="utf-8")
    fields = set(re.findall(r"\bmeta\.(\w+)", match_text))
    assert {"name", "duration_s", "size_bytes", "frame_count",
            "first_frame", "last_frame", "colors"} <= fields
    meta = client.get("/api/matches/%s" % sample["id"]).json()
    for field in fields:
        assert field in meta, field


def test_list_items_carry_the_fields_the_library_rows_render(client, sample):
    """Catches a list item missing a field the library rows render (name, duration, ...)."""
    for item in client.get("/api/matches").json()["items"]:
        for key in ("id", "name", "created_at", "frame_count", "duration_s", "size_bytes"):
            assert key in item, key


# ------------------------------------------------------------ static


def test_pages_are_served_and_root_redirects_to_the_library(app_root):
    """Catches a page route, the root redirect or /healthz that the app should serve."""
    client = TestClient(library.create_library_app(app_root))
    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == "/library.html"
    for page in ("library.html", "match.html"):
        response = client.get("/" + page)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.text == (WEB / page).read_text(encoding="utf-8")
    assert client.get("/healthz").json() == {"status": "ok"}


def test_missing_and_traversal_static_paths_are_not_served(client):
    """Catches a static mount that serves missing files or files outside the web folder."""
    assert client.get("/static/no-such-asset.js").status_code == 404
    response = client.get("/static/%2e%2e%2fmatches.py")
    assert response.status_code == 404
    source = (REPO / "src" / "sideline" / "matches.py").read_text(encoding="utf-8")
    assert source not in response.text


# ------------------------------------------------------ import hygiene


def test_importing_sideline_matches_pulls_in_no_heavy_modules():
    """Catches a torch, cv2 or numpy import leaking into sideline.matches."""
    code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "import sideline.matches\n"
        "bad = [m for m in ('torch', 'cv2', 'numpy') if m in sys.modules]\n"
        "print(','.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    ) % str(REPO / "src")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
