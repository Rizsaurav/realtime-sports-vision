"""Round trip of the recorded sample through the library app (task item 1)."""
import re

from fastapi.testclient import TestClient

from sideline import library


def upload(client, data, name="sample"):
    return client.post(
        "/api/matches",
        files={"file": ("sample.jsonl", data, "application/octet-stream")},
        data={"name": name},
    )


def test_upload_stores_the_sample_and_reupload_keeps_the_id(client, sample_bytes):
    """Catches a broken upload response or an id that changes on identical re-upload."""
    first = upload(client, sample_bytes)
    assert first.status_code == 201
    meta = first.json()
    assert re.fullmatch(r"m_[0-9a-f]{10}", meta["id"])
    assert (meta["frame_count"], meta["first_frame"], meta["last_frame"]) == (750, 0, 749)
    assert meta["duration_s"] == 30.0 and meta["delay"] == 125
    assert meta["name"] == "sample"
    second = upload(client, sample_bytes)
    assert second.status_code == 200 and second.json()["id"] == meta["id"]


def test_list_and_open_return_the_stored_match(client, sample_bytes):
    """Catches a list or open endpoint that loses or mangles the stored match."""
    meta = upload(client, sample_bytes).json()
    items = client.get("/api/matches").json()["items"]
    assert [item["id"] for item in items] == [meta["id"]]
    opened = client.get("/api/matches/%s" % meta["id"])
    assert opened.status_code == 200 and opened.json() == meta


def test_frames_slice_returns_the_recorded_frames(client, sample_bytes, sample_frames):
    """Catches a frames endpoint that serves the wrong slice or corrupts frame content."""
    meta = upload(client, sample_bytes).json()
    body = client.get("/api/matches/%s/frames?start=100&end=104" % meta["id"]).json()
    assert [frame["frame"] for frame in body["frames"]] == [100, 101, 102, 103, 104]
    assert body["total"] == 5
    recorded = next(f for f in sample_frames if f["frame"] == 100)
    assert body["frames"][0] == recorded  # uncorrupted round trip
    stepped = client.get("/api/matches/%s/frames?start=100&end=119&step=10" % meta["id"]).json()
    assert [frame["frame"] for frame in stepped["frames"]] == [100, 110]
    assert stepped["total"] == 20


def test_analytics_summary_shape(client, sample_bytes):
    """Catches analytics that miscounts frames, players, teams or heatmap grids."""
    meta = upload(client, sample_bytes).json()
    summary = client.get("/api/matches/%s/analytics" % meta["id"]).json()
    assert summary["frame_count"] == 750
    assert len(summary["players"]) > 10
    assert {"left", "right"} <= set(summary["teams"])
    for name in ("left", "right", "ball"):
        grid = summary["heatmaps"][name]
        assert len(grid) == 14, name
        for row in grid:
            assert len(row) == 21, name
            assert all(isinstance(v, int) and not isinstance(v, bool) for v in row), name


def test_csv_exports_have_one_row_per_datum(client, sample_bytes, sample_frames):
    """Catches an export whose row count drifts from the recorded data."""
    meta = upload(client, sample_bytes).json()
    positioned = sum(
        1
        for frame in sample_frames
        for person in frame.get("people") or []
        if isinstance(person.get("xy"), list) and len(person["xy"]) == 2
        and person["xy"][0] is not None and person["xy"][1] is not None
    )
    positions = client.get("/api/matches/%s/export.csv?kind=positions" % meta["id"])
    assert positions.status_code == 200
    lines = positions.text.splitlines()
    assert lines[0] == "frame,id,team,jersey,role,x,y"
    assert len(lines) - 1 == positioned
    players = client.get("/api/matches/%s/export.csv?kind=players" % meta["id"])
    assert players.status_code == 200
    summary = client.get("/api/matches/%s/analytics" % meta["id"]).json()
    assert len(players.text.splitlines()) - 1 == len(summary["players"])


def test_restarted_app_on_the_same_folder_lists_the_stored_match(tmp_path, sample_bytes):
    """Catches a store that does not persist to disk between app instances."""
    root = tmp_path / "matches"
    meta = upload(TestClient(library.create_library_app(root)), sample_bytes).json()
    again = TestClient(library.create_library_app(root))
    assert [item["id"] for item in again.get("/api/matches").json()["items"]] == [meta["id"]]
    assert again.get("/api/matches/%s" % meta["id"]).status_code == 200


def test_delete_removes_the_match_and_its_folder(client, app_root, sample_bytes):
    """Catches a delete that answers 204 but leaves the match listed or on disk."""
    meta = upload(client, sample_bytes).json()
    assert client.delete("/api/matches/%s" % meta["id"]).status_code == 204
    assert client.get("/api/matches/%s" % meta["id"]).status_code == 404
    assert client.get("/api/matches").json()["items"] == []
    assert list(app_root.iterdir()) == []
