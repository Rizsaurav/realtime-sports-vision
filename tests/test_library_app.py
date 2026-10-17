"""Acceptance tests for the match library app (spec: tasks/library_server.md)."""
import io
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from fastapi.testclient import TestClient  # noqa: E402

from sideline import library  # noqa: E402


def make_recording(n=60):
    """A tiny but realistic recording: one left and one right player moving, a ball, team colours."""
    lines = []
    for f in range(n):
        people = [
            {"id": 1, "role": "player", "team": "left", "jersey": "9", "box": [10, 10, 20, 40], "xy": [-20.0 + 0.2 * f, 5.0]},
            {"id": 2, "role": "player", "team": "left", "jersey": "4", "box": [10, 10, 20, 40], "xy": [-10.0, -6.0 + 0.1 * f]},
            {"id": 3, "role": "player", "team": "left", "jersey": "7", "box": [10, 10, 20, 40], "xy": [-15.0, 12.0]},
            {"id": 4, "role": "player", "team": "right", "jersey": "10", "box": [10, 10, 20, 40], "xy": [20.0, -5.0 + 0.2 * f]},
            {"id": 5, "role": "referee", "team": None, "jersey": None, "box": [10, 10, 20, 40], "xy": [0.0, 0.0]},
        ]
        ball = {"box": [5, 5, 6, 6], "xy": [-5.0 + 0.3 * f, 1.0]} if f % 2 == 0 else None
        lines.append(json.dumps({"frame": f, "delay": 0, "people": people, "ball": ball,
                                 "colors": {"left": "#d03030", "right": "#2050c0"}}))
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture()
def client(tmp_path):
    return TestClient(library.create_library_app(tmp_path / "matches"))


def upload(client, data=None, name="game"):
    return client.post("/api/matches", files={"file": (name + ".jsonl", data or make_recording(), "application/json")},
                       data={"name": name})


def test_root_redirects_to_the_library_page(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/library.html"


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/library.html", "/match.html"])
def test_pages_are_served_as_html(client, path):
    r = client.get(path)
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()


@pytest.mark.parametrize("path", ["/static/lib/render.js", "/static/lib/replay.js"])
def test_libraries_are_served_under_static(client, path):
    r = client.get(path)
    assert r.status_code == 200 and "export" in r.text
    assert "javascript" in r.headers["content-type"]


def test_pages_load_the_libraries_from_static(client):
    assert "/static/lib/render.js" in client.get("/library.html").text
    assert "/static/lib/render.js" in client.get("/match.html").text


def test_unknown_static_file_is_404(client):
    assert client.get("/static/lib/nothing.js").status_code == 404
    assert client.get("/nothing.html").status_code == 404


def test_root_folder_is_created(tmp_path):
    root = tmp_path / "a" / "b"
    library.create_library_app(root)
    assert root.is_dir()


def test_accepts_str_and_path_roots(tmp_path):
    for root in (str(tmp_path / "s"), tmp_path / "p"):
        c = TestClient(library.create_library_app(root))
        assert c.get("/healthz").status_code == 200


def test_upload_list_and_open(client):
    r = upload(client)
    assert r.status_code == 201
    mid = r.json()["id"]
    assert [m["id"] for m in client.get("/api/matches").json()["items"]] == [mid]
    meta = client.get(f"/api/matches/{mid}").json()
    assert meta["name"] == "game" and meta["frame_count"] == 60 and meta["colors"]["left"] == "#d03030"


def test_round_trip_analytics_uses_the_real_module(client):
    mid = upload(client).json()["id"]
    a = client.get(f"/api/matches/{mid}/analytics").json()
    assert a["frame_count"] == 60 and a["fps"] == 25
    left = [p for p in a["players"] if p["id"] == 1][0]
    assert left["team"] == "left" and left["jersey"] == "9"
    assert abs(left["distance_m"] - 0.2 * 59) < 0.05            # 59 steps of 0.2 m
    assert set(a["teams"]) == {"left", "right"} and a["teams"]["left"]["frames"] == 60
    assert a["heatmaps"]["grid"]["nx"] == 21 and len(a["heatmaps"]["left"]) == 14
    assert a["ball"]["frames_with_ball"] == 30


def test_round_trip_exports(client):
    mid = upload(client).json()["id"]
    pos = client.get(f"/api/matches/{mid}/export.csv?kind=positions")
    assert pos.status_code == 200 and pos.text.splitlines()[0] == "frame,id,team,jersey,role,x,y"
    assert len(pos.text.splitlines()) == 1 + 60 * 5
    players = client.get(f"/api/matches/{mid}/export.csv?kind=players")
    assert players.text.splitlines()[0] == "id,team,jersey,role,frames,seconds,distance_m,avg_speed_ms,max_speed_ms,sprints"
    assert len(players.text.splitlines()) == 1 + 5


def test_round_trip_frames_slice(client):
    mid = upload(client).json()["id"]
    d = client.get(f"/api/matches/{mid}/frames?start=10&end=19&step=3").json()
    assert [f["frame"] for f in d["frames"]] == [10, 13, 16, 19] and d["total"] == 10


def test_delete_round_trip(client):
    mid = upload(client).json()["id"]
    assert client.delete(f"/api/matches/{mid}").status_code == 204
    assert client.get(f"/api/matches/{mid}").status_code == 404
    assert client.get("/api/matches").json()["items"] == []


def test_matches_persist_between_app_instances(tmp_path):
    root = tmp_path / "m"
    mid = upload(TestClient(library.create_library_app(root))).json()["id"]
    again = TestClient(library.create_library_app(root))
    assert [m["id"] for m in again.get("/api/matches").json()["items"]] == [mid]


def test_library_module_does_not_import_heavy_packages():
    code = ("import sys; sys.path.insert(0, %r); import sideline.library; "
            "bad = [m for m in ('torch', 'cv2', 'numpy', 'sideline.pipeline') if m in sys.modules]; "
            "print(','.join(bad)); sys.exit(1 if bad else 0)") % os.environ.get("SIDELINE_SRC", "src")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_parser_has_the_library_subcommand_with_defaults():
    from sideline import __main__ as cli
    parser = cli.build_parser()
    a = parser.parse_args(["library", "--root", "/tmp/x"])
    assert a.cmd == "library" and a.root == "/tmp/x" and a.host == "127.0.0.1" and a.port == 8100
    b = parser.parse_args(["library", "--root", "r", "--host", "0.0.0.0", "--port", "9000"])
    assert b.host == "0.0.0.0" and b.port == 9000


def test_cli_keeps_the_existing_subcommands():
    from sideline import __main__ as cli
    parser = cli.build_parser()
    assert parser.parse_args(["serve"]).cmd == "serve"
    assert parser.parse_args(["run", "clip.mp4"]).cmd == "run"
    with pytest.raises(SystemExit):
        parser.parse_args(["library"])                             # --root is required


def test_readme_documents_the_library():
    text = open("README.md", encoding="utf-8").read()
    assert "## Match library" in text
    assert "python -m sideline library" in text
    assert text.index("## Live app") < text.index("## Match library")
