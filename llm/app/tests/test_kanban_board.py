"""Hidden acceptance tests for the kanban board (spec: llm/app/specs/kanban_board.md)."""
import os
import pathlib
import re
import subprocess
import sys
import time
import uuid

import pytest
import requests

WORKDIR = os.environ.get("APP_DIR", "/workspace")
BASE = "http://127.0.0.1:8000"
STATE = {}


def _start():
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=WORKDIR, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    deadline = time.time() + 40
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            if requests.get(BASE + "/api/board", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            time.sleep(0.3)
    err = proc.stderr.read().decode()[-1500:] if proc.poll() is not None else "(still running)"
    proc.kill()
    raise RuntimeError("server did not start: " + err)


def _stop(proc):
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module", autouse=True)
def server():
    for name in ("board.db", "board.db-wal", "board.db-shm", "board.db-journal"):
        pathlib.Path(WORKDIR, name).unlink(missing_ok=True)
    STATE["proc"] = _start()
    yield
    _stop(STATE["proc"])


def board():
    r = requests.get(BASE + "/api/board")
    assert r.status_code == 200
    return r.json()["columns"]


def column(name):
    return next(c for c in board() if c["name"] == name)


def new_column():
    name = "col-" + uuid.uuid4().hex[:8]
    r = requests.post(BASE + "/api/columns", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


def new_card(col, title, **extra):
    r = requests.post(BASE + "/api/cards", json={"column_id": col["id"], "title": title, **extra})
    assert r.status_code == 201, r.text
    return r.json()


def titles(col_id):
    return [t["title"] for c in board() if c["id"] == col_id for t in c["cards"]]


def test_default_board_on_first_start():
    cols = board()
    assert [c["name"] for c in cols][:3] == ["To Do", "In Progress", "Done"]
    assert all(isinstance(c["id"], int) and c["cards"] == [] for c in cols[:3])


def test_create_column_appended_at_end():
    c = new_column()
    assert set(c) >= {"id", "name"}
    assert board()[-1]["id"] == c["id"]


@pytest.mark.parametrize("payload", [{}, {"name": ""}])
def test_column_name_validation(payload):
    assert requests.post(BASE + "/api/columns", json=payload).status_code == 400


def test_duplicate_column_name_ignoring_case_is_409():
    c = new_column()
    r = requests.post(BASE + "/api/columns", json={"name": c["name"].upper()})
    assert r.status_code == 409


def test_add_card_fields_defaults_and_order():
    col = new_column()
    a = new_card(col, "first")
    assert set(a) >= {"id", "title", "description", "column_id"}
    assert a["description"] == "" and a["column_id"] == col["id"] and isinstance(a["id"], int)
    new_card(col, "second", description="d")
    new_card(col, "third")
    assert titles(col["id"]) == ["first", "second", "third"]
    assert [c for c in board() if c["id"] == col["id"]][0]["cards"][1]["description"] == "d"


def test_add_card_errors():
    col = new_column()
    assert requests.post(BASE + "/api/cards", json={"column_id": col["id"], "title": ""}).status_code == 400
    assert requests.post(BASE + "/api/cards", json={"column_id": col["id"]}).status_code == 400
    assert requests.post(BASE + "/api/cards", json={"column_id": 99999999, "title": "x"}).status_code == 404


def test_patch_card_updates_only_given_fields():
    col = new_column()
    c = new_card(col, "orig", description="desc")
    r = requests.patch(f"{BASE}/api/cards/{c['id']}", json={"title": "renamed"})
    assert r.status_code == 200
    got = r.json()
    assert got["title"] == "renamed" and got["description"] == "desc" and got["id"] == c["id"]
    r = requests.patch(f"{BASE}/api/cards/{c['id']}", json={"description": "new desc"})
    assert r.json()["title"] == "renamed" and r.json()["description"] == "new desc"


def test_patch_card_errors():
    col = new_column()
    c = new_card(col, "x")
    assert requests.patch(f"{BASE}/api/cards/{c['id']}", json={"title": ""}).status_code == 400
    assert requests.patch(f"{BASE}/api/cards/99999999", json={"title": "x"}).status_code == 404


def test_move_within_column_and_clamping():
    col = new_column()
    for t in "abc":
        new_card(col, t)
    cid = [t for c in board() if c["id"] == col["id"] for t in c["cards"] if t["title"] == "c"][0]["id"]
    r = requests.post(f"{BASE}/api/cards/{cid}/move", json={"column_id": col["id"], "position": 0})
    assert r.status_code == 200
    assert titles(col["id"]) == ["c", "a", "b"]
    requests.post(f"{BASE}/api/cards/{cid}/move", json={"column_id": col["id"], "position": 99})
    assert titles(col["id"]) == ["a", "b", "c"]
    requests.post(f"{BASE}/api/cards/{cid}/move", json={"column_id": col["id"], "position": -5})
    assert titles(col["id"]) == ["c", "a", "b"]


def test_move_across_columns_inserts_at_position():
    src, dst = new_column(), new_column()
    for t in ("s1", "s2", "s3"):
        new_card(src, t)
    for t in ("d1", "d2"):
        new_card(dst, t)
    s2 = [t for c in board() if c["id"] == src["id"] for t in c["cards"] if t["title"] == "s2"][0]["id"]
    r = requests.post(f"{BASE}/api/cards/{s2}/move", json={"column_id": dst["id"], "position": 1})
    assert r.status_code == 200 and r.json()["column_id"] == dst["id"]
    assert titles(src["id"]) == ["s1", "s3"]
    assert titles(dst["id"]) == ["d1", "s2", "d2"]


def test_move_errors():
    col = new_column()
    c = new_card(col, "m")
    assert requests.post(f"{BASE}/api/cards/99999999/move", json={"column_id": col["id"], "position": 0}).status_code == 404
    assert requests.post(f"{BASE}/api/cards/{c['id']}/move", json={"column_id": 99999999, "position": 0}).status_code == 404


def test_delete_card():
    col = new_column()
    c = new_card(col, "bye")
    r = requests.delete(f"{BASE}/api/cards/{c['id']}")
    assert r.status_code == 204 and r.content == b""
    assert titles(col["id"]) == []
    assert requests.delete(f"{BASE}/api/cards/{c['id']}").status_code == 404


def test_delete_column_removes_its_cards():
    col = new_column()
    c = new_card(col, "inside")
    r = requests.delete(f"{BASE}/api/columns/{col['id']}")
    assert r.status_code == 204
    assert all(x["id"] != col["id"] for x in board())
    assert requests.delete(f"{BASE}/api/cards/{c['id']}").status_code == 404
    assert requests.delete(f"{BASE}/api/columns/{col['id']}").status_code == 404


def test_api_responses_are_json():
    assert "application/json" in requests.get(BASE + "/api/board").headers.get("Content-Type", "")


def test_index_page_is_html_with_board_container_and_fetch():
    r = requests.get(BASE + "/")
    assert r.status_code == 200 and "text/html" in r.headers.get("Content-Type", "")
    html = r.text
    assert 'id="board"' in html or "id='board'" in html
    scripts = re.findall(r'<script[^>]+src=["\'](/[^"\']+)["\']', html)
    blob = html + "".join(requests.get(BASE + s).text for s in scripts)
    assert "/api/board" in blob and "fetch(" in blob
    assert "draggable" in blob


def test_local_assets_are_served():
    html = requests.get(BASE + "/").text
    paths = re.findall(r'(?:src|href)=["\'](/[^"\'#?]+)["\']', html)
    assert all(requests.get(BASE + p).status_code == 200 for p in paths)


def test_state_survives_restart():
    col = new_column()
    for t in ("one", "two", "three"):
        new_card(col, t)
    _stop(STATE["proc"])
    STATE["proc"] = _start()
    assert titles(col["id"]) == ["one", "two", "three"]
