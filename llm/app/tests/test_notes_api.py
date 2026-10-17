"""Hidden acceptance tests for the notes service (spec: llm/app/specs/notes_api.md).

Every test checks behaviour the spec states. Tests use their own unique markers (tags or tokens) so they do not
depend on each other's data, except the final restart test.
"""
import datetime
import os
import pathlib
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
            if requests.get(BASE + "/health", timeout=1).status_code == 200:
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
    for name in ("notes.db", "notes.db-wal", "notes.db-shm", "notes.db-journal"):
        pathlib.Path(WORKDIR, name).unlink(missing_ok=True)
    STATE["proc"] = _start()
    yield
    _stop(STATE["proc"])


def uid():
    return uuid.uuid4().hex[:10]


def make(**body):
    r = requests.post(BASE + "/notes", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_health():
    r = requests.get(BASE + "/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_create_returns_201_and_note_fields():
    r = requests.post(BASE + "/notes", json={"title": "First", "body": "hello", "tags": ["a", "b"]})
    assert r.status_code == 201
    n = r.json()
    assert set(n) >= {"id", "title", "body", "tags", "created_at"}
    assert isinstance(n["id"], int) and n["title"] == "First" and n["body"] == "hello" and n["tags"] == ["a", "b"]
    datetime.datetime.fromisoformat(n["created_at"].replace("Z", "+00:00"))


def test_create_defaults():
    n = make(title="only title")
    assert n["body"] == "" and n["tags"] == []


@pytest.mark.parametrize("payload", [{}, {"title": ""}, {"title": 123}, {"title": None}, {"title": "x" * 101}])
def test_title_validation(payload):
    r = requests.post(BASE + "/notes", json=payload)
    assert r.status_code == 400
    assert "error" in r.json()


def test_title_of_100_characters_is_accepted():
    n = make(title="y" * 100)
    assert len(n["title"]) == 100


@pytest.mark.parametrize("payload", [{"title": "t", "body": 5}, {"title": "t", "tags": "a"},
                                     {"title": "t", "tags": [1]}, {"title": "t", "tags": ["a", 2]}])
def test_body_and_tags_validation(payload):
    r = requests.post(BASE + "/notes", json=payload)
    assert r.status_code == 400
    assert "error" in r.json()


def test_invalid_json_body_is_400():
    r = requests.post(BASE + "/notes", data="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_get_note_and_404():
    n = make(title="g1", body="b", tags=["t"])
    r = requests.get(f"{BASE}/notes/{n['id']}")
    assert r.status_code == 200 and r.json() == n
    r = requests.get(f"{BASE}/notes/99999999")
    assert r.status_code == 404 and "error" in r.json()


def test_list_is_ordered_by_id_and_total_counts_matches():
    tag = "ord" + uid()
    ids = [make(title=f"n{i}", tags=[tag])["id"] for i in range(5)]
    r = requests.get(BASE + "/notes", params={"tag": tag})
    assert r.status_code == 200
    body = r.json()
    assert [n["id"] for n in body["items"]] == ids and body["total"] == 5


def test_pagination_limit_offset_and_total():
    tag = "pg" + uid()
    ids = [make(title=f"p{i}", tags=[tag])["id"] for i in range(12)]
    page = requests.get(BASE + "/notes", params={"tag": tag, "limit": 2, "offset": 2}).json()
    assert [n["id"] for n in page["items"]] == ids[2:4] and page["total"] == 12
    default = requests.get(BASE + "/notes", params={"tag": tag}).json()
    assert len(default["items"]) == 10 and default["total"] == 12
    beyond = requests.get(BASE + "/notes", params={"tag": tag, "offset": 50}).json()
    assert beyond["items"] == [] and beyond["total"] == 12


@pytest.mark.parametrize("params", [{"limit": 101}, {"limit": 0}, {"limit": "abc"}, {"offset": -1}, {"offset": "x"}])
def test_limit_and_offset_validation(params):
    r = requests.get(BASE + "/notes", params=params)
    assert r.status_code == 400 and "error" in r.json()


def test_limit_100_is_allowed():
    assert requests.get(BASE + "/notes", params={"limit": 100}).status_code == 200


def test_search_is_case_insensitive_over_title_and_body():
    tok = "Zq" + uid()
    a = make(title=f"title {tok.upper()}")
    b = make(title="plain", body=f"body {tok.lower()}")
    make(title="unrelated")
    ids = [n["id"] for n in requests.get(BASE + "/notes", params={"q": tok}).json()["items"]]
    assert ids == [a["id"], b["id"]]


def test_percent_in_search_matches_literally():
    tok = "pct" + uid()
    a = make(title=f"{tok}% sure")
    make(title=f"{tok}x sure")
    items = requests.get(BASE + "/notes", params={"q": f"{tok}%"}).json()["items"]
    assert [n["id"] for n in items] == [a["id"]]


def test_underscore_in_search_matches_literally():
    tok = uid()
    a = make(title=f"tok_{tok}")
    make(title=f"tokX{tok}")
    items = requests.get(BASE + "/notes", params={"q": f"tok_{tok}"}).json()["items"]
    assert [n["id"] for n in items] == [a["id"]]


def test_quotes_in_search_are_safe():
    tok = uid()
    a = make(title=f"O'Brien {tok}")
    r = requests.get(BASE + "/notes", params={"q": f"O'Brien {tok}"})
    assert r.status_code == 200 and [n["id"] for n in r.json()["items"]] == [a["id"]]
    r = requests.get(BASE + "/notes", params={"q": "'; DROP TABLE notes;--"})
    assert r.status_code == 200 and r.json()["items"] == []
    assert requests.get(f"{BASE}/notes/{a['id']}").status_code == 200


def test_tag_filter_is_exact():
    t = "alpha" + uid()
    a = make(title="x", tags=[t])
    make(title="y", tags=[t + "0"])
    items = requests.get(BASE + "/notes", params={"tag": t}).json()["items"]
    assert [n["id"] for n in items] == [a["id"]]


def test_q_and_tag_combine_with_and():
    tag, tok = "cmb" + uid(), "tk" + uid()
    a = make(title=f"{tok} one", tags=[tag])
    make(title=f"{tok} two", tags=["other"])
    make(title="three", tags=[tag])
    body = requests.get(BASE + "/notes", params={"tag": tag, "q": tok}).json()
    assert [n["id"] for n in body["items"]] == [a["id"]] and body["total"] == 1


def test_put_replaces_and_keeps_id_and_created_at():
    n = make(title="old", body="o", tags=["x"])
    r = requests.put(f"{BASE}/notes/{n['id']}", json={"title": "new", "body": "n", "tags": ["y", "z"]})
    assert r.status_code == 200
    u = r.json()
    assert u["id"] == n["id"] and u["created_at"] == n["created_at"]
    assert u["title"] == "new" and u["body"] == "n" and u["tags"] == ["y", "z"]
    assert requests.get(f"{BASE}/notes/{n['id']}").json() == u


def test_put_errors():
    n = make(title="keep")
    assert requests.put(f"{BASE}/notes/{n['id']}", json={"title": ""}).status_code == 400
    assert requests.put(f"{BASE}/notes/99999999", json={"title": "t"}).status_code == 404


def test_put_defaults_for_missing_optional_fields():
    n = make(title="d", body="body", tags=["t"])
    u = requests.put(f"{BASE}/notes/{n['id']}", json={"title": "d2"}).json()
    assert u["body"] == "" and u["tags"] == []


def test_delete_returns_204_then_404():
    n = make(title="to delete")
    r = requests.delete(f"{BASE}/notes/{n['id']}")
    assert r.status_code == 204 and r.content == b""
    assert requests.get(f"{BASE}/notes/{n['id']}").status_code == 404
    assert requests.delete(f"{BASE}/notes/{n['id']}").status_code == 404


def test_unicode_round_trip():
    n = make(title="héllo 😀", body="日本語 ñ", tags=["ключ"])
    got = requests.get(f"{BASE}/notes/{n['id']}").json()
    assert got["title"] == "héllo 😀" and got["body"] == "日本語 ñ" and got["tags"] == ["ключ"]


def test_responses_are_json():
    r = requests.get(BASE + "/notes")
    assert "application/json" in r.headers.get("Content-Type", "")


def test_notes_survive_a_restart():
    n = make(title="persist me", body="still here", tags=["keep"])
    _stop(STATE["proc"])
    STATE["proc"] = _start()
    r = requests.get(f"{BASE}/notes/{n['id']}")
    assert r.status_code == 200 and r.json() == n
