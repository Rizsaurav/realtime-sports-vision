"""Hidden acceptance tests for the URL shortener (spec: llm/pi/specs/url_shortener.md)."""
import datetime
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest
import requests

WORKDIR = os.environ.get("APP_DIR", ".")
PORT = int(os.environ.get("TEST_PORT", "8123"))
BASE = f"http://127.0.0.1:{PORT}"
STATE = {}


def _start():
    env = dict(os.environ, PORT=str(PORT))
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=WORKDIR, env=env, stdout=subprocess.DEVNULL,
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
    for name in ("links.db", "links.db-wal", "links.db-shm", "links.db-journal"):
        pathlib.Path(WORKDIR, name).unlink(missing_ok=True)
    STATE["proc"] = _start()
    yield
    _stop(STATE["proc"])


def shorten(url="https://example.com/page", **extra):
    return requests.post(BASE + "/shorten", json={"url": url, **extra})


def go(code):
    return requests.get(f"{BASE}/{code}", allow_redirects=False)


def stats(code):
    return requests.get(f"{BASE}/api/links/{code}")


def iso(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def test_health():
    r = requests.get(BASE + "/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_shorten_response_shape():
    r = shorten()
    assert r.status_code == 201
    d = r.json()
    assert set(d) >= {"code", "short_url", "url", "created_at", "expires_at"}
    assert re.fullmatch(r"[A-Za-z0-9]{7}", d["code"])
    assert d["url"] == "https://example.com/page" and d["expires_at"] is None
    assert d["short_url"] == f"http://127.0.0.1:{PORT}/{d['code']}"
    iso(d["created_at"])


def test_redirect_is_302_with_location():
    code = shorten("https://example.com/a?x=1&y=two#frag").json()["code"]
    r = go(code)
    assert r.status_code == 302 and r.headers["Location"] == "https://example.com/a?x=1&y=two#frag"


def test_percent_encoded_url_is_preserved_in_location():
    url = "https://example.com/caf%C3%A9/%E6%97%A5%E6%9C%AC?q=%C3%BC&a=b%20c"
    code = shorten(url).json()["code"]
    assert go(code).headers["Location"] == url


def test_clicks_are_counted():
    code = shorten().json()["code"]
    assert stats(code).json()["clicks"] == 0
    for _ in range(3):
        go(code)
    d = stats(code).json()
    assert d["clicks"] == 3 and d["code"] == code and d["url"] == "https://example.com/page"
    assert set(d) >= {"code", "url", "clicks", "created_at", "expires_at"}


def test_unknown_code():
    r = go("zzzzzzz")
    assert r.status_code == 404 and "error" in r.json()
    assert stats("zzzzzzz").status_code == 404
    assert requests.delete(f"{BASE}/api/links/zzzzzzz").status_code == 404


def test_custom_code_and_conflict():
    r = shorten(custom_code="my-link_1")
    assert r.status_code == 201 and r.json()["code"] == "my-link_1"
    assert go("my-link_1").status_code == 302
    assert shorten(custom_code="my-link_1").status_code == 409


@pytest.mark.parametrize("code", ["api", "shorten", "health", "ab", "a" * 33, "has space", "bad!", "ünï"])
def test_invalid_custom_codes(code):
    r = shorten(custom_code=code)
    assert r.status_code == 400 and "error" in r.json()


def test_custom_codes_are_case_sensitive():
    a = shorten("https://one.example/", custom_code="CaseTest").json()
    b = shorten("https://two.example/", custom_code="casetest").json()
    assert a["code"] == "CaseTest" and b["code"] == "casetest"
    assert go("CaseTest").headers["Location"] == "https://one.example/"
    assert go("casetest").headers["Location"] == "https://two.example/"


@pytest.mark.parametrize("url", ["", "notaurl", "ftp://example.com/a", "javascript:alert(1)", "http://", "//example.com",
                                 "http://" + "x" * 2050, 123, None])
def test_invalid_urls(url):
    r = requests.post(BASE + "/shorten", json={"url": url})
    assert r.status_code == 400 and "error" in r.json()


def test_missing_url_and_bad_json():
    assert requests.post(BASE + "/shorten", json={}).status_code == 400
    r = requests.post(BASE + "/shorten", data="{nope", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and "error" in r.json()


@pytest.mark.parametrize("ttl", [0, -5, 1.5, "10", True, 31536001])
def test_invalid_ttl(ttl):
    r = shorten(ttl_seconds=ttl)
    assert r.status_code == 400 and "error" in r.json()


def test_valid_ttl_sets_expires_at():
    d = shorten(ttl_seconds=3600).json()
    delta = (iso(d["expires_at"]) - iso(d["created_at"])).total_seconds()
    assert abs(delta - 3600) <= 2


def test_expired_link_is_410_and_not_counted():
    code = shorten(ttl_seconds=1).json()["code"]
    assert go(code).status_code == 302
    time.sleep(2.3)
    r = go(code)
    assert r.status_code == 410 and "error" in r.json()
    s = stats(code)
    assert s.status_code == 200 and s.json()["clicks"] == 1


def test_delete_then_gone_and_code_reusable():
    code = shorten(custom_code="to-delete").json()["code"]
    r = requests.delete(f"{BASE}/api/links/{code}")
    assert r.status_code == 204 and r.content == b""
    assert go(code).status_code == 404 and stats(code).status_code == 404
    assert shorten("https://again.example/", custom_code="to-delete").status_code == 201


def test_same_url_twice_gives_different_codes():
    a = shorten("https://dup.example/").json()["code"]
    b = shorten("https://dup.example/").json()["code"]
    assert a != b


def test_generated_codes_are_unique():
    codes = {shorten(f"https://bulk.example/{i}").json()["code"] for i in range(60)}
    assert len(codes) == 60


def test_links_survive_restart_with_click_counts():
    code = shorten("https://persist.example/x", custom_code="persist-me").json()["code"]
    go(code)
    go(code)
    _stop(STATE["proc"])
    STATE["proc"] = _start()
    r = go(code)
    assert r.status_code == 302 and r.headers["Location"] == "https://persist.example/x"
    assert stats(code).json()["clicks"] == 3
