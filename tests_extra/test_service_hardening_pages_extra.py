"""Extra adversarial tests for the hardening fixes (tasks/service_hardening.md).

Two regressions the acceptance suite does not pin:

- the served pages carry no inline style= attribute: the CSP's
  style-src has no 'unsafe-inline' and no hashes, so a style= attribute
  in the page markup is blocked by the browser. The library page's
  upload progress bar regressed this way (an empty, invisible div), so
  the page did not fully work under the policy the app itself serves
  (spec item 7: "The pages must still work");
- the per-client rate limit (60 non-GET requests per minute, 429 with
  Retry-After - spec item 5) end to end: the 61st non-GET of one client
  is a 429 with a positive Retry-After, GETs keep being served while the
  client is limited, and the same client is served again once the
  window has passed (the clock is advanced instead of sleeping 60 s).
"""
import os
import re
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sideline import worker as W  # noqa: E402
from sideline import library as L  # noqa: E402


def _worker_app(tmp_path):
    return W.create_worker_app(str(tmp_path))


def test_served_pages_have_no_inline_style_attribute(tmp_path):
    app = L.create_library_app(str(tmp_path))
    client = TestClient(app)
    for name in ("library.html", "match.html"):
        html = client.get("/" + name).text
        # an HTML attribute style= (any case, any surrounding whitespace)
        # is blocked by the served CSP, which allows no inline styles
        assert not re.search(r"\sstyle\s*=", html, re.I), (
            name + " carries an inline style= attribute the served CSP blocks")
        # and the page's upload progress bar is styled from the
        # same-origin stylesheet the page links
        if name == "library.html":
            assert 'id="uploadProgress" class="upload-progress"' in html
            assert 'id="uploadProgressBar" class="upload-progress-bar"' in html


def test_progress_bar_style_lives_in_the_same_origin_stylesheet(tmp_path):
    web = os.path.join(os.path.dirname(L.__file__), "web")
    css = open(os.path.join(web, "pages.css"), encoding="utf-8").read()
    assert ".upload-progress" in css
    assert ".upload-progress-bar" in css
    client = TestClient(L.create_library_app(str(tmp_path)))
    response = client.get("/static/pages.css")
    assert response.status_code == 200
    assert ".upload-progress-bar" in response.text


def test_worker_rate_limit_429_get_served_and_window_releases(tmp_path, monkeypatch):
    app = _worker_app(tmp_path)
    try:
        client = TestClient(app)
        now = [1000.0]
        monkeypatch.setattr(W.time, "monotonic", lambda: now[0])
        # 60 non-GET requests fill the client's minute
        for _ in range(60):
            assert client.post("/jobs", data={"x": "y"}).status_code == 400
        # the 61st is refused with a Retry-After header
        response = client.post("/jobs", data={"x": "y"})
        assert response.status_code == 429
        retry_after = int(response.headers["retry-after"])
        assert 1 <= retry_after <= 60
        assert "error" in response.json()
        # a GET is never counted: the service keeps serving it
        assert client.get("/health").status_code == 200
        # still inside the window: the next non-GET is refused too
        assert client.post("/jobs", data={"x": "y"}).status_code == 429
        # once the window has passed, the same client is served again
        now[0] += 60.0
        assert client.post("/jobs", data={"x": "y"}).status_code == 400
    finally:
        app.state.worker.stop()


def test_library_rate_limit_429_gets_served_while_limited(tmp_path):
    app = L.create_library_app(str(tmp_path))
    client = TestClient(app)
    headers = {"content-type": "application/octet-stream"}
    for _ in range(60):
        assert client.post("/api/matches", content=b"",
                           headers=headers).status_code == 400
    response = client.post("/api/matches", content=b"", headers=headers)
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1
    # GETs stay open for a limited client
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/limits").status_code == 200
