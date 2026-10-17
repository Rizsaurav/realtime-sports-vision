"""Record the real library API responses as fixtures for the page smoke tests.

Starts ``sideline.library.create_library_app`` in a temporary folder, uploads
``tests/fixtures/sample_match.jsonl`` through the real upload endpoint and
stages the resulting responses, which replace ``web_tests_smoke/fixtures/``
only after every check passes, so a failed run never leaves a mixed old/new
set behind:

* ``matches.json``       -- GET /api/matches with one stored match
                            (the real ``{"items": [...]}`` shape)
* ``empty_matches.json`` -- GET /api/matches on an empty library
* ``match.json``         -- GET /api/matches/{id}
* ``analytics.json``     -- GET /api/matches/{id}/analytics
* ``frames_0_249.json``  -- GET /api/matches/{id}/frames?start=0&end=249
                            (the first 250 frames)
* ``limits.json``        -- GET /api/limits on a library without a worker
* ``limits_video.json``  -- GET /api/limits on a library with an injected
                            (in-process fake) worker client

Run with: ``python tests_integration/make_fixtures.py``
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from sideline import library  # noqa: E402

SAMPLE = REPO / "tests" / "fixtures" / "sample_match.jsonl"
FIXTURES = REPO / "web_tests_smoke" / "fixtures"
NAME = "Sample match"


class _NullWorker:
    """An injected worker client: only the limits endpoint reads it.

    It carries the worker's default upload limit so the video-enabled
    limits fixture reports the number the page would show.
    """

    max_bytes = 2 * 1024 ** 3

# The API stamps the current UTC time into meta["created_at"]. Pinning it to
# a fixed value before writing keeps regenerated fixtures byte-identical.
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"


def check(condition, message):
    if not condition:
        raise SystemExit("make_fixtures: " + message)


def normalize(payload):
    """Recursively pin volatile fields (created_at) to fixed values."""
    if isinstance(payload, dict):
        return {k: (FIXED_CREATED_AT if k == "created_at" else normalize(v))
                for k, v in payload.items()}
    if isinstance(payload, list):
        return [normalize(v) for v in payload]
    return payload


def write_fixture(stage, name, payload):
    (stage / name).write_text(json.dumps(normalize(payload), indent=2) + "\n", encoding="utf-8")


def install(stage):
    """Swap the staged fixtures into FIXTURES, one atomic rename each.

    Copy to a temp name inside FIXTURES first, because the stage directory
    may live on another filesystem.
    """
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for path in sorted(stage.iterdir()):
        if path.is_file():
            tmp = FIXTURES / (path.name + ".tmp")
            shutil.copyfile(str(path), str(tmp))
            os.replace(str(tmp), str(FIXTURES / path.name))


def main():
    check(SAMPLE.is_file(), "missing %s" % SAMPLE)

    # Stage everything in a temp directory; only a run that passes every
    # check installs it, so a failed check leaves the committed fixtures
    # untouched (the page tests derive their expectations from them).
    stage = Path(tempfile.mkdtemp(prefix="sideline-smoke-fixtures-"))
    try:
        # The empty list comes from a fresh, empty library.
        with tempfile.TemporaryDirectory(prefix="sideline-smoke-empty-") as tmp:
            client = TestClient(library.create_library_app(Path(tmp) / "matches"))
            res = client.get("/api/matches")
            check(res.status_code == 200 and res.json() == {"items": []},
                  "unexpected empty library list: %s %s" % (res.status_code, res.text))
            write_fixture(stage, "empty_matches.json", res.json())

        with tempfile.TemporaryDirectory(prefix="sideline-smoke-") as tmp:
            client = TestClient(library.create_library_app(Path(tmp) / "matches"))

            # The limits answer the page reads at start (no worker here:
            # video processing not configured).
            res = client.get("/api/limits")
            check(res.status_code == 200 and isinstance(res.json(), dict)
                  and res.json().get("video_enabled") is False
                  and isinstance(res.json().get("max_upload_bytes"), int)
                  and isinstance(res.json().get("max_video_bytes"), int),
                  "unexpected limits: %s %s" % (res.status_code, res.text))
            write_fixture(stage, "limits.json", res.json())

            res = client.post(
                "/api/matches",
                files={"file": ("sample_match.jsonl", SAMPLE.read_bytes(), "application/json")},
                data={"name": NAME},
            )
            check(res.status_code == 201, "upload failed: %s %s" % (res.status_code, res.text))
            meta = res.json()
            match_id = meta["id"]

            res = client.get("/api/matches")
            body = res.json()
            check(res.status_code == 200 and isinstance(body, dict) and isinstance(body.get("items"), list)
                  and [m["id"] for m in body["items"]] == [match_id],
                  "unexpected match list: %s" % res.text)
            write_fixture(stage, "matches.json", body)

            res = client.get("/api/matches/%s" % match_id)
            check(res.status_code == 200 and res.json() == meta,
                  "unexpected match meta: %s" % res.text)
            write_fixture(stage, "match.json", res.json())

            res = client.get("/api/matches/%s/analytics" % match_id)
            summary = res.json()
            check(res.status_code == 200 and isinstance(summary, dict)
                  and summary.get("frame_count") == meta["frame_count"]
                  and len(summary.get("players") or []) > 0
                  and "left" in (summary.get("teams") or {}) and "right" in (summary.get("teams") or {})
                  and "left" in (summary.get("heatmaps") or {}),
                  "unexpected analytics summary: %s" % res.text[:200])
            write_fixture(stage, "analytics.json", summary)

            res = client.get("/api/matches/%s/frames" % match_id, params={"start": 0, "end": 249})
            body = res.json()
            check(res.status_code == 200 and body.get("total") == 250
                  and len(body.get("frames") or []) == 250
                  and body["frames"][0]["frame"] == 0 and body["frames"][-1]["frame"] == 249,
                  "unexpected first chunk: %s" % res.text[:200])
            write_fixture(stage, "frames_0_249.json", body)

        # The video-enabled limits answer, from a library that has an
        # injected worker client (no worker process involved).
        with tempfile.TemporaryDirectory(prefix="sideline-smoke-video-") as tmp:
            client = TestClient(library.create_library_app(Path(tmp) / "matches",
                                                           worker_client=_NullWorker()))
            res = client.get("/api/limits")
            check(res.status_code == 200 and isinstance(res.json(), dict)
                  and res.json().get("video_enabled") is True
                  and res.json().get("max_video_bytes") == _NullWorker.max_bytes,
                  "unexpected video limits: %s %s" % (res.status_code, res.text))
            write_fixture(stage, "limits_video.json", res.json())

        install(stage)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
