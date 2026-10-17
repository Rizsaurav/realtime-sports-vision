"""Failure behaviour: unknown ids, malformed uploads, path traversal (task item 4)."""
import pytest
from fastapi.testclient import TestClient

from sideline import library

# A well-formed id that was never stored.
UNKNOWN = "m_0000000000"


def test_unknown_match_id_is_404_on_every_endpoint(client):
    """Catches an id endpoint that errors or serves data for an id that was never stored."""
    for method, path in (
        ("GET", "/api/matches/{id}"),
        ("GET", "/api/matches/{id}/frames"),
        ("GET", "/api/matches/{id}/analytics"),
        ("GET", "/api/matches/{id}/export.csv"),
        ("DELETE", "/api/matches/{id}"),
    ):
        assert client.request(method, path.format(id=UNKNOWN)).status_code == 404, (
            method, path)


BAD_UPLOADS = {
    "invalid json line": b'{"frame": 0, "people": []}\n{oops\n',
    "frames not increasing": b'{"frame": 5, "people": []}\n{"frame": 5, "people": []}\n',
    "frame not an integer": b'{"frame": "1", "people": []}\n',
    "people missing": b'{"frame": 1}\n',
    "no frames at all": b"\n\n",
    "not utf-8": b"\xff\xfe\x00\x01",
}


@pytest.mark.parametrize("payload", list(BAD_UPLOADS.values()), ids=list(BAD_UPLOADS))
def test_malformed_upload_is_400_and_leaves_no_folder(tmp_path, payload):
    """Catches a rejected upload that still creates a match folder or file on disk."""
    root = tmp_path / "matches"
    client = TestClient(library.create_library_app(root))
    response = client.post(
        "/api/matches",
        files={"file": ("bad.jsonl", payload, "application/octet-stream")},
        data={"name": "bad"},
    )
    assert response.status_code == 400
    assert list(root.iterdir()) == []


def test_upload_without_a_file_field_is_400_and_leaves_no_folder(tmp_path):
    """Catches a missing file field being accepted or half-stored."""
    root = tmp_path / "matches"
    client = TestClient(library.create_library_app(root))
    assert client.post("/api/matches", data={"name": "no file"}).status_code == 400
    assert list(root.iterdir()) == []


TRAVERSAL_URLS = (
    "/api/matches/..",
    "/api/matches/..%2F..%2Fetc",
    "/api/matches/%2e%2e",
    "/api/matches/..%2F..%2F..%2F..%2Fetc%2Fpasswd",
)


def test_path_traversal_id_is_404_and_touches_nothing_outside_the_root(tmp_path):
    """Catches an id endpoint that resolves or writes outside the store root."""
    root = tmp_path / "matches"
    client = TestClient(library.create_library_app(root))
    outside_before = sorted(entry.name for entry in tmp_path.iterdir())
    for url in TRAVERSAL_URLS:
        for method in ("GET", "DELETE"):
            assert client.request(method, url).status_code == 404, (method, url)
    outside_after = sorted(entry.name for entry in tmp_path.iterdir())
    assert outside_after == outside_before
    assert list(root.iterdir()) == []
