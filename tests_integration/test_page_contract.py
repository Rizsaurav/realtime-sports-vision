"""Page-to-API contract: every URL the pages request is served (task item 2)."""
import re

from sideline import library

from page_scan import PAGES, page_api_calls


def route_table(app):
    """(method, normalized path template) pairs the app serves."""
    table = set()
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not path:
            continue
        for method in methods:
            if method in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                table.add((method, re.sub(r"\{[^/}]+\}", "{id}", path)))
    return table


# The requests the pages are known to make; pinning them keeps the url
# extraction in page_scan honest instead of silently matching nothing.
CORE_CALLS = {
    ("GET", "/api/matches"),
    ("POST", "/api/matches"),
    ("DELETE", "/api/matches/{id}"),
    ("GET", "/api/matches/{id}"),
    ("GET", "/api/matches/{id}/analytics"),
    ("GET", "/api/matches/{id}/frames"),
    ("GET", "/api/matches/{id}/export.csv"),
}


def test_pages_only_request_routes_the_app_serves(tmp_path):
    """Catches a page fetching an endpoint or method the app does not serve (404/405)."""
    used = set()
    for page in PAGES.values():
        used.update(page_api_calls(page.read_text(encoding="utf-8")))
    assert CORE_CALLS <= used, "extraction missed: %s" % sorted(CORE_CALLS - used)
    table = route_table(library.create_library_app(tmp_path / "m"))
    missing = sorted(call for call in used if call not in table)
    assert not missing, "page requests with no matching route and method: %s" % missing


def upload(client, data, name="sample"):
    return client.post(
        "/api/matches",
        files={"file": ("sample.jsonl", data, "application/octet-stream")},
        data={"name": name},
    )


def test_list_is_an_object_with_items_and_the_page_reads_items(client, sample_bytes):
    """Catches the page expecting a bare array while the API wraps it in an items object."""
    upload(client, sample_bytes)
    body = client.get("/api/matches").json()
    assert isinstance(body, dict) and isinstance(body.get("items"), list)
    library_text = PAGES["library"].read_text(encoding="utf-8")
    assert "matches.items" in library_text  # the page parses the items key


def test_analytics_fields_the_match_page_reads_exist_in_a_real_response(client, sample_bytes):
    """Catches the match page reading analytics fields the API does not return."""
    meta = upload(client, sample_bytes).json()
    match_text = PAGES["match"].read_text(encoding="utf-8")
    fields = set(re.findall(r"\bsummary\.(\w+)", match_text))
    assert {"players", "teams", "heatmaps"} <= fields
    heatmap_fields = set(re.findall(r"\bheat\.(\w+)", match_text))
    assert {"left", "right", "ball"} <= heatmap_fields
    summary = client.get("/api/matches/%s/analytics" % meta["id"]).json()
    for field in fields:
        assert field in summary, field
    for field in heatmap_fields:
        assert field in summary["heatmaps"], field
