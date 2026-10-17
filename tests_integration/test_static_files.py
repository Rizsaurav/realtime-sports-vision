"""Static assets: everything the pages load is served and shipped (task item 3)."""
from sideline import library

from page_scan import PAGES, loaded_static_urls


def closure():
    """All static URLs the two pages load, following module imports."""
    return loaded_static_urls(
        [page.read_text(encoding="utf-8") for page in PAGES.values()],
        lambda url: (library.WEB_DIR / url.removeprefix("/static/")).read_text(encoding="utf-8"),
    )


# The assets the pages are known to load; pinning them keeps the scan honest.
CORE_ASSETS = {
    "/static/style.css",
    "/static/pages.css",
    "/static/lib/render.js",
    "/static/lib/replay.js",
    "/static/lib/pitch.js",
}


def test_every_asset_the_pages_load_answers_200(client):
    """Catches a page loading a script or stylesheet the app does not serve."""
    urls = closure()
    assert CORE_ASSETS <= set(urls), "scan missed: %s" % sorted(CORE_ASSETS - set(urls))
    for url in sorted(set(urls)):
        assert client.get(url).status_code == 200, url


def test_every_module_import_target_is_shipped():
    """Catches an import from a /static path that is not shipped in the web folder."""
    for url in sorted(set(closure())):
        assert (library.WEB_DIR / url.removeprefix("/static/")).is_file(), url
