"""Read the two library pages as text and derive what they request.

The "read the code" half of the contract tests: these helpers turn page
source into (method, url) request pairs, with {id} marking the match-id
slot, and into the set of static assets the pages load (scripts,
stylesheets and the modules they import).
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGES = {
    "library": REPO / "src" / "sideline" / "web" / "library.html",
    "match": REPO / "src" / "sideline" / "web" / "match.html",
}

_CONST_RE = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*([^;\n]+);")
_TOKEN_RE = re.compile(
    r'["\']([^"\']*)["\']|\b(?:enc|encodeURIComponent)\s*\(|\b([A-Za-z_$][\w$]*)\b')
_HREF_RE = re.compile(r"\.href\s*=\s*([^;]+);")


def _call_bodies(text, name):
    """Argument bodies of every call to name(...), balanced over parentheses."""
    bodies = []
    for match in re.finditer(r"\b%s\s*\(" % re.escape(name), text):
        i, depth = match.end(), 1
        while i < len(text) and depth:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        bodies.append(text[match.end():i - 1])
    return bodies


def _split_args(body):
    args, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(body[start:i])
            start = i + 1
    args.append(body[start:])
    return [arg.strip() for arg in args]


def _method_of(options):
    match = re.search(r"method\s*:\s*[\"'](\w+)[\"']", options or "")
    return match.group(1).upper() if match else "GET"


def const_url_templates(text):
    """Template of every simple const that builds an absolute URL (e.g. `base`)."""
    consts = {}
    for match in _CONST_RE.finditer(text):
        template = url_template(match.group(2), {})
        if template:
            consts[match.group(1)] = template
    return consts


def url_template(expr, consts):
    """Absolute URL that an expression builds, {id} where the match id is encoded.

    Returns None when the expression does not build an absolute URL, so call
    sites that pass a bare variable (fetchJson's own `url` parameter) drop
    out of the result.
    """
    if expr is None:
        return None
    expr = expr.strip()
    simple = re.fullmatch(r"([A-Za-z_$][\w$]*)", expr)
    if simple and simple.group(1) in consts:
        return consts[simple.group(1)]
    parts = []
    for tok in _TOKEN_RE.finditer(expr):
        literal, ident = tok.group(1), tok.group(2)
        if literal is not None:
            if not literal.startswith("/"):
                if not parts:
                    return None  # relative start: not an absolute URL
                continue  # query fragments fill parameters, not the path
            parts.append(literal)
        elif ident is None:
            if not parts:
                return None  # a URL starting with the encoded id
            parts.append("{id}")
        elif ident in consts:
            parts.append(consts[ident])
        # other identifiers (start, end, ...) fill query parameters
    template = "".join(parts)
    if not template.startswith("/"):
        return None
    return template.split("?", 1)[0]


def page_api_calls(text):
    """(method, url template) pairs for every request the page makes.

    fetch(...) calls contribute their own method (default GET), fetchJson(...)
    calls are GET, and `.href = ...` assignments are GET navigations (the
    download links). Templates carry {id} where the page plugs in the match
    id, including `base = "/api/matches/" + enc(id)` style concatenations.
    """
    consts = const_url_templates(text)
    calls = []
    for body in _call_bodies(text, "fetch"):
        args = _split_args(body)
        if args:
            template = url_template(args[0], consts)
            if template:
                calls.append((_method_of(args[1] if len(args) > 1 else ""), template))
    for body in _call_bodies(text, "fetchJson"):
        args = _split_args(body)
        if args:
            template = url_template(args[0], consts)
            if template:
                calls.append(("GET", template))
    for match in _HREF_RE.finditer(text):
        template = url_template(match.group(1), consts)
        if template:
            calls.append(("GET", template))
    return calls


# ---------------------------------------------------------------- assets


def page_asset_urls(html):
    """Absolute URLs of the scripts and stylesheets a page loads."""
    urls = re.findall(r'<script[^>]+src="(/[^"]+)"', html)
    for tag in re.findall(r"<link\b[^>]*>", html):
        if 'rel="stylesheet"' in tag:
            match = re.search(r'href="(/[^"]+)"', tag)
            if match:
                urls.append(match.group(1))
    return urls


def js_import_targets(js):
    """Absolute module specifiers a JS module imports."""
    return re.findall(r'from\s+["\'](/[^"\']+)["\']', js) + re.findall(
        r'^\s*import\s+["\'](/[^"\']+)["\']', js, re.M)


def loaded_static_urls(page_texts, read_module_text):
    """Every static URL the pages load, following ES module imports.

    read_module_text(url) returns the text of a /static/*.js file; it is
    called for each module the pages or their modules import.
    """
    assets = []
    todo = []
    for text in page_texts:
        for url in page_asset_urls(text):
            assets.append(url)
            if url.endswith(".js"):
                todo.append(url)
        # the pages also load scripts as ES modules from inline <script>
        for target in js_import_targets(text):
            assets.append(target)
            if target.endswith(".js"):
                todo.append(target)
    seen = set()
    while todo:
        url = todo.pop()
        if url in seen:
            continue
        seen.add(url)
        for target in js_import_targets(read_module_text(url)):
            assets.append(target)
            if target.endswith(".js"):
                todo.append(target)
    return assets
