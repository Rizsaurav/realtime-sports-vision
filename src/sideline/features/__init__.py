"""Plug-in features for the match library app.

Every module in this package whose name does not start with ``_`` is a
feature. A feature defines ``register(router, ctx)`` where ``router`` is
an APIRouter whose routes are mounted under ``/api/matches/{match_id}``
(a feature declares paths like ``@router.get("/possession")``) and
``ctx`` is a FeatureContext; it may also define ``register_app(app,
ctx)`` for routes that are not under a single match.

``register_all(app, ctx)`` imports every feature module and registers
it: a feature that raises on import or while registering is skipped (its
partial routes are rolled back), the error is logged as one JSON line
and the other features still load. A SystemExit raised by a feature
(sys.exit() in the module or in register) counts as such a failure too;
a KeyboardInterrupt is not a feature failure and still stops the
process. It also adds ``GET /api/features``,
which reports every feature sorted by name.

Standard library plus FastAPI only: importing this package never pulls
in the heavy pipeline dependencies (torch, cv2, numpy).
"""

import importlib
import json
import logging
import pkgutil

from fastapi import APIRouter

from ._common import FeatureContext

__all__ = ["FeatureContext", "register_all"]

_log = logging.getLogger(__name__)

# Where the match-scoped feature routes land in the app (the prefix is
# already applied to the routes' paths once they are added to the router).
_MATCH_PREFIX = "/api/matches/{match_id}"


def _failed(entry, name, exc):
    """Mark the feature as failed: record the error and log one JSON line."""
    message = str(exc)
    entry["error"] = ("%s: %s" % (type(exc).__name__, message) if message
                      else type(exc).__name__)
    _log.error(json.dumps({"event": "feature_load_failed", "feature": name,
                           "error": entry["error"]}))
    return entry


def _register_one(app, match_router, ctx, name):
    """Import and register one feature module; always returns its entry.

    The entry is {"name", "ok", "error", "routes"}; routes are the full
    app paths the feature added (match-scoped routes first, then the
    app-level ones), in registration order. A feature that raises on
    import or while registering (register or register_app) is skipped:
    any routes it added in the meantime are rolled back, so a broken
    feature does not leave partial endpoints behind.
    """
    entry = {"name": name, "ok": False, "error": None, "routes": []}
    try:
        module = importlib.import_module("%s.%s" % (__name__, name))
    except (Exception, SystemExit) as exc:  # sys.exit() in a module is a feature failure
        return _failed(entry, name, exc)
    match_at = len(match_router.routes)
    app_at = len(app.routes)
    try:
        module.register(match_router, ctx)
        register_app = getattr(module, "register_app", None)
        if callable(register_app):
            register_app(app, ctx)
    except (Exception, SystemExit) as exc:  # sys.exit() in register is a feature failure
        del match_router.routes[match_at:]
        del app.routes[app_at:]
        return _failed(entry, name, exc)
    entry["routes"] = ([route.path for route in match_router.routes[match_at:]]
                       + [route.path for route in app.routes[app_at:]])
    entry["ok"] = True
    return entry


def register_all(app, ctx):
    """Register every feature module of this package on the app.

    Called once by ``create_library_app``. Adds the features' routes
    (their match-scoped router is mounted under
    ``/api/matches/{match_id}``) and the ``GET /api/features`` endpoint,
    which reports every feature sorted by name.
    """
    match_router = APIRouter(prefix=_MATCH_PREFIX)
    results = []
    for info in pkgutil.iter_modules(__path__):
        name = info.name
        if name.startswith("_"):
            continue
        results.append(_register_one(app, match_router, ctx, name))
    results.sort(key=lambda entry: entry["name"])
    app.include_router(match_router)

    @app.get("/api/features", include_in_schema=False)
    def features_index():
        return {"features": results}

    return results
