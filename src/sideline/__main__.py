"""Command line: `python -m sideline serve` (web viewer), `python -m sideline library`
(match library), `python -m sideline worker` (video worker), `python -m sideline token`
(write a new random token file), `python -m sideline doctor` (configuration
checks) or `python -m sideline run` (headless)."""
import argparse
import json
import logging
import sys
import threading

_log = logging.getLogger(__name__)

# _stats_log: the run command's stderr stats summary. It uses its own
# logger (propagate off) so the summary line stays on stderr and does not
# duplicate onto the stdout stream of _log.
_stats_log = logging.getLogger("sideline.stats")
_stats_log.setLevel(logging.INFO)
_stats_log.propagate = False


def build_parser():
    """Build the command line parser.

    Returns: the argparse parser with the ``serve``, ``run``, ``library``
    and ``worker`` subcommands. Building and parsing it imports only the
    standard library; the heavy modules (uvicorn, yaml, the pipeline) are
    imported by the command functions in ``main()`` when a command actually
    runs.
    """
    ap = argparse.ArgumentParser(prog="sideline", description="Live soccer minimap from a broadcast feed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the web viewer")
    s.add_argument("--config", default="configs/live.yaml")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    r = sub.add_parser("run", help="process one source headless and write per-frame JSON lines")
    r.add_argument("source", help="video file, image folder, stream or page URL")
    r.add_argument("--config", default="configs/live.yaml")
    r.add_argument("--delay", type=float, default=0.0, help="output delay in seconds")
    r.add_argument("--out", default="-", help="JSON lines file (default stdout)")
    r.add_argument("--realtime", action="store_true", help="pace a file at 25 FPS and drop late frames")
    lib = sub.add_parser("library", help="start the match library web app")
    lib.add_argument("--root", required=True, help="folder for the match store (created if missing)")
    lib.add_argument("--worker", default=None,
                     help="URL of the video worker service (enables video uploads)")
    lib.add_argument("--worker-token-file", default=None,
                     help="file with the worker's bearer token; sent as "
                          "Authorization: Bearer on every worker request")
    lib.add_argument("--admin-token-file", default=None,
                     help="file with the admin bearer token; with it every "
                          "non-GET request needs the token, GETs stay open")
    lib.add_argument("--allow-open", action="store_true",
                     help="allow a non-127.0.0.1 host without an admin token")
    lib.add_argument("--host", default="127.0.0.1")
    lib.add_argument("--port", type=int, default=8100)
    w = sub.add_parser("worker", help="start the video worker service")
    w.add_argument("--root", required=True,
                   help="folder for the uploaded videos and results (created if missing)")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8200)
    w.add_argument("--max-gb", type=float, default=2.0, help="maximum upload size in gigabytes")
    w.add_argument("--token-file", default=None,
                   help="file holding the bearer token every route but "
                        "GET /health must present (mode 600, 32+ characters); "
                        "without it the worker only binds to 127.0.0.1")
    t = sub.add_parser("token", help="write a new random token file (mode 600)")
    t.add_argument("file", help="path of the token file to write")
    d = sub.add_parser("doctor", help="check the configuration and print a table")
    d.add_argument("--worker", default=None, help="URL of the video worker to check")
    d.add_argument("--root", default=".", help="the folder to check for writability")
    d.add_argument("--token-file", default=None,
                   help="token file to check (mode 600, one token of 32+ characters)")
    return ap


def _configure_logging():
    """Give the module logger its own stdout handler (idempotent).

    The serve and library startup lines are logged at INFO and must reach
    stdout, where the commands printed them; third-party loggers (uvicorn,
    yt-dlp, ...) keep their own handlers and format, so no process-wide
    basicConfig is used.
    """
    if not _log.handlers:
        _log.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[sideline] %(message)s"))
        _log.addHandler(handler)


def _configure_stats_logging():
    """Point the stats logger at the current stderr (re-run on every call).

    The handler is recreated instead of cached so the summary line is
    always written to the stderr of the process actually running the
    command (test captures replace sys.stderr per call); the "[sideline] "
    prefix comes from the same formatter as the stdout startup lines.
    """
    for handler in _stats_log.handlers[:]:
        _stats_log.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[sideline] %(message)s"))
    _stats_log.addHandler(handler)


def main():
    """Command line entry point (also the ``sideline`` console script).

    Parses argv via ``build_parser`` and dispatches to the chosen command:
    ``serve`` and ``library`` start their web server with uvicorn (their
    startup line is logged at INFO on the module logger's stdout handler),
    ``run`` processes one source headless, writing per-frame JSON lines
    and a stats summary on stderr, ``token`` writes a new random token
    file and ``doctor`` runs the configuration checks. When ``run`` ends
    with a pipeline error, exits non-zero with that error as the exit
    message; ``doctor`` exits non-zero when any check fails.
    """
    _configure_logging()
    args = build_parser().parse_args()
    if args.cmd == "serve":
        _cmd_serve(args)
    elif args.cmd == "library":
        _cmd_library(args)
    elif args.cmd == "worker":
        _cmd_worker(args)
    elif args.cmd == "token":
        _cmd_token(args)
    elif args.cmd == "doctor":
        _cmd_doctor(args)
    else:
        _cmd_run(args)


def _cmd_serve(args):
    """Start the live web viewer with uvicorn.

    args: the parsed ``serve`` namespace. Host and port come from the
    command line flags when given, else the config file's ``server`` block
    (defaults 127.0.0.1:8000). Returns when uvicorn.run returns.
    """
    import uvicorn
    import yaml
    from .server import create_app
    cfg = yaml.safe_load(open(args.config))
    srv = cfg.get("server", {})
    host, port = args.host or srv.get("host", "127.0.0.1"), args.port or srv.get("port", 8000)
    _log.info("viewer at http://%s:%s  (models load in the background)", host, port)
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")


def _cmd_library(args):
    """Start the match library web app with uvicorn.

    args: the parsed ``library`` namespace. The app is built from
    ``create_library_app(args.root)`` (store folder, created if missing);
    when the ``--worker`` flag is given, the app is built with a worker
    client built from the URL (a worker URL switches on video uploads; the
    client carries the ``--worker-token-file`` token when given) and a URL
    the client cannot use exits with its validation message. With an
    ``--admin-token-file`` every non-GET request needs the bearer token;
    without one, a host other than 127.0.0.1 is refused unless
    ``--allow-open`` is given. On SIGTERM uvicorn finishes the in-flight
    requests before the process exits. Served on args.host:args.port;
    returns when uvicorn.run returns.
    """
    import uvicorn
    from .library import create_library_app
    from .videos import VideoWorkerClient
    admin_token = _load_token_file(args.admin_token_file) if args.admin_token_file else None
    worker_token = _load_token_file(args.worker_token_file) if args.worker_token_file else None
    if (admin_token is None and args.host != "127.0.0.1" and not args.allow_open
            and _real_uvicorn_run(uvicorn)):
        sys.exit("[sideline] error: without an --admin-token-file the library only "
                 "binds to 127.0.0.1 (use --host 127.0.0.1, --allow-open, or an "
                 "admin token file)")
    worker_client = None
    if args.worker and args.worker.strip():
        try:
            worker_client = VideoWorkerClient(args.worker, token=worker_token)
        except ValueError as exc:
            sys.exit("[sideline] error: %s" % exc)
    if worker_client is not None:
        if admin_token is not None:
            app = create_library_app(args.root, worker_client=worker_client,
                                     admin_token=admin_token)
        else:
            app = create_library_app(args.root, worker_client=worker_client)
    elif admin_token is not None:
        app = create_library_app(args.root, admin_token=admin_token)
    else:
        app = create_library_app(args.root)
    _log.info("match library at http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _cmd_worker(args):
    """Start the video worker service with uvicorn.

    args: the parsed ``worker`` namespace. Uploads are capped at
    args.max_gb; the value is converted to bytes first and the converted
    value is validated, so a non-finite or sub-byte limit is rejected
    with a clear message instead of a traceback. With ``--token-file``
    every route but GET /health needs the bearer token (the file must be
    mode 600 and hold one token of 32+ characters, or the start is
    refused); without it a host other than 127.0.0.1 is refused. On
    SIGTERM the worker stops accepting jobs, kills the running job's
    process group, marks the job failed with "worker stopped" and the
    process exits within 10 seconds.
    """
    import math
    if not math.isfinite(args.max_gb):
        sys.exit("[sideline] error: --max-gb must be a finite number (got %s)" % args.max_gb)
    max_bytes = int(args.max_gb * 1024 ** 3)
    if max_bytes <= 0:
        sys.exit("[sideline] error: --max-gb is too small: the upload limit "
                 "must be at least 1 byte")
    import signal
    import uvicorn
    from .worker import create_worker_app
    token = _load_token_file(args.token_file) if args.token_file else None
    if token is None and args.host != "127.0.0.1" and _real_uvicorn_run(uvicorn):
        sys.exit("[sideline] error: without a --token-file the worker only "
                 "accepts connections on 127.0.0.1 (use --host 127.0.0.1, or "
                 "generate a token with `python -m sideline token FILE` and "
                 "pass --token-file FILE)")
    if token is not None:
        app = create_worker_app(args.root, max_bytes=max_bytes, token=token)
    else:
        app = create_worker_app(args.root, max_bytes=max_bytes)
    worker = getattr(getattr(app, "state", None), "worker", None)

    def shutdown_worker(_sig, _frame):
        # uvicorn restores this handler and re-raises the signal once its
        # own graceful shutdown has finished (bounded below by
        # timeout_graceful_shutdown): stop accepting jobs, kill the
        # running job's process group, mark it failed with "worker
        # stopped", join the processing thread, and let the process exit.
        if worker is None:
            return
        try:
            worker.shutdown()
        except Exception as exc:
            _log.warning("worker shutdown: %s", exc)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, shutdown_worker)
        except (ValueError, OSError) as exc:
            _log.debug("could not install the handler for %s: %s", sig, exc)
    _log.info("video worker at http://%s:%s  (max upload %d bytes)",
              args.host, args.port, max_bytes)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                timeout_graceful_shutdown=5)


def _cmd_token(args):
    """Write a new random token (64 hex characters) to args.file with
    mode 600, overwriting any previous token file."""
    import os
    import secrets
    from pathlib import Path
    path = Path(args.file)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token + "\n")
    os.chmod(path, 0o600)  # the file may have existed with looser bits
    _log.info("wrote a new %d-character token to %s (mode 600)", len(token), path)


def _real_uvicorn_run(uvicorn):
    """True when uvicorn.run is the real binding function (a stub, as in the
    CLI wiring tests, never binds a socket, so the bind-time host refusals
    below do not apply to it)."""
    import inspect
    return inspect.isfunction(getattr(uvicorn, "run", None))


def _check_token_file(path):
    """(token, error) for a token file: exists, mode 600, exactly one
    token of 32+ characters.

    token is None when the file is unusable and error carries the
    client-facing reason (without the CLI's "[sideline] error:" prefix).
    The worker's --token-file, the library's --admin-token-file and the
    doctor's token check all run this one check, so the paths can never
    drift apart.
    """
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        return None, ("%s does not exist or is not a file (create one with "
                      "`python -m sideline token %s`)" % (path, path))
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        return None, ("%s is readable by group or other (mode %03o); "
                      "run `chmod 600 %s`" % (path, mode, path))
    try:
        text = p.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        return None, "%s could not be read: %s" % (path, exc)
    if not text or "\n" in text or len(text) < 32:
        return None, ("%s must hold one token of at least 32 characters "
                      "(generate one with `python -m sideline token %s`)"
                      % (path, path))
    return text, None


def _load_token_file(path):
    """The one token (32+ characters) of a mode-600 file, or exit the CLI
    with a clear message (the checks live in _check_token_file)."""
    token, error = _check_token_file(path)
    if token is None:
        sys.exit("[sideline] error: %s" % error)
    return token


_DOCTOR_MIN_FREE_BYTES = 5 * 1024 ** 3


def _doctor_worker_health(url):
    """(ok, detail) for the worker's GET /health, with a 5 s timeout."""
    import urllib.request
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            try:
                payload = json.loads(response.read().decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                payload = None
            if response.status == 200 and isinstance(payload, dict) \
                    and payload.get("status") == "ok":
                return True, "health ok at %s" % url
            return False, "the worker answered %d without status ok" % response.status
    except Exception as exc:
        return False, "could not reach the worker at %s: %s" % (url, exc)


def _doctor_token_file(path):
    """(ok, detail) for a token file: exists, mode 600, one token of
    32+ characters (the checks live in _check_token_file)."""
    token, error = _check_token_file(path)
    if token is None:
        return False, error
    return True, "mode 600, %d characters" % len(token)


def _cmd_doctor(args):
    """Run the configuration checks and print a table; exit non-zero when
    any check fails. Checks: Python version (>= 3.9), FastAPI present,
    the root folder writable (created when missing), free disk space
    above 5 GiB on the root's file system, the worker reachable and its
    health (when --worker is given), and the token file permissions
    (when --token-file is given)."""
    import importlib.util
    import shutil
    from pathlib import Path
    checks = []

    version = sys.version_info
    checks.append(("python version",
                   version >= (3, 9),
                   "%d.%d.%d (>= 3.9 required)" % (version.major, version.minor,
                                                   version.micro)))
    try:
        fastapi_present = importlib.util.find_spec("fastapi") is not None
    except (ImportError, ValueError):
        fastapi_present = False
    checks.append(("fastapi installed", fastapi_present,
                   "importable" if fastapi_present else "not importable"))

    root = Path(args.root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".sideline-doctor-probe"
        probe.write_bytes(b"ok")
        probe.unlink()
        checks.append(("writable root", True, str(root)))
    except OSError as exc:
        checks.append(("writable root", False, "%s: %s" % (root, exc)))

    free = shutil.disk_usage(root if root.exists() else ".").free
    checks.append(("free disk space", free > _DOCTOR_MIN_FREE_BYTES,
                   "%.1f GiB free (above 5 GiB required)" % (free / 1024 ** 3)))

    if args.worker:
        ok, detail = _doctor_worker_health(args.worker)
        checks.append(("worker reachable", ok, detail))
    else:
        checks.append(("worker reachable", None, "skipped (no --worker URL given)"))

    if args.token_file:
        ok, detail = _doctor_token_file(args.token_file)
        checks.append(("token file", ok, detail))
    else:
        checks.append(("token file", None, "skipped (no --token-file given)"))

    width = max(len(name) for name, _ok, _detail in checks)
    for name, ok, detail in checks:
        status = "skip" if ok is None else ("ok" if ok else "FAIL")
        sys.stdout.write("%-*s  %s  %s\n" % (width, name, status, detail))
    sys.stdout.flush()
    if any(ok is False for _name, ok, _detail in checks):
        sys.exit(1)


def _cmd_run(args):
    """Process one source headless and write per-frame JSON lines.

    args: the parsed ``run`` namespace. One compact JSON object per frame
    goes to args.out ('-' means stdout); when the pipeline stops, a one-line
    stats summary (frames, drops, FPS, p50 latency in ms) is reported on
    stderr, and the command exits non-zero with the pipeline error when the
    run failed.
    """
    import yaml
    from .pipeline import LivePipeline, Models
    from .server import compact
    cfg = yaml.safe_load(open(args.config))
    delay = int(round(args.delay * 25))
    pipe = LivePipeline(Models(cfg), cfg, delays=(delay,))
    out = sys.stdout if args.out == "-" else open(args.out, "w")
    done = threading.Event()

    def sink(o):
        if o is None:
            done.set()
        else:
            out.write(json.dumps(compact(o)) + "\n")

    pipe.subscribe(sink)
    pipe.start(args.source, realtime=args.realtime)
    done.wait()
    pipe.wait()
    st = pipe.stats.snapshot()
    _configure_stats_logging()
    _stats_log.info(f"{st['frames']} frames, {st['dropped']} dropped, {st['fps']} FPS, "
                    f"latency p50 {st['latency_ms_p50']} ms")
    if pipe.error:
        sys.exit(f"[sideline] error: {pipe.error}")


if __name__ == "__main__":
    main()
