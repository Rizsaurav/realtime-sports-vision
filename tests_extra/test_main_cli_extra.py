"""Extra tests for the command line entry point (src/sideline/__main__.py).

The acceptance suite only exercises build_parser, so main() was never
executed. These tests run all three subcommands through cli.main() with the
heavy dependencies (uvicorn, yaml, the GPU pipeline, the live server)
replaced by fakes, and verify the dispatch and each command's wiring:
which app object is handed to uvicorn.run, on which host/port, which
pipeline the run command constructs, and what it writes where.
"""

import json
import os
import sys
import types
from unittest import mock

import pytest

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

from sideline import __main__ as cli  # noqa: E402
from sideline import library  # noqa: E402


# ---------------------------------------------------------------- fakes

class FakeModels:
    """Stand-in for sideline.pipeline.Models (constructor takes the config)."""

    def __init__(self, cfg):
        self.cfg = cfg


class FakePipeline:
    """Stand-in for sideline.pipeline.LivePipeline.

    Emits three frames and a None stop marker through the sink from
    start(), so the command's wait() returns immediately. When fail is
    true, start() records a pipeline error so the command must exit.
    """

    fail = False
    instances = []

    def __init__(self, models, cfg, delays=()):
        self.models = models
        self.cfg = cfg
        self.delays = delays
        self.error = None
        self.sink = None
        self.stats = mock.Mock()
        self.stats.snapshot.return_value = {
            "frames": 3, "dropped": 1, "fps": 25.0, "latency_ms_p50": 56.0}
        FakePipeline.instances.append(self)

    def subscribe(self, sink):
        self.sink = sink

    def start(self, source, realtime=False):
        self.source = source
        self.realtime = realtime
        self.error = "boom" if self.fail else None
        for i in range(3):
            self.sink({"frame": i})
        self.sink(None)

    def wait(self):
        pass


@pytest.fixture()
def argv(monkeypatch):
    """Setter for sys.argv (the program name is filled in for us)."""
    def set_argv(*parts):
        monkeypatch.setattr(sys, "argv", ["sideline", *parts])
    return set_argv


def make_uvicorn(monkeypatch):
    uvicorn = mock.MagicMock(name="uvicorn")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    return uvicorn


def make_yaml(monkeypatch):
    """A stand-in yaml module; safe_load returns state["cfg"] at call time."""
    state = {"cfg": {}}
    mod = types.ModuleType("yaml")
    mod.safe_load = lambda _f: state["cfg"]
    monkeypatch.setitem(sys.modules, "yaml", mod)
    return state


def make_server_module(monkeypatch, compact=None, create_app=None):
    mod = types.ModuleType("sideline.server")
    mod.compact = compact if compact is not None else (lambda o: o)
    mod.create_app = create_app if create_app is not None else (lambda cfg: object())
    monkeypatch.setitem(sys.modules, "sideline.server", mod)
    return mod


def make_pipeline_module(monkeypatch):
    mod = types.ModuleType("sideline.pipeline")
    mod.LivePipeline = FakePipeline
    mod.Models = FakeModels
    monkeypatch.setitem(sys.modules, "sideline.pipeline", mod)
    return mod


def write_config(tmp_path):
    cfg = tmp_path / "live.yaml"
    cfg.write_text("")
    return cfg


# ---------------------------------------------------------------- main()

def test_main_library_starts_uvicorn_with_the_built_app(monkeypatch, tmp_path, argv, caplog):
    uvicorn = make_uvicorn(monkeypatch)
    built = {}
    real = library.create_library_app

    def spy(root):
        built["root"] = root
        built["app"] = real(root)
        return built["app"]

    monkeypatch.setattr(library, "create_library_app", spy)
    root = tmp_path / "store"
    argv("library", "--root", str(root))
    with caplog.at_level("INFO"):
        cli.main()

    assert built["root"] == str(root) and root.is_dir()
    assert uvicorn.run.call_args.args[0] is built["app"]
    assert uvicorn.run.call_args.kwargs == {"host": "127.0.0.1", "port": 8100, "log_level": "warning"}
    assert "match library at http://127.0.0.1:8100" in caplog.text


def test_main_library_honours_host_and_port_flags(monkeypatch, tmp_path, argv):
    uvicorn = make_uvicorn(monkeypatch)
    monkeypatch.setattr(library, "create_library_app", lambda root: object())
    argv("library", "--root", str(tmp_path / "s"), "--host", "0.0.0.0", "--port", "9123")
    cli.main()
    assert uvicorn.run.call_args.kwargs == {"host": "0.0.0.0", "port": 9123, "log_level": "warning"}


def test_main_serve_builds_the_viewer_app_and_uses_config_and_flags(monkeypatch, tmp_path, argv, caplog):
    uvicorn = make_uvicorn(monkeypatch)
    cfg_state = make_yaml(monkeypatch)
    app = object()
    seen_cfgs = []
    make_server_module(monkeypatch, create_app=lambda c: seen_cfgs.append(c) or app)
    cfgfile = write_config(tmp_path)

    with_server = {"server": {"host": "10.0.0.5", "port": 7100}}
    cfg_state["cfg"] = with_server
    argv("serve", "--config", str(cfgfile))
    with caplog.at_level("INFO"):
        cli.main()

    cfg_state["cfg"] = with_server
    argv("serve", "--config", str(cfgfile), "--host", "0.0.0.0", "--port", "7777")
    cli.main()

    cfg_state["cfg"] = {}
    argv("serve", "--config", str(cfgfile))
    cli.main()

    assert [c.args[0] for c in uvicorn.run.call_args_list] == [app] * 3
    assert uvicorn.run.call_args_list[0].kwargs == {"host": "10.0.0.5", "port": 7100, "log_level": "warning"}
    assert uvicorn.run.call_args_list[1].kwargs == {"host": "0.0.0.0", "port": 7777, "log_level": "warning"}
    assert uvicorn.run.call_args_list[2].kwargs == {"host": "127.0.0.1", "port": 8000, "log_level": "warning"}
    assert seen_cfgs == [with_server, with_server, cfg_state["cfg"]]
    assert "viewer at http://10.0.0.5:7100" in caplog.text


def test_main_run_writes_compact_json_lines_to_stdout(monkeypatch, tmp_path, argv, capsys):
    make_yaml(monkeypatch)
    make_server_module(monkeypatch, compact=lambda o: {"f": o["frame"]})
    make_pipeline_module(monkeypatch)
    cfgfile = write_config(tmp_path)
    argv("run", "clip.mp4", "--config", str(cfgfile), "--delay", "1", "--realtime")
    cli.main()

    out = [line for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert [json.loads(line)["f"] for line in out] == [0, 1, 2]
    p = FakePipeline.instances[-1]
    assert p.source == "clip.mp4" and p.realtime is True
    assert p.delays == (25,) and isinstance(p.models, FakeModels)


def test_main_run_exits_nonzero_with_the_pipeline_error(monkeypatch, tmp_path, argv):
    make_yaml(monkeypatch)
    make_server_module(monkeypatch)
    make_pipeline_module(monkeypatch)
    cfgfile = write_config(tmp_path)
    out_file = tmp_path / "frames.jsonl"
    FakePipeline.fail = True
    try:
        argv("run", "clip.mp4", "--config", str(cfgfile), "--out", str(out_file))
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        FakePipeline.fail = False
    assert exc.value.code == "[sideline] error: boom"
    assert FakePipeline.instances[-1].error == "boom"


# ------------------------------------------------------------ log streams

def test_main_serve_startup_line_goes_to_stdout_not_stderr(monkeypatch, tmp_path, argv, capsys):
    uvicorn = make_uvicorn(monkeypatch)
    make_yaml(monkeypatch)
    make_server_module(monkeypatch)
    cfgfile = write_config(tmp_path)
    # Force a fresh handler so the line is written to the captured stdout.
    monkeypatch.setattr(cli._log, "handlers", [])
    argv("serve", "--config", str(cfgfile))
    cli.main()
    out, err = capsys.readouterr()
    assert "[sideline] viewer at http://127.0.0.1:8000  (models load in the background)" in out
    assert err == ""


def test_main_library_startup_line_goes_to_stdout_not_stderr(monkeypatch, tmp_path, argv, capsys):
    uvicorn = make_uvicorn(monkeypatch)
    monkeypatch.setattr(library, "create_library_app", lambda root: object())
    monkeypatch.setattr(cli._log, "handlers", [])
    argv("library", "--root", str(tmp_path / "s"))
    cli.main()
    out, err = capsys.readouterr()
    assert "[sideline] match library at http://127.0.0.1:8100" in out
    assert err == ""


def test_main_run_stats_line_goes_to_stderr(monkeypatch, tmp_path, argv, capsys):
    make_yaml(monkeypatch)
    make_server_module(monkeypatch)
    make_pipeline_module(monkeypatch)
    cfgfile = write_config(tmp_path)
    out_file = tmp_path / "frames.jsonl"
    argv("run", "clip.mp4", "--config", str(cfgfile), "--out", str(out_file))
    cli.main()
    out, err = capsys.readouterr()
    assert "[sideline] 3 frames, 1 dropped, 25.0 FPS, latency p50 56.0 ms" in err
    assert out == ""
