"""Integration tests for the command line entry point (``python -m sideline``).

The other integration tests cover the library pages and API; these run the
three subcommands through ``main()`` with the heavy dependencies (uvicorn,
yaml, the live server, the pipeline) replaced by fakes, and check the
wiring: which app is handed to uvicorn.run and on which host/port, what
the run command writes to its output file and to the stderr stats line,
and its exit code when the pipeline reports an error.
"""
import json
import sys
import types
from unittest import mock

import pytest

from sideline import __main__ as cli
from sideline import library


class _Models:
    """Stand-in for sideline.pipeline.Models (constructor takes the config)."""

    def __init__(self, cfg):
        self.cfg = cfg


class _Pipeline:
    """Stand-in for sideline.pipeline.LivePipeline.

    Emits three frames and a None stop marker from start(), so the
    command's wait() returns immediately. When fail is set on the class,
    start() records a pipeline error, so the command must exit non-zero.
    """

    fail = False
    instances = []

    def __init__(self, models, cfg, delays=()):
        self.models = models
        self.delays = delays
        self.error = None
        self.sink = None
        self.stats = mock.Mock()
        self.stats.snapshot.return_value = {"frames": 3, "dropped": 1, "fps": 25.0,
                                            "latency_ms_p50": 56.0}
        _Pipeline.instances.append(self)

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


def _fake_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


@pytest.fixture()
def argv(monkeypatch):
    """Setter for sys.argv (the program name is filled in for us)."""
    def set_argv(*parts):
        monkeypatch.setattr(sys, "argv", ["sideline", *parts])
    return set_argv


def _fake_run_modules(monkeypatch):
    """Fakes for the heavy modules _cmd_run imports (yaml, server, pipeline)."""
    monkeypatch.setitem(sys.modules, "yaml", _fake_module("yaml", safe_load=lambda _f: {}))
    monkeypatch.setitem(sys.modules, "sideline.server",
                        _fake_module("sideline.server", compact=lambda o: {"frame": o["frame"]}))
    monkeypatch.setitem(sys.modules, "sideline.pipeline",
                        _fake_module("sideline.pipeline", LivePipeline=_Pipeline, Models=_Models))


# ---------------------------------------------------------------- library

def test_library_command_serves_the_app_on_the_flags(monkeypatch, tmp_path, argv, capsys):
    uvicorn = mock.MagicMock(name="uvicorn")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    # Force a fresh stdout handler so the startup line is captured here.
    monkeypatch.setattr(cli._log, "handlers", [])
    root = tmp_path / "store"
    argv("library", "--root", str(root), "--host", "0.0.0.0", "--port", "9123")
    cli.main()
    out, err = capsys.readouterr()
    assert "[sideline] match library at http://0.0.0.0:9123" in out
    assert err == ""
    assert root.is_dir()
    assert isinstance(uvicorn.run.call_args.args[0], type(library.create_library_app(root)))
    assert uvicorn.run.call_args.kwargs == {"host": "0.0.0.0", "port": 9123,
                                            "log_level": "warning"}


# ----------------------------------------------------------------- serve

def test_serve_command_uses_the_config_server_block_then_flags(monkeypatch, tmp_path, argv):
    uvicorn = mock.MagicMock(name="uvicorn")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    state = {"cfg": {"server": {"host": "10.0.0.5", "port": 7100}}}
    monkeypatch.setitem(sys.modules, "yaml",
                        _fake_module("yaml", safe_load=lambda _f: state["cfg"]))
    app = object()
    monkeypatch.setitem(sys.modules, "sideline.server",
                        _fake_module("sideline.server", create_app=lambda cfg: app))
    cfgfile = tmp_path / "live.yaml"
    cfgfile.write_text("")

    argv("serve", "--config", str(cfgfile))
    cli.main()
    state["cfg"] = {}
    argv("serve", "--config", str(cfgfile), "--host", "127.0.0.1", "--port", "7777")
    cli.main()

    assert all(c.args[0] is app for c in uvicorn.run.call_args_list)
    assert uvicorn.run.call_args_list[0].kwargs == {"host": "10.0.0.5", "port": 7100,
                                                    "log_level": "warning"}
    assert uvicorn.run.call_args_list[1].kwargs == {"host": "127.0.0.1", "port": 7777,
                                                    "log_level": "warning"}


# ------------------------------------------------------------------- run

def test_run_command_writes_frames_and_the_stats_line(monkeypatch, tmp_path, argv, capsys):
    _fake_run_modules(monkeypatch)
    cfgfile = tmp_path / "live.yaml"
    cfgfile.write_text("")
    argv("run", "clip.mp4", "--config", str(cfgfile), "--delay", "1", "--realtime")
    cli.main()

    out, err = capsys.readouterr()
    frames = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert [f["frame"] for f in frames] == [0, 1, 2]
    assert "[sideline] 3 frames, 1 dropped, 25.0 FPS, latency p50 56.0 ms" in err
    pipe = _Pipeline.instances[-1]
    assert pipe.source == "clip.mp4" and pipe.realtime is True
    assert pipe.delays == (25,) and isinstance(pipe.models, _Models)


def test_run_command_exits_nonzero_with_the_pipeline_error(monkeypatch, tmp_path, argv):
    _fake_run_modules(monkeypatch)
    cfgfile = tmp_path / "live.yaml"
    cfgfile.write_text("")
    _Pipeline.fail = True
    out_file = tmp_path / "frames.jsonl"
    try:
        argv("run", "clip.mp4", "--config", str(cfgfile), "--out", str(out_file))
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        _Pipeline.fail = False
    assert exc.value.code == "[sideline] error: boom"
    assert _Pipeline.instances[-1].error == "boom"
