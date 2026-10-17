"""Shared fixtures for the integration suite: the recorded sample and the app."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sideline import library  # noqa: E402

SAMPLE = REPO / "tests" / "fixtures" / "sample_match.jsonl"


@pytest.fixture(scope="session")
def sample_bytes():
    return SAMPLE.read_bytes()


@pytest.fixture(scope="session")
def sample_frames():
    return [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.fixture()
def app_root(tmp_path):
    return tmp_path / "matches"


@pytest.fixture()
def client(app_root):
    return TestClient(library.create_library_app(app_root))
