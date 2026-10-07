"""Make the project's src/ layout importable for tests.

Scripts in this repo use `sys.path.insert(0, <project>/src)` and import
top-level packages (tracking, benchmark, data, ...); tests follow the
same convention so they exercise the exact code the pipeline runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
