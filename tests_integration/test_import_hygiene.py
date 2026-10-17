"""Import hygiene: the lightweight library import stays lightweight (task item 5)."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_importing_sideline_library_pulls_in_no_heavy_modules():
    """Catches a torch, cv2, numpy or pipeline import leaking into sideline.library."""
    code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "import sideline.library\n"
        "bad = [m for m in ('torch', 'cv2', 'numpy', 'sideline.pipeline') if m in sys.modules]\n"
        "print(','.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    ) % str(SRC)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
