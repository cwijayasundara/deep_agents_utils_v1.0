"""Tests always run against the live source tree."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
