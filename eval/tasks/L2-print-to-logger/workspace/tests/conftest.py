"""让 tests 能 import workspace 根下的 app 包."""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
