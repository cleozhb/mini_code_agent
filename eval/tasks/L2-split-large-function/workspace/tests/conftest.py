"""让 tests 能 import 工作区根目录下的 order_processor."""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
