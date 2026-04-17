"""让 pytest 能直接 `from user import xxx`。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
