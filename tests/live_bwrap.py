"""When bwrap cannot start: skip locally, fail on GitHub Actions.

CI loads the production AppArmor bwrap profile (and a sysctl fallback) so
live sandbox tests execute. A skip there would hide a production regression.
"""

from __future__ import annotations

import os

import pytest


def require_live_bwrap(reason: str) -> None:
    if os.environ.get("GITHUB_ACTIONS"):
        pytest.fail(reason)
    pytest.skip(reason)
