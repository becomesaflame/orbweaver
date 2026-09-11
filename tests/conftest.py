import pytest

from orbweaver.auth import mint_token
from orbweaver.ratelimit import reset_rate_limiter_for_tests


@pytest.fixture
def auth_header() -> dict[str, str]:
    return {"authorization": f"Bearer {mint_token('t')}"}


@pytest.fixture(autouse=True)
def _no_workspace_checkpoints(monkeypatch):
    """Turns in tests must not write refs/orbweaver/checkpoints into this checkout.

    Tests that exercise checkpoints set settings.orbweaver_checkpoints back to True.
    """
    from orbweaver.config import settings

    monkeypatch.setattr(settings, "orbweaver_checkpoints", False)


@pytest.fixture(autouse=True)
def _isolated_rate_limiter(tmp_path_factory):
    data = tmp_path_factory.mktemp("rate-limit")
    reset_rate_limiter_for_tests(data)
    yield
    reset_rate_limiter_for_tests(data)
