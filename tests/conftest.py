import pytest

from orbweaver.auth import mint_token
from orbweaver.config import settings
from orbweaver.ratelimit import reset_rate_limiter_for_tests
from orbweaver.turns import reset_for_tests as reset_turns_for_tests


@pytest.fixture
def auth_header() -> dict[str, str]:
    return {"authorization": f"Bearer {mint_token('t')}"}


@pytest.fixture(autouse=True)
def _dev_insecure_secret(monkeypatch):
    """Tests sign with the default secret; opt into dev mode so startup passes.

    test_auth_boundary flips this back off to prove the gateway refuses to start.
    """
    monkeypatch.setattr(settings, "orbweaver_dev_insecure", True)


@pytest.fixture(autouse=True)
def _isolated_rate_limiter(tmp_path_factory):
    data = tmp_path_factory.mktemp("rate-limit")
    reset_rate_limiter_for_tests(data)
    yield
    reset_rate_limiter_for_tests(data)


@pytest.fixture(autouse=True)
def _isolated_turn_registry():
    reset_turns_for_tests()
    yield
    reset_turns_for_tests()
