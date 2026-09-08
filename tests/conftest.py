import pytest

from orbweaver.auth import mint_token
from orbweaver.ratelimit import reset_rate_limiter_for_tests


@pytest.fixture
def auth_header() -> dict[str, str]:
    return {"authorization": f"Bearer {mint_token('t')}"}


@pytest.fixture(autouse=True)
def _isolated_rate_limiter(tmp_path_factory):
    data = tmp_path_factory.mktemp("rate-limit")
    reset_rate_limiter_for_tests(data)
    yield
    reset_rate_limiter_for_tests(data)
