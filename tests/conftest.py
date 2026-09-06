import pytest

from orbweaver.auth import mint_token


@pytest.fixture
def auth_header() -> dict[str, str]:
    return {"authorization": f"Bearer {mint_token('t')}"}
