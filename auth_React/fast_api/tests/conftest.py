# tests/conftest.py
import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# --- Ensure Python can import fast_api/main.py no matter where pytest is run ---
FAST_API_DIR = Path(__file__).resolve().parents[1]   # .../auth_React/fast_api
if str(FAST_API_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_API_DIR))


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    """
    Build the FastAPI app once per test session, using an isolated SQLite file DB
    and forcing the chat provider to 'echo' (no external calls).
    """
    # Temp SQLite file for this test session (doesn't touch your dev DB)
    dbfile = tmp_path_factory.mktemp("data") / "test.db"
    os.environ["DB_URL"] = f"sqlite://{dbfile}"
    os.environ["CHAT_PROVIDER"] = "echo"  # keep tests fast & deterministic

    # Import AFTER env vars are set so settings pick them up
    if "main" in sys.modules:
        del sys.modules["main"]
    from main import app as fastapi_app

    return fastapi_app


@pytest.fixture()
def client(app):
    """Starlette TestClient around the FastAPI app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def make_user_and_token(client):
    """
    Helper to create a fresh user via the API and return (username, token).
    Uses values that satisfy your Pydantic validators.
    """
    def _make(password: str = "Abcd!234"):
        uname = f"u_{uuid.uuid4().hex[:8]}"

        # ✅ firstName / lastName >= 2 chars to satisfy your schema
        register_payload = {
            "firstName": "Test",
            "lastName": "User",
            "dob": "1995-01-01",
            "username": uname,
            "password": password,
            "confirmPassword": password,
            "phone": "+1 555 000 0000",
        }
        r = client.post("/api/register", json=register_payload)
        assert r.status_code == 201, r.text

        r = client.post("/api/login", json={"username": uname, "password": password})
        assert r.status_code == 200, r.text
        token = r.json()["access_token"]

        return uname, token

    return _make
