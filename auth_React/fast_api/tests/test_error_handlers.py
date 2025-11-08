import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_validation_error_handler():
    # call register with invalid firstName (too short) to trigger validation error
    payload = {
        "firstName": "A",
        "lastName": "User",
        "dob": "1990-01-01",
        "username": "testuser_validation",
        "password": "Testpass123",
        "confirmPassword": "Testpass123",
    }
    r = client.post("/api/register", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_generic_exception_handler():
    r = client.get("/__test/raise")
    assert r.status_code == 500
    # generic handler returns generic message
    assert r.json().get("detail") == "Internal server error"
