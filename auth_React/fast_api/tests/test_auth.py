# tests/test_auth.py
def auth_header(token: str):
    return {"Authorization": f"Bearer {token}"}

def test_register_login_me(client, make_user_and_token):
    uname, token = make_user_and_token()

    # /api/me with valid token
    r = client.get("/api/me", headers=auth_header(token))
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == uname

def test_register_duplicate_username(client, make_user_and_token):
    uname, _ = make_user_and_token()
    # try to register same username again with a VALID payload
    r = client.post("/api/register", json={
        "firstName": "Test",            # <- was "T" (too short)
        "lastName": "User",
        "dob": "1995-01-01",
        "username": uname,              # same username -> should hit 409 now
        "password": "Abcd!234",
        "confirmPassword": "Abcd!234",
        "phone": "+1 555 000 0000",
    })
    assert r.status_code == 409
    assert r.json()["detail"] == "Username already taken"

def test_login_invalid_password(client, make_user_and_token):
    uname, _ = make_user_and_token()
    r = client.post("/api/login", json={"username":uname, "password":"Wrong!234"})
    assert r.status_code == 401
    assert "Invalid username or password" in r.json()["detail"]

def test_me_requires_token(client):
    r = client.get("/api/me")
    assert r.status_code in (401, 403)
