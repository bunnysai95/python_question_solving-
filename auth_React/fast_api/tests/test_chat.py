# tests/test_chat.py
def auth_header(token: str):
    return {"Authorization": f"Bearer {token}"}

def test_chat_echo(client, make_user_and_token):
    _, token = make_user_and_token()
    r = client.post("/api/chat",
        headers={**auth_header(token), "Content-Type":"application/json"},
        json={"messages":[{"role":"user","content":"hello test"}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["reply"].lower().startswith("echo:")
