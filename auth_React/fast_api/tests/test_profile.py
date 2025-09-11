# tests/test_profile.py
import io
from datetime import date, timedelta

def auth_header(token: str):
    return {"Authorization": f"Bearer {token}"}

def big_text(words=160):
    return " ".join([f"w{i}" for i in range(words)])

def test_create_profile_success(client, make_user_and_token):
    uname, token = make_user_and_token()

    # fake small PDF file
    fake_pdf = io.BytesIO(b"%PDF-1.4\n%%EOF\n")
    files = {"file": ("resume.pdf", fake_pdf, "application/pdf")}

    today = date.today()
    payload = {
        "firstName":"Sai",
        "lastName":"Mateti",
        "dob": "1995-01-01",
        "gender":"male",
        "phone":"+1 555 111 2222",
        "address":"123 Test Street",
        "pincode":"560001",
        "country":"India",
        "aboutMe": big_text(160),
        "acknowledge": "true"
    }

    r = client.post("/api/profile", headers=auth_header(token), data=payload, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["username"] == uname
    assert body["country"] == "India"
    assert body["firstName"] == "Sai"

def test_about_me_too_short(client, make_user_and_token):
    _, token = make_user_and_token()
    files = {"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4\n%%EOF\n"), "application/pdf")}
    payload = {
        "firstName":"A","lastName":"B","dob":"1990-01-01","gender":"female",
        "phone":"+1 555","address":"Anywhere","pincode":"12345","country":"USA",
        "aboutMe":"too short", "acknowledge":"true"
    }
    r = client.post("/api/profile", headers=auth_header(token), data=payload, files=files)
    assert r.status_code == 422
    assert "least 150 words" in r.json()["detail"]
