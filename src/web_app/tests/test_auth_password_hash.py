from src.web_app.core.security import hash_password, verify_password


def test_password_hash_and_verify():
    hashed = hash_password("secret-pass")
    assert hashed != "secret-pass"
    assert verify_password("secret-pass", hashed)
