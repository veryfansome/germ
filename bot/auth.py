from itsdangerous import TimestampSigner, BadSignature, SignatureExpired

USER_COOKIE = "germ_user"
MAX_COOKIE_AGE: int = 60 * 60 * 24 * 7   # 7 days
SIGNING_SECRET = "CHANGE_ME"             # 32+ random bytes – env-var in prod

ts_signer = TimestampSigner(SIGNING_SECRET, salt="germ-bot-salt")


def get_user_id(signed_cookie: str) -> str | None:
    if not signed_cookie:
        return None
    try:
        value = ts_signer.unsign(signed_cookie, max_age=MAX_COOKIE_AGE)
        return value.decode()
    except (BadSignature, SignatureExpired):
        return None


def sign_cookie(user_id: str) -> str:
    return ts_signer.sign(user_id.encode()).decode()
