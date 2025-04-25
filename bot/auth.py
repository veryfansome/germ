from itsdangerous import TimestampSigner, BadSignature, SignatureExpired

from settings.germ_settings import COOKIE_SIGNING_SECRET

USER_COOKIE = "germ_user"
MAX_COOKIE_AGE: int = 60 * 60 * 24 * 7   # 7 days

ts_signer = TimestampSigner(COOKIE_SIGNING_SECRET, salt="germ-user-cookie")


def get_decoded_cookie(signed_cookie: str) -> str | None:
    if not signed_cookie:
        return None
    try:
        encoded_cookie = ts_signer.unsign(signed_cookie, max_age=MAX_COOKIE_AGE)
        return encoded_cookie.decode()
    except (BadSignature, SignatureExpired):
        return None


def sign_cookie(cookie: str) -> str:
    return ts_signer.sign(cookie.encode()).decode()
