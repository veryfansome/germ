from itsdangerous import TimestampSigner, BadSignature, SignatureExpired
from sqlalchemy import MetaData, Table, insert, select
import bcrypt
import logging

from bot.db.models import AsyncSessionLocal, engine
from settings.germ_settings import COOKIE_SIGNING_SECRET

logger = logging.getLogger(__name__)

USER_COOKIE = "germ_user"
MAX_COOKIE_AGE: int = 60 * 60 * 24 * 7   # 7 days

chat_user_table = Table('chat_user', MetaData(), autoload_with=engine)
ts_signer = TimestampSigner(COOKIE_SIGNING_SECRET, salt="germ-user-cookie")


async def add_new_user(username: str, plain_text_password: str) -> bool:
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            try:
                insert_stmt = insert(chat_user_table).values(
                    user_name=username,
                    password_hash=hash_password(plain_text_password),
                )
                await rdb_session.execute(insert_stmt)
                await rdb_session.commit()
                logger.info(f"New user {username} inserted successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to insert new user {username}: {e}")
                await rdb_session.rollback()
                return False


def get_decoded_cookie(signed_cookie: str) -> str | None:
    if not signed_cookie:
        return None
    try:
        encoded_cookie = ts_signer.unsign(signed_cookie, max_age=MAX_COOKIE_AGE)
        return encoded_cookie.decode()
    except (BadSignature, SignatureExpired):
        return None


async def get_user_password_hash(username: str):
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            chat_user_stmt = select(chat_user_table.c.password_hash).where(
                chat_user_table.c.user_name == username
            )
            password_hash_record = (await rdb_session.execute(chat_user_stmt)).one_or_none()
            if password_hash_record is not None:
                return password_hash_record[0]
            return password_hash_record


def hash_password(plain_text_password: str) -> str:
    return bcrypt.hashpw(
        plain_text_password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')


def sign_cookie(cookie: str) -> str:
    return ts_signer.sign(cookie.encode()).decode()


def verify_password(plain_text_password: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(
        plain_text_password.encode('utf-8'),
        stored_hash.encode('utf-8')
    )
