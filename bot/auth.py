from itsdangerous import TimestampSigner, BadSignature, SignatureExpired
from sqlalchemy import MetaData, Table, insert, select
import bcrypt
import json
import logging

from bot.api.models import CookieData
from bot.db.models import AsyncSessionLocal, engine
from settings.germ_settings import COOKIE_SIGNING_SECRET

logger = logging.getLogger(__name__)

USER_COOKIE = "germ_user"
MAX_COOKIE_AGE: int = 60 * 60 * 24 * 7   # 7 days

chat_user_table = Table('chat_user', MetaData(), autoload_with=engine)
ts_signer = TimestampSigner(COOKIE_SIGNING_SECRET, salt="germ-user-cookie")


async def add_new_user(username: str, plain_text_password: str) -> int | None:
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            try:
                insert_stmt = insert(chat_user_table).values(
                    user_name=username,
                    password_hash=hash_password(plain_text_password),
                ).returning(chat_user_table.c.user_id)
                result = await rdb_session.execute(insert_stmt)
                await rdb_session.commit()
                user_id = result.scalar()
                logger.info(f"New user '{username}' inserted successfully with user_id {user_id}")
                return user_id
            except Exception as e:
                logger.error(f"Failed to insert new user {username}: {e}")
                await rdb_session.rollback()
                return None


async def get_single_chat_user_attr_by_username(username: str, attr):
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            chat_user_stmt = select(attr).where(
                chat_user_table.c.user_name == username
            )
            record = (await rdb_session.execute(chat_user_stmt)).one_or_none()
            if record is not None:
                return record[0]


def get_decoded_cookie(signed_cookie: str) -> CookieData | None:
    if not signed_cookie:
        return None
    try:
        encoded_cookie = ts_signer.unsign(signed_cookie, max_age=MAX_COOKIE_AGE)
        return CookieData.model_validate(json.loads(encoded_cookie.decode()))
    except (BadSignature, SignatureExpired):
        return None


async def get_user_id(username: str):
    return await get_single_chat_user_attr_by_username(username, chat_user_table.c.user_id)


async def get_user_password_hash(username: str):
    return await get_single_chat_user_attr_by_username(username, chat_user_table.c.password_hash)


def hash_password(plain_text_password: str) -> str:
    return bcrypt.hashpw(
        plain_text_password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')


def new_cookie(user_id: int, username: str) -> str:
    return json.dumps({"user_id": user_id, "username": username})


def sign_cookie(cookie: str) -> str:
    return ts_signer.sign(cookie.encode()).decode()


def verify_password(plain_text_password: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(
        plain_text_password.encode('utf-8'),
        stored_hash.encode('utf-8')
    )
