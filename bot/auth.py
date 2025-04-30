from datetime import datetime
from sqlalchemy import Table, insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker as async_pg_session_maker
import bcrypt
import logging

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "ssid"
MAX_COOKIE_AGE: int = 60 * 60 * 24 * 7   # 7 days


class AuthHelper:
    def __init__(self, chat_user_table: Table, pg_session_maker: async_pg_session_maker):
        self.pg_session_maker = pg_session_maker
        self.chat_user_table = chat_user_table

    async def add_new_user(self, username: str, plain_text_password: str) -> (int, datetime):
        async with (self.pg_session_maker() as rdb_session):
            async with rdb_session.begin():
                try:
                    insert_stmt = insert(self.chat_user_table).values(
                        user_name=username,
                        password_hash=hash_password(plain_text_password),
                    ).returning(
                        self.chat_user_table.c.user_id,
                        self.chat_user_table.c.dt_created
                    )
                    result = await rdb_session.execute(insert_stmt)
                    row = result.first()
                    await rdb_session.commit()
                    user_id, dt_created = row
                    logger.info(f"New user '{username}' inserted successfully with user_id {user_id} at {dt_created}")
                    return user_id, dt_created
                except Exception as e:
                    logger.error(f"Failed to insert new user {username}: {e}")
                    await rdb_session.rollback()
                    raise


    async def get_single_chat_user_attr_by_username(self, username: str, attr):
        async with (self.pg_session_maker() as rdb_session):
            async with rdb_session.begin():
                chat_user_stmt = select(attr).where(
                    self.chat_user_table.c.user_name == username
                )
                record = (await rdb_session.execute(chat_user_stmt)).one_or_none()
                if record is not None:
                    return record[0]


    async def get_user_id(self, username: str):
        return await self.get_single_chat_user_attr_by_username(username, self.chat_user_table.c.user_id)


    async def get_user_password_hash(self, username: str):
        return await self.get_single_chat_user_attr_by_username(username, self.chat_user_table.c.password_hash)


def hash_password(plain_text_password: str) -> str:
    return bcrypt.hashpw(
        plain_text_password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')


def verify_password(plain_text_password: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(
        plain_text_password.encode('utf-8'),
        stored_hash.encode('utf-8')
    )
