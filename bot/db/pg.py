from sqlalchemy import Engine, MetaData, Table

from settings import germ_settings

DATABASE_URL = "{name}:{password}@{host}/germ".format(
    host=germ_settings.DB_HOST,
    name=germ_settings.POSTGRES_USER,
    password=germ_settings.POSTGRES_PASSWORD,
)


class TableHelper:
    def __init__(self, engine: Engine):
        self.chat_message_table = Table('chat_message', MetaData(), autoload_with=engine)
        self.chat_user_table = Table('chat_user', MetaData(), autoload_with=engine)
        self.conversation_table = Table('conversation', MetaData(), autoload_with=engine)
        self.engine = engine