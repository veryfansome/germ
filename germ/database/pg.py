from sqlalchemy import Engine, MetaData, Table

from germ.settings import germ_settings

DATABASE_URL = "{name}:{password}@{host}/germ".format(
    host=germ_settings.POSTGRES_HOST,
    name=germ_settings.POSTGRES_USER,
    password=germ_settings.POSTGRES_PASSWORD,
)


class TableHelper:
    def __init__(self, engine: Engine):
        self.chat_user_table = Table('chat_user', MetaData(), autoload_with=engine)
        self.conversation_state_table = Table('conversation_state', MetaData(), autoload_with=engine)
        self.conversation_table = Table('conversation', MetaData(), autoload_with=engine)
        self.struct_type_table = Table('struct_type', MetaData(), autoload_with=engine)
        self.engine = engine