from sqlalchemy import (create_engine, Boolean, Column, DateTime, ForeignKey,
                        Index, Integer, JSON, String, UUID)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

from settings import germ_settings

# Database Configuration
DATABASE_URL = "{name}:{password}@{host}/germ".format(
    host=germ_settings.DB_HOST,
    name=germ_settings.POSTGRES_USER,
    password=germ_settings.POSTGRES_PASSWORD,
)
async_engine = create_async_engine(f"postgresql+asyncpg://{DATABASE_URL}", echo=False)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

engine = create_engine(f"postgresql+psycopg2://{DATABASE_URL}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class ChatSession(Base):
    """
    A session is created each time a websocket connection is established. Messages received and sent on that connection
    are associated with the session.
    """
    __tablename__ = "chat_session"
    chat_session_id = Column(Integer, primary_key=True, autoincrement=True)
    is_hidden = Column(Boolean, default=False)
    summary = Column(String)
    time_started = Column(DateTime(timezone=True))
    time_stopped = Column(DateTime(timezone=True))


class ChatRequestReceived(Base):
    __tablename__ = "chat_request_received"
    chat_session_id = Column(Integer, ForeignKey("chat_session.chat_session_id"), nullable=False)
    chat_request_received_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_request = Column(JSON)
    time_received = Column(DateTime(timezone=True))


class ChatResponseSent(Base):
    __tablename__ = "chat_response_sent"
    chat_session_id = Column(Integer, ForeignKey("chat_session.chat_session_id"), nullable=False)
    chat_request_received_id = Column(Integer, ForeignKey("chat_request_received.chat_request_received_id"),
                                      nullable=True)
    chat_response_sent_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_response = Column(JSON)
    time_sent = Column(DateTime(timezone=True))

    __table_args__ = (
        # Since we look up responses by session for bookmarks.
        Index("idx_chat_response_sent_chat_session_id", "chat_session_id"),
    )


class CodeBlock(Base):
    __tablename__ = "code_block"
    code_block_id = Column(Integer, primary_key=True, autoincrement=True)

    code_block_signature = Column(UUID)
    text = Column(String)

    __table_args__ = (
        Index('idx_code_block_signature', 'code_block_signature'),  # Secondary index
    )


class Paragraph(Base):
    __tablename__ = "paragraph"
    paragraph_id = Column(Integer, primary_key=True, autoincrement=True)

    paragraph_signature = Column(UUID)
    text = Column(String)

    __table_args__ = (
        Index('idx_paragraph_signature', 'paragraph_signature'),  # Secondary index
    )


class Sentence(Base):
    __tablename__ = "sentence"
    sentence_id = Column(Integer, primary_key=True, autoincrement=True)

    sentence_openai_parameters = Column(JSON)
    sentence_openai_parameters_time_changed = Column(DateTime(timezone=True))
    sentence_openai_parameters_fetch_count = Column(Integer, default=0)

    sentence_signature = Column(UUID)
    text = Column(String)

    __table_args__ = (
        Index('idx_sentence_signature', 'sentence_signature'),  # Secondary index
    )


# Create the tables in the database
Base.metadata.create_all(bind=engine)
