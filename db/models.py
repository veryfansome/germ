from sqlalchemy import (create_engine, Boolean, Column, DateTime, ForeignKey,
                        Index, Integer, JSON, String)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from settings import germ_settings

# Database Configuration
DATABASE_URL = "{name}:{password}@{host}/germ".format(
    host=germ_settings.DB_HOST,
    name=germ_settings.POSTGRES_USER,
    password=germ_settings.POSTGRES_PASSWORD,
)
engine = create_engine(f"postgresql+psycopg2://{DATABASE_URL}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "chat_session"
    chat_session_id = Column(Integer, primary_key=True, autoincrement=True)
    is_hidden = Column(Boolean, default=False)
    summary = Column(String)
    time_started = Column(DateTime)
    time_stopped = Column(DateTime)

    # Relationship to ChatUser through the association table
    users = relationship("ChatUser", secondary="chat_session_user", back_populates="sessions")


class ChatRequestReceived(Base):
    __tablename__ = "chat_request_received"
    chat_session_id = Column(Integer, ForeignKey("chat_session.chat_session_id"), nullable=False)
    chat_request_received_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_request = Column(JSON)
    time_received = Column(DateTime)


class ChatResponseSent(Base):
    __tablename__ = "chat_response_sent"
    chat_session_id = Column(Integer, ForeignKey("chat_session.chat_session_id"), nullable=False)
    chat_request_received_id = Column(Integer, ForeignKey("chat_request_received.chat_request_received_id"), nullable=False)
    chat_response_sent_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_response = Column(JSON)
    time_sent = Column(DateTime)

    __table_args__ = (
        # Since we look up responses by session for bookmarks.
        Index("idx_chat_response_sent_chat_session_id", "chat_session_id"),
    )


class ChatUser(Base):
    __tablename__ = "chat_user"
    chat_user_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_user_name = Column(String)

    # Relationship to ChatSession through the association table
    sessions = relationship("ChatSession", secondary="chat_session_user", back_populates="users")


class ChatSessionUser(Base):
    __tablename__ = "chat_session_user"
    chat_session_id = Column(Integer, ForeignKey('chat_session.chat_session_id'), primary_key=True)
    chat_user_id = Column(Integer, ForeignKey('chat_user.chat_user_id'), primary_key=True)


# Create the tables in the database
Base.metadata.create_all(bind=engine)
