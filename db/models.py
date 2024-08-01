from sqlalchemy import create_engine, Boolean, Column, DateTime, ForeignKey, Integer, JSON, LargeBinary, String
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import os

# Database Configuration
DATABASE_URL = "{name}:{password}@{host}/germ".format(
    host=os.getenv("DB_HOST"),
    name=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)
engine = create_engine(f"postgresql+psycopg2://{DATABASE_URL}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "chat_session"
    chat_session_id = Column(Integer, primary_key=True, autoincrement=True)
    is_hidden = Column(Boolean, default=False)
    summary = Column(String)
    time_started = Column(DateTime, default=datetime.datetime.now(datetime.UTC))
    time_stopped = Column(DateTime)


class ChatRequestReceived(Base):
    __tablename__ = "chat_request_received"
    chat_session_id = Column(Integer, ForeignKey("chat_session.chat_session_id"), nullable=False)
    chat_request_received_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_request = Column(JSON)
    time_received = Column(DateTime, default=datetime.datetime.now(datetime.UTC))


class ChatResponseSent(Base):
    __tablename__ = "chat_response_sent"
    chat_session_id = Column(Integer, ForeignKey("chat_session.chat_session_id"), nullable=False)
    chat_request_received_id = Column(Integer, ForeignKey("chat_request_received.chat_request_received_id"), nullable=False)
    chat_response_sent_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_response = Column(JSON)
    time_sent = Column(DateTime, default=datetime.datetime.now(datetime.UTC))


class MessageBookmark(Base):
    __tablename__ = "message_bookmark"
    id = Column(Integer, primary_key=True, autoincrement=True)
    is_test = Column(Boolean, default=True)
    message_received_id = Column(Integer, ForeignKey("message_received.id"), nullable=False)
    message_replied_id = Column(Integer, ForeignKey("message_replied.id"), nullable=False)
    message_summary = Column(LargeBinary)
    timestamp = Column(DateTime, default=datetime.datetime.now(datetime.UTC))


class MessageReceived(Base):
    __tablename__ = "message_received"
    id = Column(Integer, primary_key=True, index=True)
    chat_frame = Column(JSON)
    content = Column(LargeBinary)
    role = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.now(datetime.UTC))


class MessageReplied(Base):
    __tablename__ = "message_replied"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(LargeBinary)
    message_received_id = Column(Integer, ForeignKey("message_received.id"))
    role = Column(String)
    tool_calls = Column(JSON)
    timestamp = Column(DateTime, default=datetime.datetime.now(datetime.UTC))


# Create the tables in the database
Base.metadata.create_all(bind=engine)
