from sqlalchemy import (create_engine, Boolean, Column, DateTime, ForeignKey,
                        Index, Integer, JSON, PrimaryKeyConstraint, String, UniqueConstraint)
from sqlalchemy.orm import declarative_base, sessionmaker

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


class ImageModel(Base):
    __tablename__ = "image_model"
    image_model_id = Column(Integer, primary_key=True, autoincrement=True)
    image_model_name = Column(String, unique=True, index=True)

    __table_args__ = (
        # Since we do these "does it exist?" checks based on the model name.
        Index("idx_image_model_image_model_name", "image_model_name"),
    )


class ImageModelChatRequestLink(Base):
    __tablename__ = "image_model_chat_request_link"
    chat_request_received_id = Column(Integer, ForeignKey("chat_request_received.chat_request_received_id"), nullable=False)
    image_model_id = Column(Integer, ForeignKey("image_model.image_model_id"), nullable=False)

    __table_args__ = (
        # Since each request / label relationship is unique.
        PrimaryKeyConstraint('chat_request_received_id', 'image_model_id', name='pk_image_model_chat_request_link'),
        # We'll mostly do this to look up labels for requests.
        Index("idx_image_model_chat_request_link_chat_request_received_id", "chat_request_received_id"),
        # We might want this to look up all the requests that match a certain label.
        Index("idx_image_model_chat_request_link_image_model_id", "image_model_id"),
    )


# Create the tables in the database
Base.metadata.create_all(bind=engine)
