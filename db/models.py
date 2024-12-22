from sqlalchemy import (create_engine, Boolean, Column, DateTime, ForeignKey,
                        Index, Integer, JSON, String, UUID)
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
    """
    A session is created each time a websocket connection is established. Messages received and sent on that connection
    are associated with the session.
    """
    __tablename__ = "chat_session"
    chat_session_id = Column(Integer, primary_key=True, autoincrement=True)
    is_hidden = Column(Boolean, default=False)
    summary = Column(String)
    time_started = Column(DateTime)
    time_stopped = Column(DateTime)

    # Relationship to ChatUser through the link table
    chat_users = relationship(
        "ChatUser",
        secondary="chat_session_chat_user_link", back_populates="chat_sessions")
    # Relationship to ChatUserProfile through the link table
    chat_user_profiles = relationship(
        "ChatUserProfile",
        secondary="chat_session_chat_user_profile_link", back_populates="chat_sessions")


class ChatRequestReceived(Base):
    __tablename__ = "chat_request_received"
    chat_session_id = Column(Integer, ForeignKey("chat_session.chat_session_id"), nullable=False)
    chat_request_received_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_request = Column(JSON)
    time_received = Column(DateTime)


class ChatResponseSent(Base):
    __tablename__ = "chat_response_sent"
    chat_session_id = Column(Integer, ForeignKey("chat_session.chat_session_id"), nullable=False)
    chat_request_received_id = Column(Integer, ForeignKey("chat_request_received.chat_request_received_id"),
                                      nullable=True)
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
    chat_user_first_name = Column(String)
    chat_user_last_name = Column(String)
    chat_user_middle_name_or_initials = Column(String)

    # Relationship to ChatSession through the link table
    chat_sessions = relationship(
        "ChatSession",
        secondary="chat_session_chat_user_link",
        back_populates="chat_users")


class ChatSessionChatUserLink(Base):
    __tablename__ = "chat_session_chat_user_link"
    chat_session_id = Column(Integer, ForeignKey('chat_session.chat_session_id'), primary_key=True)
    chat_user_id = Column(Integer, ForeignKey('chat_user.chat_user_id'), primary_key=True)


class ChatUserProfile(Base):
    __tablename__ = "chat_user_profile"
    chat_user_profile_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_user_profile = Column(JSON)

    # Relationship to ChatSession through the link table
    chat_sessions = relationship(
        "ChatSession",
        secondary="chat_session_chat_user_profile_link",
        back_populates="chat_user_profiles")


class ChatSessionChatUserProfileLink(Base):
    __tablename__ = "chat_session_chat_user_profile_link"
    chat_session_id = Column(Integer, ForeignKey('chat_session.chat_session_id'), primary_key=True)
    chat_user_profile_id = Column(Integer, ForeignKey('chat_user_profile.chat_user_profile_id'), primary_key=True)


class Sentence(Base):
    __tablename__ = "sentence"
    sentence_id = Column(Integer, primary_key=True, autoincrement=True)
    sentence_flair_text_features = Column(JSON)
    sentence_node_type = Column(String)
    sentence_openai_emotion_features = Column(JSON)
    sentence_openai_entity_features = Column(JSON)
    sentence_openai_text_features = Column(JSON)
    sentence_signature = Column(UUID)
    text = Column(String)

    __table_args__ = (
        Index('idx_sentence_signature', 'sentence_signature'),  # Secondary index
    )


# Create the tables in the database
Base.metadata.create_all(bind=engine)
