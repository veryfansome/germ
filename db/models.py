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


class ChatMessageIdea(Base):
    """
    Message ideas are options for proactively interacting with users. They are generated over the course of chat
    sessions and prioritized for experimentation.
    """
    __tablename__ = "chat_message_idea"
    chat_message_idea_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_session_id = Column(Integer, ForeignKey("chat_session.chat_session_id"), nullable=False)
    context = Column(String)
    idea = Column(String, nullable=False)
    time_created = Column(DateTime)

    # Relationship to progenitors ideas through the link table
    progenitors = relationship(
        "ChatMessageIdea",
        secondary="chat_message_idea_chat_message_idea_link",
        primaryjoin="ChatMessageIdea.chat_message_idea_id == ChatMessageIdeaChatMessageIdeaLink.derivative_idea_id",
        foreign_keys="[ChatMessageIdeaChatMessageIdeaLink.derivative_idea_id]",
        back_populates="derivatives")
    # Relationship to derivatives ideas through the link table
    derivatives = relationship(
        "ChatMessageIdea",
        secondary="chat_message_idea_chat_message_idea_link",
        primaryjoin="ChatMessageIdea.chat_message_idea_id == ChatMessageIdeaChatMessageIdeaLink.progenitor_idea_id",
        foreign_keys="[ChatMessageIdeaChatMessageIdeaLink.progenitor_idea_id]",
        back_populates="progenitors")


class ChatMessageIdeaChatMessageIdeaLink(Base):
    __tablename__ = "chat_message_idea_chat_message_idea_link"
    progenitor_idea_id = Column(Integer, ForeignKey('chat_message_idea.chat_message_idea_id'), primary_key=True)
    derivative_idea_id = Column(Integer, ForeignKey('chat_message_idea.chat_message_idea_id'), primary_key=True)


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


# Create the tables in the database
Base.metadata.create_all(bind=engine)
