from sqlalchemy import create_engine, Column, DateTime, ForeignKey, Integer, LargeBinary, String, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import os

# Database Configuration
DATABASE_URL = "postgresql://{name}:{password}@{host}/germ".format(
    host=os.getenv("DB_HOST"),
    name=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class MessageBookmark(Base):
    __tablename__ = "message_bookmark"
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_received_id = Column(Integer, ForeignKey("message_received.id"), nullable=False)
    message_sent_id = Column(Integer, ForeignKey("message_sent.id"), nullable=False)
    message_summary = Column(LargeBinary)
    __table_args__ = (
        UniqueConstraint('message_received_id', 'message_sent_id', name='unique_message_received_id_message_sent_id'),
    )


# Define the ChatHistory model
class MessageReceived(Base):
    __tablename__ = "message_received"
    id = Column(Integer, primary_key=True, index=True)
    role = Column(String)
    content = Column(LargeBinary)
    chat_frame = Column(LargeBinary)
    timestamp = Column(DateTime, default=datetime.datetime.now(datetime.UTC))


class MessageSent(Base):
    __tablename__ = "message_sent"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(LargeBinary)
    message_received_id = Column(Integer, ForeignKey("message_received.id"))
    timestamp = Column(DateTime, default=datetime.datetime.now(datetime.UTC))


class MessageThumbsDown(Base):
    __tablename__ = "message_thumbs_down"
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_received_id = Column(Integer, ForeignKey("message_received.id"), nullable=False)
    message_sent_id = Column(Integer, ForeignKey("message_sent.id"), nullable=False)


# Create the tables in the database
Base.metadata.create_all(bind=engine)