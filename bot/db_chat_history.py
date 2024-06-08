from sqlalchemy import create_engine, Column, DateTime, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import os

# Database Configuration
DATABASE_URL = "postgresql://{name}:{password}@{host}/chat_history".format(
    host=os.getenv("CHAT_HISTORY_DB_HOST"),
    name=os.getenv("CHAT_HISTORY_POSTGRES_USER"),
    password=os.getenv("CHAT_HISTORY_POSTGRES_PASSWORD"),
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


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


# Create the tables in the database
Base.metadata.create_all(bind=engine)
