# models.py
import os
from sqlalchemy import Column, Integer, String, Date, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class PowerballDraw(Base):
    __tablename__ = 'powerball_draws'
    id = Column(Integer, primary_key=True, autoincrement=True)
    draw_date = Column(Date, nullable=False, unique=True)
    white_balls = Column(String, nullable=False)  # e.g. "5 17 32 47 63"
    powerball = Column(String, nullable=False)     # e.g. "25"
    jackpot = Column(String, nullable=True)

class MegamillionsDraw(Base):
    __tablename__ = 'megamillions_draws'
    id = Column(Integer, primary_key=True, autoincrement=True)
    draw_date = Column(Date, nullable=False, unique=True)
    white_balls = Column(String, nullable=False)   # e.g. "3 15 18 32 57"
    megaball = Column(String, nullable=False)      # e.g. "9"
    jackpot = Column(String, nullable=True)

# Database setup
DB_FILE = "lottery.db"
engine = create_engine(f"sqlite:///{DB_FILE}", echo=False)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    # Initialize DB tables
    init_db()
    print("Database initialized.")
