# scripts/update_draws.py
import datetime
from db.models import SessionLocal, PowerballDraw, MegamillionsDraw

def fetch_new_powerball():
    """
    STUB: fetch new draws from scraping or API
    Returns a list of dicts:
    [{
      'draw_date': datetime.date(2025, 3, 22),
      'white_balls': "1 2 3 4 5",
      'powerball': "6",
      'jackpot': "$100 Million"
    }, ...]
    """
    return []

def fetch_new_megamillions():
    """
    Same format as above
    """
    return []

def update_powerball():
    session = SessionLocal()
    new_draws = fetch_new_powerball()
    for d in new_draws:
        dt = d['draw_date']
        existing = session.query(PowerballDraw).filter_by(draw_date=dt).first()
        if existing:
            continue
        record = PowerballDraw(
            draw_date=dt,
            white_balls=d['white_balls'],
            powerball=d['powerball'],
            jackpot=d.get('jackpot','')
        )
        session.add(record)
    session.commit()
    session.close()

def update_megamillions():
    session = SessionLocal()
    new_draws = fetch_new_megamillions()
    for d in new_draws:
        dt = d['draw_date']
        existing = session.query(MegamillionsDraw).filter_by(draw_date=dt).first()
        if existing:
            continue
        record = MegamillionsDraw(
            draw_date=dt,
            white_balls=d['white_balls'],
            megaball=d['megaball'],
            jackpot=d.get('jackpot','')
        )
        session.add(record)
    session.commit()
    session.close()

if __name__ == "__main__":
    update_powerball()
    update_megamillions()
    print("Weekly update completed.")
