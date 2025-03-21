# scripts/import_csv.py
import csv
import datetime
from db.models import init_db, SessionLocal, PowerballDraw, MegamillionsDraw

def import_powerball(csv_file):
    session = SessionLocal()
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # e.g. "Draw Date", "White Balls", "Powerball", "Jackpot"
            draw_date_str = row["Draw Date"]
            white_balls = row["White Balls"]
            pb = row["Powerball"]
            jp = row.get("Jackpot", "")

            mm, dd, yyyy = draw_date_str.split("/")
            dt = datetime.date(int(yyyy), int(mm), int(dd))

            existing = session.query(PowerballDraw).filter_by(draw_date=dt).first()
            if existing:
                continue

            record = PowerballDraw(
                draw_date=dt,
                white_balls=white_balls,
                powerball=pb,
                jackpot=jp
            )
            session.add(record)
    session.commit()
    session.close()

def import_megamillions(csv_file):
    session = SessionLocal()
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # e.g. "Draw Date", "White Balls", "MegaBall", "Jackpot"
            draw_date_str = row["Draw Date"]
            white_balls = row["White Balls"]
            mb = row["MegaBall"]
            jp = row.get("Jackpot", "")

            mm, dd, yyyy = draw_date_str.split("/")
            dt = datetime.date(int(yyyy), int(mm), int(dd))

            existing = session.query(MegamillionsDraw).filter_by(draw_date=dt).first()
            if existing:
                continue

            record = MegamillionsDraw(
                draw_date=dt,
                white_balls=white_balls,
                megaball=mb,
                jackpot=jp
            )
            session.add(record)
    session.commit()
    session.close()

if __name__ == "__main__":
    init_db()
    import_powerball("powerball_results.csv")
    import_megamillions("megamillions_results.csv")
    print("Imported CSV data successfully!")
