#!/usr/bin/env python3.11
"""
auto_update.py

This script does the following:
1. For each game (Powerball and Mega Millions):
   - Checks the database for the most recent draw date.
   - Sets the scraping window to cover the previous week (Monday–Saturday relative to Sunday 10AM EST).
   - Calls the scraping function (scrape_draws from update_draws.py) to fetch new draws.
   - Inserts new draws into the database.
2. Triggers retraining of the AI models by running the training scripts.
"""

import os
from datetime import datetime, date, timedelta
from db.models import SessionLocal, PowerballDraw, MegamillionsDraw
from scripts.update_draws import scrape_draws

def get_last_date(game):
    session = SessionLocal()
    if game == "Powerball":
        record = session.query(PowerballDraw).order_by(PowerballDraw.draw_date.desc()).first()
        default_date = datetime.strptime("2015-10-07", "%Y-%m-%d").date()
    else:
        record = session.query(MegamillionsDraw).order_by(MegamillionsDraw.draw_date.desc()).first()
        default_date = datetime.strptime("2017-10-31", "%Y-%m-%d").date()
    session.close()
    return record.draw_date if record else default_date

def update_game(game, new_start, new_end):
    # Get existing draw dates in DB
    session = SessionLocal()
    if game == "Powerball":
        existing_records = session.query(PowerballDraw.draw_date).all()
    else:
        existing_records = session.query(MegamillionsDraw.draw_date).all()
    session.close()
    existing_dates = set(d.strftime("%m/%d/%Y") for (d,) in existing_records if d)
    
    # Scrape new draws
    new_records = scrape_draws(game, new_start, new_end, existing_dates)
    print(f"[INFO] {game}: Found {len(new_records)} new records.")
    
    session = SessionLocal()
    if game == "Powerball":
        for rec in new_records:
            dt = datetime.strptime(rec["Draw Date"], "%m/%d/%Y").date()
            if session.query(PowerballDraw).filter_by(draw_date=dt).first():
                continue
            new_rec = PowerballDraw(
                draw_date=dt,
                white_balls=rec["White Balls"],
                powerball=rec["Powerball"],
                jackpot=rec.get("Jackpot", "")
            )
            session.add(new_rec)
    else:
        for rec in new_records:
            dt = datetime.strptime(rec["Draw Date"], "%m/%d/%Y").date()
            if session.query(MegamillionsDraw).filter_by(draw_date=dt).first():
                continue
            new_rec = MegamillionsDraw(
                draw_date=dt,
                white_balls=rec["White Balls"],
                megaball=rec["MegaBall"],
                jackpot=rec.get("Jackpot", "")
            )
            session.add(new_rec)
    session.commit()
    session.close()
    print(f"[INFO] {game} database updated.")

def run_training():
    print("[INFO] Starting training for Powerball model...")
    os.system("python3.11 training/train_powerball.py")
    print("[INFO] Starting training for Mega Millions model...")
    os.system("python3.11 training/train_megamillions.py")
    print("[INFO] Training complete.")

def main():
    # Set scraping window for the previous week (assuming this runs on Sunday at 10AM EST)
    today = date.today()  # This should be Sunday.
    # Previous week: Monday to Saturday.
    scraped_start = today - timedelta(days=6)  # Monday (if today is Sunday, weekday() returns 6)
    scraped_end = today - timedelta(days=1)      # Saturday

    print(f"[INFO] Scraping window: {scraped_start} to {scraped_end}")

    for game in ["Powerball", "Megamillions"]:
        last_date = get_last_date(game)
        new_start = max(scraped_start, last_date + timedelta(days=1))
        new_end = scraped_end
        print(f"[INFO] {game}: Last DB date: {last_date}. New range: {new_start} to {new_end}")
        update_game(game, new_start, new_end)

    run_training()

if __name__ == "__main__":
    main()
