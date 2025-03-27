# scripts/update_draws.py
import datetime
import time
import random
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException, WebDriverException
from db.models import SessionLocal, PowerballDraw, MegamillionsDraw

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.5735.110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.5735.110 Safari/537.36 Edg/114.0.0.0",
]

def get_driver():
    """
    Creates an undetected-chromedriver instance with random user-agent,
    random window size, and extended timeouts.
    """
    user_agent = random.choice(USER_AGENTS)
    options = uc.ChromeOptions()
    # Uncomment below if you want to see the browser actions:
    # options.add_argument("--headless=new")
    width = random.randint(1000, 1600)
    height = random.randint(700, 900)
    options.add_argument(f"--window-size={width},{height}")
    options.add_argument(f"--user-agent={user_agent}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    driver = uc.Chrome(options=options)
    driver.set_page_load_timeout(300)  # up to 5 minutes
    return driver

def scrape_draws(game, start_date, end_date, existing_dates):
    """
    Scrapes new winning number draws for a given game between start_date and end_date.
    Parameters:
      game: "Powerball" or "Megamillions"
      start_date, end_date: datetime.date objects specifying the window.
      existing_dates: set of draw date strings ("MM/DD/YYYY") already in the database.
    
    Returns:
      A list of dictionaries with keys:
        - "Draw Date": string in "MM/DD/YYYY"
        - "White Balls": string with 5 numbers separated by spaces
        - "Powerball" or "MegaBall": string (depending on the game)
        - "Jackpot": string (if available)
    """
    if game == "Powerball":
        base_url = "https://www.usamega.com/powerball/results/"
        table_selector = "table.results.pb tbody tr"
        red_key = "Powerball"
    else:
        base_url = "https://www.usamega.com/mega-millions/results/"
        table_selector = "table.results.mm tbody tr"
        red_key = "MegaBall"

    all_new_records = []
    driver = get_driver()
    page = 1
    try:
        while True:
            url = f"{base_url}{page}"
            print(f"[INFO] Loading page: {url}")
            success = False
            for attempt in range(3):
                try:
                    driver.get(url)
                    success = True
                    break
                except TimeoutException:
                    print(f"[WARNING] Timeout on {url}, attempt {attempt+1}. Retrying...")
                    time.sleep(3)
                except WebDriverException as e:
                    print(f"[ERROR] WebDriverException: {e}")
                    success = False
                    break
            if not success:
                print("[ERROR] Could not load page after retries. Stopping.")
                break

            time.sleep(random.uniform(2,5))
            html = driver.page_source
            # Check for Cloudflare block or similar issues.
            if any(s in html for s in ["cf-error-details", "Access Denied", "You have been blocked"]):
                print("[WARNING] Detected block. Stopping scraping.")
                break

            soup = BeautifulSoup(html, "html.parser")
            rows = soup.select(table_selector)
            if not rows:
                print(f"[INFO] No rows found on page {page}. End of pagination.")
                break

            found_any = False
            for tr in rows:
                tds = tr.find_all("td")
                if len(tds) < 2:
                    continue
                section = tds[0].select_one("section.results")
                if not section:
                    continue
                date_a = section.find("a")
                if not date_a:
                    continue
                date_text = date_a.get_text(strip=True)  # e.g. "Wed, March, 19, 2025"
                parts = [p.strip() for p in date_text.split(",")]
                if len(parts) < 4:
                    continue
                date_str = f"{parts[1]} {parts[2]} {parts[3]}"
                try:
                    draw_dt = datetime.datetime.strptime(date_str, "%B %d %Y").date()
                except ValueError:
                    continue
                if draw_dt < start_date or draw_dt > end_date:
                    continue
                draw_date_str = draw_dt.strftime("%m/%d/%Y")
                if draw_date_str in existing_dates:
                    continue

                ul = section.find("ul")
                if not ul:
                    continue
                lis = ul.find_all("li")
                white_balls = []
                red_ball = None
                for li in lis:
                    classes = li.get("class", [])
                    val = li.get_text(strip=True)
                    if "bonus" in classes:
                        red_ball = val
                    elif "multiplier" in classes:
                        continue
                    else:
                        white_balls.append(val)
                if len(white_balls) < 5 or not red_ball:
                    continue

                jackpot_a = tds[1].find("a")
                jackpot = jackpot_a.get_text(strip=True) if jackpot_a else ""
                record = {
                    "Draw Date": draw_date_str,
                    "White Balls": " ".join(white_balls[:5]),
                    red_key: red_ball,
                    "Jackpot": jackpot,
                }
                all_new_records.append(record)
                found_any = True

            if not found_any:
                print(f"[INFO] No valid draws found on page {page}. Ending pagination.")
                break

            # Check for a "next page" link.
            next_link = soup.select_one(f'a.button[href*="/results/{page+1}"]')
            if next_link:
                time.sleep(random.uniform(2,6))
                page += 1
            else:
                print("[INFO] No next page link found. Ending scraping.")
                break
    finally:
        driver.quit()
    return all_new_records

def update_powerball():
    session = SessionLocal()
    existing_records = session.query(PowerballDraw.draw_date).all()
    existing_dates = set(d.strftime("%m/%d/%Y") for (d,) in existing_records if d)
    
    # In auto_update.py, the start_date and end_date will be passed dynamically.
    # For testing purposes, you can uncomment and set:
    # start_date = datetime.datetime.strptime("2025-03-15", "%Y-%m-%d").date()
    # end_date = datetime.datetime.strptime("2025-03-20", "%Y-%m-%d").date()
    # new_draws = scrape_draws("Powerball", start_date, end_date, existing_dates)
    
    new_draws = []  # placeholder if not using auto_update.py
    for d in new_draws:
        dt = datetime.datetime.strptime(d['Draw Date'], "%m/%d/%Y").date()
        if session.query(PowerballDraw).filter_by(draw_date=dt).first():
            continue
        record = PowerballDraw(
            draw_date=dt,
            white_balls=d['White Balls'],
            powerball=d['Powerball'],
            jackpot=d.get('Jackpot','')
        )
        session.add(record)
    session.commit()
    session.close()

def update_megamillions():
    session = SessionLocal()
    existing_records = session.query(MegamillionsDraw.draw_date).all()
    existing_dates = set(d.strftime("%m/%d/%Y") for (d,) in existing_records if d)
    new_draws = []  # placeholder if not using auto_update.py
    for d in new_draws:
        dt = datetime.datetime.strptime(d['Draw Date'], "%m/%d/%Y").date()
        if session.query(MegamillionsDraw).filter_by(draw_date=dt).first():
            continue
        record = MegamillionsDraw(
            draw_date=dt,
            white_balls=d['White Balls'],
            megaball=d['MegaBall'],
            jackpot=d.get('Jackpot','')
        )
        session.add(record)
    session.commit()
    session.close()

if __name__ == "__main__":
    update_powerball()
    update_megamillions()
    print("Weekly update completed.")
