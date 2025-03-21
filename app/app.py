# app/app.py
import os
from flask import Flask, render_template, request
from datetime import datetime, date
import numpy as np
import torch
import torch.nn.functional as F

from db.models import init_db, SessionLocal, PowerballDraw, MegamillionsDraw

############################
# Load saved models if they exist
############################

POWERBALL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "training", "powerball_model.pt")
MEGAMILLIONS_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "training", "megamillions_model.pt")

app = Flask(__name__)

model_powerball = None
model_megamillions = None

# We define MLP classes inline to match loaded weights:

class PowerballMLP(torch.nn.Module):
    def __init__(self, input_dim=95, hidden_dim=128, output_dim=95):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class MegamillionsMLP(torch.nn.Module):
    def __init__(self, input_dim=95, hidden_dim=128, output_dim=95):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

# Attempt to load them
if os.path.exists(POWERBALL_MODEL_PATH):
    model_powerball = PowerballMLP()
    model_powerball.load_state_dict(torch.load(POWERBALL_MODEL_PATH, map_location="cpu"))
    model_powerball.eval()

if os.path.exists(MEGAMILLIONS_MODEL_PATH):
    model_megamillions = MegamillionsMLP()
    model_megamillions.load_state_dict(torch.load(MEGAMILLIONS_MODEL_PATH, map_location="cpu"))
    model_megamillions.eval()

############################
# Frequency calculations
############################

def fetch_powerball_in_range(start_dt, end_dt):
    session = SessionLocal()
    draws = (session.query(PowerballDraw)
             .filter(PowerballDraw.draw_date >= start_dt,
                     PowerballDraw.draw_date <= end_dt)
             .order_by(PowerballDraw.draw_date)
             .all())
    session.close()
    return draws

def fetch_megamillions_in_range(start_dt, end_dt):
    session = SessionLocal()
    draws = (session.query(MegamillionsDraw)
             .filter(MegamillionsDraw.draw_date >= start_dt,
                     MegamillionsDraw.draw_date <= end_dt)
             .order_by(MegamillionsDraw.draw_date)
             .all())
    session.close()
    return draws

def frequency_powerball(draws):
    white_freq = {i:0 for i in range(1,70)}
    red_freq = {i:0 for i in range(1,27)}
    for d in draws:
        whites = [int(x) for x in d.white_balls.split()]
        pb = int(d.powerball)
        for w in whites:
            white_freq[w] += 1
        red_freq[pb] += 1
    return white_freq, red_freq

def frequency_megamillions(draws):
    white_freq = {i:0 for i in range(1,71)}
    red_freq = {i:0 for i in range(1,26)}
    for d in draws:
        whites = [int(x) for x in d.white_balls.split()]
        mb = int(d.megaball)
        for w in whites:
            white_freq[w] += 1
        red_freq[mb] += 1
    return white_freq, red_freq

############################
# Minimal feature vector for AI predictions
############################

def feature_vector_powerball(draws):
    white = np.zeros(69, dtype=np.float32)
    red = np.zeros(26, dtype=np.float32)
    for d in draws:
        ws = [int(x) for x in d.white_balls.split()]
        pb = int(d.powerball)
        for w in ws:
            white[w-1] += 1
        red[pb-1] += 1
    total = len(draws)
    if total>0:
        white/= total
        red/= total
    return np.concatenate([white, red])  # shape (95,)

def feature_vector_megamillions(draws):
    white = np.zeros(70, dtype=np.float32)
    red = np.zeros(25, dtype=np.float32)
    for d in draws:
        ws = [int(x) for x in d.white_balls.split()]
        mb = int(d.megaball)
        for w in ws:
            white[w-1]+=1
        red[mb-1]+=1
    total = len(draws)
    if total>0:
        white/= total
        red/= total
    return np.concatenate([white, red])  # shape (95,)

############################
# Flask Routes
############################

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/frequency", methods=["POST"])
def frequency():
    game = request.form.get("game")
    start_str = request.form.get("start_date")
    end_str = request.form.get("end_date")

    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()
    except:
        return "Invalid date format. Use YYYY-MM-DD."

    if game=="Powerball":
        draws = fetch_powerball_in_range(start_dt, end_dt)
        white_freq, red_freq = frequency_powerball(draws)
    else:
        draws = fetch_megamillions_in_range(start_dt, end_dt)
        white_freq, red_freq = frequency_megamillions(draws)

    # Turn freq dict into sorted lists
    def to_xy(freq):
        items = sorted(freq.items(), key=lambda x: x[0])
        x = [str(k) for k,_ in items]
        y = [v for _,v in items]
        return x, y

    xw, yw = to_xy(white_freq)
    xr, yr = to_xy(red_freq)

    return render_template("frequency.html",
                           game=game,
                           start_date=start_dt,
                           end_date=end_dt,
                           xw=xw,
                           yw=yw,
                           xr=xr,
                           yr=yr)

@app.route("/ai_predict", methods=["POST"])
def ai_predict():
    game = request.form.get("game")
    session = SessionLocal()

    if game=="Powerball":
        # get last 20
        draws = (session.query(PowerballDraw)
                 .order_by(PowerballDraw.draw_date.desc())
                 .limit(20)
                 .all())
        draws = list(reversed(draws))
        feats = feature_vector_powerball(draws)
        session.close()
        if model_powerball is None:
            return "No Powerball model loaded."

        with torch.no_grad():
            x_t = torch.from_numpy(feats).unsqueeze(0)
            logits = model_powerball(x_t)
            probs = torch.sigmoid(logits).numpy().flatten()

        white_probs=probs[:69]
        red_probs=probs[69:]
        # pick top 5 white
        top5 = np.argpartition(white_probs, -5)[-5:]
        top5 = top5[np.argsort(white_probs[top5])[::-1]]
        best_whites = list(top5+1)
        best_red = int(np.argmax(red_probs)+1)

        return render_template("predict.html",
                               game=game,
                               white_balls=best_whites,
                               red_ball=best_red)
    else:
        # Mega Millions
        draws = (session.query(MegamillionsDraw)
                 .order_by(MegamillionsDraw.draw_date.desc())
                 .limit(20)
                 .all())
        draws = list(reversed(draws))
        feats = feature_vector_megamillions(draws)
        session.close()

        if model_megamillions is None:
            return "No Mega Millions model loaded."

        with torch.no_grad():
            x_t = torch.from_numpy(feats).unsqueeze(0)
            logits = model_megamillions(x_t)
            probs = torch.sigmoid(logits).numpy().flatten()

        white_probs = probs[:70]
        red_probs   = probs[70:]
        top5 = np.argpartition(white_probs, -5)[-5:]
        top5 = top5[np.argsort(white_probs[top5])[::-1]]
        best_whites = list(top5+1)
        best_red = int(np.argmax(red_probs)+1)

        return render_template("predict.html",
                               game=game,
                               white_balls=best_whites,
                               red_ball=best_red)

@app.route("/manual_combos", methods=["GET","POST"])
def manual_combos():
    if request.method=="GET":
        return render_template("combos.html", combos=None)

    game = request.form.get("game")
    white_str = request.form.get("white_list","")
    red_str   = request.form.get("red_list","")

    try:
        white_nums = sorted(set(int(x) for x in white_str.split(",") if x.strip()))
        red_nums   = sorted(set(int(x) for x in red_str.split(",") if x.strip()))
    except ValueError:
        return "Invalid input, please use comma-separated integers."

    if game=="Powerball":
        # 1..69 for white, 1..26 for red
        white_nums=[w for w in white_nums if 1<=w<=69]
        red_nums=[r for r in red_nums if 1<=r<=26]
    else:
        white_nums=[w for w in white_nums if 1<=w<=70]
        red_nums=[r for r in red_nums if 1<=r<=25]

    if len(white_nums)<5 or len(red_nums)<1:
        return "Not enough candidate numbers."

    import itertools
    combos=[]
    for combo in itertools.combinations(white_nums, 5):
        combo_sorted=sorted(combo)
        for r in red_nums:
            combos.append((combo_sorted, r))

    return render_template("combos.html",
                           combos=combos,
                           white_list=white_str,
                           red_list=red_str,
                           game=game)

if __name__=="__main__":
    init_db()
    app.run(debug=True)
