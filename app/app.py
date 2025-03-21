# app/app.py
import os
from flask import Flask, render_template, request
from datetime import datetime, date
import numpy as np
import torch
import torch.nn.functional as F

from db.models import init_db, SessionLocal, PowerballDraw, MegamillionsDraw

###############################################
# Load PyTorch Models for Predictions
###############################################
POWERBALL_MODEL_PATH = os.path.join("training", "powerball_model.pt")
MEGAMILLIONS_MODEL_PATH = os.path.join("training", "megamillions_model.pt")

app = Flask(__name__)

model_powerball = None
model_megamillions = None

# Define example MLP classes (should match your training)
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

if os.path.exists(POWERBALL_MODEL_PATH):
    model_powerball = PowerballMLP()
    model_powerball.load_state_dict(torch.load(POWERBALL_MODEL_PATH, map_location="cpu"))
    model_powerball.eval()

if os.path.exists(MEGAMILLIONS_MODEL_PATH):
    model_megamillions = MegamillionsMLP()
    model_megamillions.load_state_dict(torch.load(MEGAMILLIONS_MODEL_PATH, map_location="cpu"))
    model_megamillions.eval()

###############################################
# Utility Functions: Frequency Calculation
###############################################
def fetch_powerball_draws(start_dt, end_dt):
    session = SessionLocal()
    q = session.query(PowerballDraw)
    if start_dt is not None:
        q = q.filter(PowerballDraw.draw_date >= start_dt)
    if end_dt is not None:
        q = q.filter(PowerballDraw.draw_date <= end_dt)
    draws = q.order_by(PowerballDraw.draw_date).all()
    session.close()
    return draws

def fetch_megamillions_draws(start_dt, end_dt):
    session = SessionLocal()
    q = session.query(MegamillionsDraw)
    if start_dt is not None:
        q = q.filter(MegamillionsDraw.draw_date >= start_dt)
    if end_dt is not None:
        q = q.filter(MegamillionsDraw.draw_date <= end_dt)
    draws = q.order_by(MegamillionsDraw.draw_date).all()
    session.close()
    return draws

def frequency_powerball(draws):
    white_freq = {i: 0 for i in range(1, 70)}
    red_freq   = {i: 0 for i in range(1, 27)}
    for d in draws:
        ws = [int(x) for x in d.white_balls.split()]
        pb = int(d.powerball)
        for w in ws:
            white_freq[w] += 1
        red_freq[pb] += 1
    return white_freq, red_freq

def frequency_megamillions(draws):
    white_freq = {i: 0 for i in range(1, 71)}
    red_freq   = {i: 0 for i in range(1, 26)}
    for d in draws:
        ws = [int(x) for x in d.white_balls.split()]
        mb = int(d.megaball)
        for w in ws:
            white_freq[w] += 1
        red_freq[mb] += 1
    return white_freq, red_freq

def make_xy(freq_dict):
    items = sorted(freq_dict.items(), key=lambda x: x[0])
    x = [str(k) for k, _ in items]
    y = [v for _, v in items]
    return x, y

###############################################
# Utility Functions: Feature Vectors for AI Prediction
###############################################
def feature_vector_powerball(draws):
    """Compute normalized frequency vector from given draws for Powerball."""
    white_counts = np.zeros(69, dtype=np.float32)
    red_counts = np.zeros(26, dtype=np.float32)
    for d in draws:
        ws = [int(x) for x in d.white_balls.split()]
        pb = int(d.powerball)
        for w in ws:
            white_counts[w - 1] += 1
        red_counts[pb - 1] += 1
    total = len(draws)
    if total:
        white_counts /= total
        red_counts /= total
    return np.concatenate([white_counts, red_counts])  # shape (95,)

def feature_vector_megamillions(draws):
    white_counts = np.zeros(70, dtype=np.float32)
    red_counts = np.zeros(25, dtype=np.float32)
    for d in draws:
        ws = [int(x) for x in d.white_balls.split()]
        mb = int(d.megaball)
        for w in ws:
            white_counts[w - 1] += 1
        red_counts[mb - 1] += 1
    total = len(draws)
    if total:
        white_counts /= total
        red_counts /= total
    return np.concatenate([white_counts, red_counts])  # shape (95,)

###############################################
# Routes
###############################################

@app.route("/")
def index():
    # Render the home page
    return render_template("index.html")

@app.route("/frequency", methods=["POST"])
def show_frequency():
    game = request.form.get("game")
    start_str = request.form.get("start_date", "")
    end_str = request.form.get("end_date", "")
    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_dt   = datetime.strptime(end_str, "%Y-%m-%d").date()
    except Exception as e:
        return "Invalid date format. Please use YYYY-MM-DD."
    
    if game == "Powerball":
        draws = fetch_powerball_draws(start_dt, end_dt)
        wfreq, rfreq = frequency_powerball(draws)
    else:
        draws = fetch_megamillions_draws(start_dt, end_dt)
        wfreq, rfreq = frequency_megamillions(draws)
    
    xw, yw = make_xy(wfreq)
    xr, yr = make_xy(rfreq)
    
    # Convert dates to strings for display in the template.
    return render_template("frequency.html",
                           game=game,
                           start_date=start_dt.strftime("%Y-%m-%d"),
                           end_date=end_dt.strftime("%Y-%m-%d"),
                           xw=xw, yw=yw,
                           xr=xr, yr=yr)

@app.route("/ai_predict", methods=["GET", "POST"])
def ai_predict():
    if request.method == "POST":
        game = request.form.get("game")
        start_str = request.form.get("start_date", "")
        end_str   = request.form.get("end_date", "")
        num_sets  = request.form.get("num_sets", 0, type=int)
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
            end_dt   = datetime.strptime(end_str, "%Y-%m-%d").date()
        except Exception as e:
            return "Invalid date format. Please use YYYY-MM-DD."

        # Convert dates to strings for display.
        start_date_str = start_dt.strftime("%Y-%m-%d")
        end_date_str   = end_dt.strftime("%Y-%m-%d")

        session = SessionLocal()
        if game == "Powerball":
            draws = fetch_powerball_draws(start_dt, end_dt)
            if not draws or len(draws) < 5:
                # Fallback: use the last 20 draws
                draws = session.query(PowerballDraw).order_by(PowerballDraw.draw_date.desc()).limit(20).all()
                draws = list(reversed(draws))
            else:
                draws = sorted(draws, key=lambda d: d.draw_date)
            feats = feature_vector_powerball(draws)
            session.close()
            if model_powerball is None:
                return "No Powerball model loaded."
            with torch.no_grad():
                x_t = torch.from_numpy(feats).unsqueeze(0)
                logits = model_powerball(x_t)
                probs = torch.sigmoid(logits).numpy().flatten()
            # For Powerball: first 69 for white, next 26 for red.
            white_probs = probs[:69]
            red_probs = probs[69:]
        else:
            draws = fetch_megamillions_draws(start_dt, end_dt)
            if not draws or len(draws) < 5:
                draws = session.query(MegamillionsDraw).order_by(MegamillionsDraw.draw_date.desc()).limit(20).all()
                draws = list(reversed(draws))
            else:
                draws = sorted(draws, key=lambda d: d.draw_date)
            feats = feature_vector_megamillions(draws)
            session.close()
            if model_megamillions is None:
                return "No Mega Millions model loaded."
            with torch.no_grad():
                x_t = torch.from_numpy(feats).unsqueeze(0)
                logits = model_megamillions(x_t)
                probs = torch.sigmoid(logits).numpy().flatten()
            # For Mega Millions: first 70 for white, next 25 for red.
            white_probs = probs[:70]
            red_probs = probs[70:]
        
        # Use weighted random sampling to produce different sets.
        def generate_ai_sets(white_probs, red_probs, num_sets):
            white_probs = np.array(white_probs)
            red_probs = np.array(red_probs)
            # Normalize distributions
            if white_probs.sum() == 0:
                white_probs = np.ones_like(white_probs) / len(white_probs)
            else:
                white_probs = white_probs / white_probs.sum()
            if red_probs.sum() == 0:
                red_probs = np.ones_like(red_probs) / len(red_probs)
            else:
                red_probs = red_probs / red_probs.sum()
            sets = []
            for _ in range(num_sets):
                # Randomly choose 5 white balls (without replacement)
                white_set = np.random.choice(np.arange(1, len(white_probs)+1),
                                             size=5, replace=False, p=white_probs)
                white_set = sorted(white_set.tolist())
                red_ball = int(np.random.choice(np.arange(1, len(red_probs)+1), p=red_probs))
                sets.append({"whites": white_set, "red": red_ball})
            return sets
        
        ai_sets = generate_ai_sets(white_probs, red_probs, num_sets)
        reason = "The model generated these sets based on the weighted probability distribution computed from draws in your selected date range."
        
        return render_template("predict.html",
                               game=game,
                               start_date=start_date_str,
                               end_date=end_date_str,
                               ai_sets=ai_sets,
                               ai_reason=reason)
    else:
        # GET: Show the input form for AI Prediction with no default prediction results.
        return render_template("predict.html",
                               game="",
                               start_date="",
                               end_date="",
                               ai_sets=None,  # explicitly pass None so no results are shown
                               ai_reason="")



@app.route("/manual_combos", methods=["POST", "GET"])
def manual_combos():
    if request.method == "POST":
        game = request.form.get("game")
        start_str = request.form.get("start_date", "")
        end_str = request.form.get("end_date", "")
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()
        except:
            return "Invalid date format. Please use YYYY-MM-DD."
        
        # For frequency graph on combos page, get draws and compute frequencies.
        if game == "Powerball":
            draws = fetch_powerball_draws(start_dt, end_dt)
            wfreq, rfreq = frequency_powerball(draws)
            xw, yw = make_xy(wfreq)
            xr, yr = make_xy(rfreq)
        else:
            draws = fetch_megamillions_draws(start_dt, end_dt)
            wfreq, rfreq = frequency_megamillions(draws)
            xw, yw = make_xy(wfreq)
            xr, yr = make_xy(rfreq)
        
        white_str = request.form.get("white_list", "")
        red_str = request.form.get("red_list", "")
        import itertools
        combos = []
        try:
            w_nums = sorted(set(int(x) for x in white_str.split(",") if x.strip()))
            r_nums = sorted(set(int(x) for x in red_str.split(",") if x.strip()))
            for combo in itertools.combinations(w_nums, 5):
                for r in r_nums:
                    combos.append((combo, r))
        except Exception as e:
            return "Error parsing numbers: " + str(e)
        
        return render_template("combos.html",
                               game=game,
                               start_date=start_dt.strftime("%Y-%m-%d"),
                               end_date=end_dt.strftime("%Y-%m-%d"),
                               xw=xw, yw=yw,
                               xr=xr, yr=yr,
                               combos=combos)
    else:
        # GET: supply default empty frequency data and fields.
        return render_template("combos.html",
                               game="",
                               start_date="",
                               end_date="",
                               xw=[], yw=[], xr=[], yr=[],
                               combos=None)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
