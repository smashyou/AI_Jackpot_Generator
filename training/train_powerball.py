# training/train_powerball.py
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from db.models import SessionLocal, PowerballDraw

class PowerballMLP(nn.Module):
    def __init__(self, input_dim=95, hidden_dim=128, output_dim=95):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc_out(x)

def load_powerball_data():
    session = SessionLocal()
    draws = (session.query(PowerballDraw)
             .order_by(PowerballDraw.draw_date)
             .all())
    data = []
    for d in draws:
        date_ = d.draw_date
        whites = [int(x) for x in d.white_balls.split()]
        pb = int(d.powerball)
        data.append((date_, whites, pb))
    session.close()
    return data

def label_vector(whites, pb):
    # 69 white + 26 red = 95 total
    v = np.zeros(95, dtype=np.float32)
    for w in whites:
        v[w-1] = 1.0
    v[69+(pb-1)] = 1.0
    return v

def feature_vector(past_draws):
    # rolling frequency approach
    white_c = np.zeros(69, dtype=np.float32)
    red_c = np.zeros(26, dtype=np.float32)
    for (_, ws, pb) in past_draws:
        for w in ws:
            white_c[w-1] += 1
        red_c[pb-1] += 1
    total = len(past_draws)
    if total > 0:
        white_c /= total
        red_c /= total
    return np.concatenate([white_c, red_c])  # shape (95,)

def build_dataset(data, window=20):
    # data is sorted ascending
    X_list = []
    Y_list = []
    for i in range(window, len(data)):
        past_segment = data[i-window:i]
        feats = feature_vector(past_segment)
        (_, ws, pb) = data[i]
        lbl = label_vector(ws, pb)
        X_list.append(feats)
        Y_list.append(lbl)
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y

def train_powerball(X, Y, epochs=50, lr=1e-3, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PowerballMLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    train_size = X.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(train_size)
        X_t = X_t[perm]
        Y_t = Y_t[perm]
        num_batches = (train_size + batch_size - 1) // batch_size
        total_loss = 0
        for b in range(num_batches):
            start = b*batch_size
            end = start+batch_size
            bx = X_t[start:end].to(device)
            by = Y_t[start:end].to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")
    return model

def main():
    data = load_powerball_data()
    if len(data) < 30:
        print("Not enough draws to train.")
        return
    X, Y = build_dataset(data, window=20)
    if X.shape[0] < 2:
        print("Not enough data after building dataset.")
        return
    # Simple split
    split_idx = int(X.shape[0]*0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    model = train_powerball(X_train, Y_train, epochs=50)
    # saving
    torch.save(model.state_dict(), "powerball_model.pt")
    print("Powerball model saved -> powerball_model.pt")

if __name__ == "__main__":
    main()
