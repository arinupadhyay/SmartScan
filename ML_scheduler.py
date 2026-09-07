# ML_scheduler.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class PatternPredictor(nn.Module):
    def __init__(self, num_bands, seq_len=3):
        super(PatternPredictor, self).__init__()
        self.num_bands = num_bands
        self.seq_len = seq_len
        
        self.net = nn.Sequential(
            nn.Linear(num_bands * seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_bands)
        )

    def forward(self, x):
        return self.net(x)


class SmartScheduler:
    def __init__(self, num_bands, seq_len=3):
        self.num_bands = num_bands
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PatternPredictor(num_bands, seq_len).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.is_trained = False

    def train_initial_model(self, env):
        """Train directly on the environment sequence pattern so it accurately tracks hops."""
        X_train, Y_train = [], []
        
        # Build sequence datasets from RFEnvironment grid
        for t in range(self.seq_len, env.num_time_slots - 1):
            # Input features: flatten past 'seq_len' slots of signal presence
            seq = env.grid[t - self.seq_len : t, :].flatten().astype(np.float32)
            # Target: active band in the NEXT slot (t)
            target = env.grid[t, :]
            
            X_train.append(seq)
            Y_train.append(target)

        if len(X_train) == 0:
            return

        X_t = torch.FloatTensor(np.array(X_train)).to(self.device)
        Y_t = torch.FloatTensor(np.array(Y_train)).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        
        # Fast supervised fit on jammer sequence dynamics
        for epoch in range(250):
            self.optimizer.zero_grad()
            out = self.model(X_t)
            loss = criterion(out, Y_t)
            loss.backward()
            self.optimizer.step()

        self.is_trained = True

    def select_next_band(self, current_time, env_grid):
        """Selects band based on observed environment pattern sequence up to current time."""
        if not self.is_trained or current_time < self.seq_len:
            return random.randint(0, self.num_bands - 1)

        # Get actual sequence up to time 'current_time'
        recent_seq = env_grid[current_time - self.seq_len : current_time, :].flatten().astype(np.float32)
        state_tensor = torch.FloatTensor([recent_seq]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(state_tensor)
            predicted_band = logits.argmax(dim=1).item()

        return predicted_band

    def update_online(self, prev_history, band, hit, power, current_history):
        pass


class SequentialScanner:
    def __init__(self, num_bands):
        self.num_bands = num_bands
        self.current_band = 0

    def select_next_band(self, current_time):
        band = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        return band


class RandomScanner:
    def __init__(self, num_bands, seed=42):
        self.num_bands = num_bands
        self.rng = np.random.RandomState(seed)

    def select_next_band(self, current_time):
        return int(self.rng.randint(0, self.num_bands))