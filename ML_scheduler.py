# ML_scheduler.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. DUELING DOUBLE DQN NETWORK WITH LSTM
# ==========================================

class DuelingLSTMDQN(nn.Module):
    def __init__(self, num_bands, seq_len=5, hidden_dim=64):
        super(DuelingLSTMDQN, self).__init__()
        self.num_bands = num_bands
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.input_dim = 3  # [scanned_band, hit_result, active_power_estimate]

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_bands)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]

        values = self.value_stream(last_step)
        advantages = self.advantage_stream(last_step)

        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


# ==========================================
# 2. PRIORITIZED EXPERIENCE REPLAY (PER)
# ==========================================

class PrioritizedReplayBuffer:
    def __init__(self, capacity=5000, alpha=0.6, beta_start=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(device)

        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device),
            indices,
            weights
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = max(prio, 1e-5)


# ==========================================
# 3. SMART RL SCHEDULER ENGINE
# ==========================================

class SmartScheduler:
    def __init__(self, num_bands=10, seq_len=5, lr=0.001, gamma=0.95):
        self.num_bands = num_bands
        self.seq_len = seq_len
        self.gamma = gamma

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.985
        self.switch_penalty = 0.25

        self.policy_net = DuelingLSTMDQN(num_bands, seq_len).to(device)
        self.target_net = DuelingLSTMDQN(num_bands, seq_len).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(capacity=5000)

    def _extract_state_sequence(self, history):
        seq = []
        for entry in history[-self.seq_len:]:
            band_norm = entry['band'] / self.num_bands
            hit = 1.0 if entry['result'] else 0.0
            pwr_estimate = entry.get('power', 0.0) / 100.0
            seq.append([band_norm, hit, pwr_estimate])

        while len(seq) < self.seq_len:
            seq.insert(0, [0.0, 0.0, 0.0])

        return np.array(seq, dtype=np.float32)

    def select_next_band(self, current_time, history):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_bands - 1)

        state_seq = self._extract_state_sequence(history)
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(device)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()

        return torch.argmax(q_values).item()

    def train_step(self, batch_size=32):
        if len(self.memory.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = (weights * (q_values - target_q_values) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_initial_model(self, env, warmup_steps=60):
        history = []
        prev_band = 0

        for t in range(min(warmup_steps, env.num_time_slots)):
            current_band = self.select_next_band(t, history)
            
            is_hit = env.grid[t, current_band]
            slot_pwr = env.get_slot_power(t)[current_band] if hasattr(env, 'get_slot_power') else 0.0

            base_reward = 2.0 if is_hit else -0.2
            switch_cost = self.switch_penalty if (current_band != prev_band) else 0.0
            reward = base_reward - switch_cost

            state = self._extract_state_sequence(history)
            history.append({'time': t, 'band': current_band, 'result': is_hit, 'power': slot_pwr})
            next_state = self._extract_state_sequence(history)

            done = 1.0 if t == warmup_steps - 1 else 0.0
            self.memory.add(state, current_band, reward, next_state, done)

            if len(self.memory.buffer) >= 16:
                self.train_step(batch_size=16)

            prev_band = current_band

        self.update_target_network()


        # Add to the bottom of ML_scheduler.py

class SequentialScanner:
    """Conventional full-spectrum sequential sweep baseline."""
    def __init__(self, num_bands):
        self.num_bands = num_bands
        self.current_band = 0

    def select_next_band(self, current_time, history=None):
        band = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        return band


class RandomScanner:
    """Uninformed pseudo-random scan baseline."""
    def __init__(self, num_bands, seed=42):
        self.num_bands = num_bands
        self.rng = np.random.default_rng(seed)

    def select_next_band(self, current_time, history=None):
        return self.rng.integers(0, self.num_bands)