"""
RL Position Sizer — Reinforcement Learning on Top of Ensemble Voting
====================================================================
Architecture:
  Ensemble (25 XGBoost) → Probabilities + Market State → DQN Agent → Sizing Decision
                                                                     ↓
                                                            (0%, 5%, 10%, 15%, 20%)

The DQN agent learns WHEN to size up/down based on:
  - Ensemble confidence (probability)
  - Cross-TF consensus (spread between TF groups)
  - Market regime (volatility, recent returns)
  - Current position state

Usage:
  python scripts/rl_position_sizer.py [--symbol BTCUSDT] [--episodes 500] [--threshold 0.60]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json, warnings, requests, random
from collections import deque
warnings.filterwarnings('ignore')
import xgboost as xgb

# --------------------  IMPORTS with fallbacks  --------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAIL = True
except ImportError:
    print("⚠️ PyTorch not installed. Using numpy-only fallback.")
    TORCH_AVAIL = False
    # Create dummy modules for type hints
    nn = type('nn', (), {'Module': object})()
    F = None
    optim = None

CACHE_DIR = Path("data/ml_cache")
TF_GROUPS = ['full', 'm15', 'm30', 'h1', 'h4']
SEEDS = [42, 101, 202, 303, 404]

# Trading params
COMMISSION = 0.0004
SLIPPAGE = 0.0005
INITIAL_CAPITAL = 5000.0

# RL params
STATE_DIM = 12
ACTION_DIM = 5  # 0:0%, 1:5%, 2:10%, 3:15%, 4:20%
ACTION_SIZES = [0.0, 0.05, 0.10, 0.15, 0.20]
ENTRY_THRESHOLD = 0.60
HOLD_BARS = 9
COOLDOWN = 3
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.997
BATCH_SIZE = 128
MEMORY_SIZE = 20000
LR = 0.001
TARGET_UPDATE = 100


# ===================== HELPER: Fetch data =====================
def fetch_ohlcv(symbol, start_time=None, end_time=None, days_back=140):
    """Fetch 5m OHLCV from Binance public API."""
    if end_time is None:
        end_time = datetime.now()
    if start_time is None:
        start_time = end_time - timedelta(days=days_back)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    all_bars, last_ts = [], start_ms
    while last_ts < end_ms:
        params = {'symbol': symbol, 'interval': '5m', 'limit': 1000,
                  'startTime': last_ts, 'endTime': end_ms}
        r = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=30)
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        all_bars.extend(batch)
        last_ts = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(all_bars, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    return df.set_index('open_time')[['open', 'high', 'low', 'close', 'volume']]


# ===================== ENSEMBLE PREDICTOR =====================
class EnsemblePredictor:
    """Loads and runs 25 XGBoost models (5 groups × 5 seeds) for one symbol."""

    def __init__(self, symbol):
        self.symbol = symbol
        self.models = {}  # {tf_group: [(seed, model, features), ...]}
        self._load()

    def _load(self):
        models_dir = Path("data/ml_models")
        # Try loading cached models first
        for tf in TF_GROUPS:
            tf_models = []
            if tf == 'full':
                pattern = f"{self.symbol}_xgb_ens_"
            else:
                pattern = f"{self.symbol}_{tf}_xgb_ens_"
            for seed in SEEDS:
                # Look for seed number in filename
                candidates = list(models_dir.glob(f"{pattern}*.json"))
                # Filter by seed if filename contains seed
                seed_candidates = [c for c in candidates if str(seed) in c.stem or str(seed) == str(seed)]
                # Also check the _seed<N> convention
                exact = [c for c in candidates if f"_seed{seed}" in c.stem or f"_{seed}." in c.stem]
                if exact:
                    candidates = exact
                if not candidates:
                    continue
                # Need to train on the fly — skip for now, will train in predictor
            self.models[tf] = tf_models

    def train_predict(self, feat_df, split_date, ohlcv_df, warmup=200):
        """Train models on train split, generate predictions + state features for test."""
        train = feat_df[feat_df.index < split_date].copy()
        test = feat_df[feat_df.index >= split_date].copy()

        feature_sets = {
            'full': [c for c in feat_df.columns if c != 'target']
        }
        for tf in ['m15', 'm30', 'h1', 'h4']:
            prefix = f"{tf}_"
            feature_sets[tf] = [c for c in feat_df.columns if c.startswith(prefix) and c != 'target']

        models_group = {}
        for tf in TF_GROUPS:
            features = feature_sets[tf]
            if len(features) < 10:
                continue
            y_train = train['target']
            valid = np.isfinite(train[features].fillna(0)).all(axis=1) & np.isfinite(y_train)
            train_clean = train.loc[valid]
            y_clean = y_train[valid]
            if len(train_clean) < 1000:
                continue
            tf_models = []
            for seed in SEEDS:
                random.seed(seed * 7)
                n_feat = max(15, int(len(features) * random.uniform(0.5, 0.75)))
                sampled = random.sample(list(features), n_feat)
                X = train_clean[sampled].fillna(0).clip(-10, 10)
                model = xgb.XGBClassifier(
                    n_estimators=120, max_depth=3, learning_rate=0.04,
                    subsample=0.6, colsample_bytree=0.6,
                    reg_alpha=1.0, reg_lambda=3.0, min_child_weight=7, gamma=1.0,
                    random_state=seed, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss', early_stopping_rounds=20
                )
                model.fit(X, y_clean,
                          eval_set=[(X[:min(3000, len(X)//5)], y_clean[:min(3000, len(y_clean)//5)])],
                          verbose=False)
                tf_models.append((str(seed), model, sampled))
            if tf_models:
                models_group[tf] = tf_models
        self.models = models_group

        # Predict on test set
        group_probs = {}
        for tf in TF_GROUPS:
            if tf not in models_group:
                continue
            all_probs = []
            for seed, model, sampled in models_group[tf]:
                avail = [c for c in sampled if c in test.columns]
                if len(avail) < 5:
                    continue
                X = test[avail].fillna(0).clip(-10, 10)
                all_probs.append(model.predict_proba(X)[:, 1])
            if all_probs:
                group_probs[tf] = np.mean(all_probs, axis=0)

        if len(group_probs) < 2:
            raise ValueError("Not enough TF groups for prediction")

        # Individual TF probabilities for state
        tf_probs_df = pd.DataFrame(group_probs, index=test.index)

        # Average probability (voting)
        avg_probs = tf_probs_df.mean(axis=1).values

        # Consensus measure: spread between max-min TF prob
        consensus = tf_probs_df.max(axis=1).values - tf_probs_df.min(axis=1).values

        # Align with OHLCV
        ohlcv_start = split_date - timedelta(hours=12)
        df_aligned = ohlcv_df[ohlcv_df.index >= ohlcv_start].copy()

        # Compute additional state features
        close_prices = df_aligned['close'].values
        returns_1h = np.full(len(df_aligned), np.nan)
        returns_4h = np.full(len(df_aligned), np.nan)
        atr = np.full(len(df_aligned), np.nan)

        for i in range(12, len(df_aligned)):
            returns_1h[i] = (close_prices[i] - close_prices[i-12]) / close_prices[i-12] * 100
        for i in range(48, len(df_aligned)):
            returns_4h[i] = (close_prices[i] - close_prices[i-48]) / close_prices[i-48] * 100
        for i in range(24, len(df_aligned)):
            atr[i] = np.mean([
                df_aligned.iloc[j]['high'] - df_aligned.iloc[j]['low']
                for j in range(i-24, i+1)
            ]) / close_prices[i] * 100  # ATR as % of price

        # Align probabilities with full OHLCV index
        prob_idx = tf_probs_df.index
        common_idx = df_aligned.index.intersection(prob_idx)
        avg_aligned = pd.Series(avg_probs, index=prob_idx).loc[common_idx].values
        consensus_aligned = pd.Series(consensus, index=prob_idx).loc[common_idx].values

        # Build state features DataFrame
        state_df = pd.DataFrame({
            'avg_proba': avg_aligned,
            'consensus': consensus_aligned,
        }, index=common_idx)

        # Get per-TF probabilities
        for tf in TF_GROUPS:
            if tf in group_probs:
                state_df[f'{tf}_proba'] = pd.Series(group_probs[tf], index=test.index).loc[common_idx].values

        state_df['ret_1h'] = pd.Series(returns_1h, index=df_aligned.index).loc[common_idx].values
        state_df['ret_4h'] = pd.Series(returns_4h, index=df_aligned.index).loc[common_idx].values
        state_df['atr_pct'] = pd.Series(atr, index=df_aligned.index).loc[common_idx].values
        state_df['signal_strength'] = np.abs(state_df['avg_proba'].values - 0.5)
        state_df['close'] = df_aligned['close'].loc[common_idx].values
        state_df['high'] = df_aligned['high'].loc[common_idx].values
        state_df['low'] = df_aligned['low'].loc[common_idx].values

        # Drop NaN rows, then warmup
        state_df = state_df.dropna().iloc[warmup:]
        return state_df


# ===================== TRADING ENVIRONMENT =====================
class TradingEnv:
    """
    Gym-like environment for RL position sizing.
    State: 12-dim vector
    Action: 5 discrete (position size 0-20%)
    Direction: determined by ensemble (prob > threshold = long, prob < 1-threshold = short)
    Reward: PnL when trade closes, 0 otherwise
    """

    def __init__(self, state_df, initial_capital=INITIAL_CAPITAL):
        self.data = state_df
        self.initial_capital = initial_capital
        self.reset()

    def reset(self):
        self.idx = 0
        self.capital = self.initial_capital
        self.position = 0  # -1, 0, 1
        self.hold_rem = 0
        self.cooldown = 0
        self.entry_price = 0.0
        self.entry_qty = 0.0
        self.trades = []
        self.equity_curve = [self.capital]
        self.equity_peak = self.capital
        self.max_steps = len(self.data) - 1
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.idx]
        s = np.array([
            float(row['avg_proba']),            # 0: ensemble prob
            float(row['signal_strength']),       # 1: |prob - 0.5|
            float(row['ret_1h']),                # 2: 1h return
            float(row['ret_4h']),                # 3: 4h return
            float(row['atr_pct']),               # 4: volatility
            self.position,                        # 5: current position
            self.hold_rem / HOLD_BARS,            # 6: hold remaining (normalized)
            self._current_pnl(),                  # 7: current trade PnL%
            float(row.get('m15_proba', 0.5)),     # 8: m15 prob
            float(row.get('h1_proba', 0.5)),      # 9: h1 prob
            float(row.get('h4_proba', 0.5)),      # 10: h4 prob
            float(row['consensus']),              # 11: cross-TF spread
        ], dtype=np.float32)
        return s

    def _current_pnl(self):
        if self.position == 0 or self.entry_price == 0:
            return 0.0
        price = float(self.data.iloc[self.idx]['close'])
        if self.position == 1:
            return (price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - price) / self.entry_price * 100

    def step(self, action):
        """action: 0-4 → position size 0-20%"""
        row = self.data.iloc[self.idx]
        price = float(row['close'])
        prob = float(row['avg_proba'])
        size_pct = ACTION_SIZES[action]

        reward = 0.0
        done = False
        trade_closed = False

        # --- Cooldown ---
        if self.cooldown > 0:
            self.cooldown -= 1

        # --- Exit if held long enough or RL says 0 ---
        if self.position != 0:
            self.hold_rem -= 1

            # Exit conditions: hold expired, OR action=0 (exit)
            if self.hold_rem <= 0 or action == 0:
                exit_px = price * (1 - SLIPPAGE) if self.position == 1 else price * (1 + SLIPPAGE)
                raw = (self.entry_qty * (exit_px - self.entry_price)) if self.position == 1 else \
                      (self.entry_qty * (self.entry_price - exit_px))
                comm = (self.entry_qty * self.entry_price + self.entry_qty * exit_px) * COMMISSION
                pnl = raw - comm
                self.capital += raw
                reward = pnl / max(1, (HOLD_BARS - self.hold_rem))  # reward per bar held
                self.trades.append({
                    'direction': 'long' if self.position == 1 else 'short',
                    'pnl': pnl,
                    'entry_price': self.entry_price,
                    'exit_price': exit_px,
                    'bars_held': HOLD_BARS - self.hold_rem,
                })
                self.position = 0
                self.cooldown = COOLDOWN
                trade_closed = True

        # --- Entry logic ---
        if self.position == 0 and self.cooldown <= 0:
            direction = 0
            if prob >= ENTRY_THRESHOLD:
                direction = 1
            elif prob <= (1 - ENTRY_THRESHOLD):
                direction = -1

            if direction != 0 and size_pct > 0.0:
                entry_px = price * (1 + SLIPPAGE) if direction == 1 else price * (1 - SLIPPAGE)
                self.entry_qty = (self.capital * size_pct) / entry_px
                self.position = direction
                self.hold_rem = HOLD_BARS
                self.entry_price = entry_px
            elif action > 0 and size_pct > 0.0:
                # RL wants to enter but no ensemble signal — reward penalty
                reward -= 0.01

        # --- Drawdown penalty ---
        self.equity_peak = max(self.equity_peak, self.capital)
        dd = (self.equity_peak - self.capital) / self.equity_peak
        if dd > 0.05:
            reward -= dd * 0.1  # small penalty for >5% drawdown

        # --- Advance step ---
        self.equity_curve.append(self.capital)
        self.idx += 1
        if self.idx >= self.max_steps:
            done = True

        next_state = self._get_state() if not done else np.zeros(STATE_DIM, dtype=np.float32)
        return next_state, reward, done, {'trade_closed': trade_closed}

    @property
    def total_return(self):
        return (self.capital - self.initial_capital) / self.initial_capital * 100


# ===================== DQN AGENT =====================
class DQN(nn.Module if TORCH_AVAIL else object):
    def __init__(self, state_dim, action_dim):
        if not TORCH_AVAIL:
            return
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.epsilon = EPS_START

        if TORCH_AVAIL:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.policy_net = DQN(state_dim, action_dim).to(self.device)
            self.target_net = DQN(state_dim, action_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        else:
            self.device = 'cpu'
            self.policy_net = None
            self.target_net = None

        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state, valid_actions=None):
        """ε-greedy. valid_actions: optional mask of allowed actions."""
        if np.random.random() < self.epsilon:
            if valid_actions is not None:
                valid = np.where(valid_actions)[0]
                return int(np.random.choice(valid)) if len(valid) > 0 else 0
            return int(np.random.randint(self.action_dim))

        if TORCH_AVAIL:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t).cpu().numpy()[0]
        else:
            # Random fallback without PyTorch
            if valid_actions is not None:
                valid = np.where(valid_actions)[0]
                return int(np.random.choice(valid)) if len(valid) > 0 else 0
            return int(np.random.randint(self.action_dim))

        if valid_actions is not None:
            q_values[~valid_actions] = -np.inf
        return int(np.argmax(q_values))

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        if not TORCH_AVAIL:
            return 0.0

        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(s).gather(1, a).squeeze()
        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0]
            target = r + GAMMA * next_q * (1 - d)

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def save(self, path):
        if TORCH_AVAIL:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
            }, path)

    def load(self, path):
        if TORCH_AVAIL:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']


# ===================== TRAIN & EVALUATE =====================
def train_rl(env, agent, episodes=300, eval_interval=20):
    """Train DQN agent on environment."""
    episode_rewards = []
    episode_returns = []
    eval_results = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        ep_steps = 0
        ep_trades = 0

        while True:
            # Valid actions: if no position and no signal, only action 0 is valid
            prob = state[0]
            has_signal = prob >= ENTRY_THRESHOLD or prob <= (1 - ENTRY_THRESHOLD)
            in_position = abs(state[5]) > 0.5  # state[5] = position

            if not in_position and not has_signal:
                valid = np.zeros(ACTION_DIM, dtype=bool)
                valid[0] = True  # Only skip
            elif not in_position and has_signal:
                valid = np.ones(ACTION_DIM, dtype=bool)  # All actions available
            else:
                # In position: can hold (any action) or exit (action 0)
                valid = np.ones(ACTION_DIM, dtype=bool)

            action = agent.select_action(state, valid)
            next_state, reward, done, info = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)
            total_reward += reward
            ep_steps += 1
            if info.get('trade_closed'):
                ep_trades += 1

            loss = agent.optimize()
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        ret = env.total_return
        episode_rewards.append(total_reward)
        episode_returns.append(ret)

        if (ep + 1) % eval_interval == 0:
            eval_res = evaluate_policy(env.__class__, env.data, agent)
            eval_results.append({
                'episode': ep + 1,
                'epsilon': agent.epsilon,
                'total_return': ret,
                'eval_return': eval_res.get('return', 0),
                'eval_trades': eval_res.get('trades', 0),
                'eval_wr': eval_res.get('win_rate', 0),
            })
            print(f"  ep={ep+1:4d} ε={agent.epsilon:.3f} "
                  f"ret={ret:+.2f}% eval_ret={eval_res.get('return', 0):+.2f}% "
                  f"trades={ep_trades}")

    return {'rewards': episode_rewards, 'returns': episode_returns, 'eval': eval_results}


def evaluate_policy(env_class, data, agent, deterministic=True):
    """Evaluate agent deterministically (ε=0)."""
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    env = env_class(data)
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        prob = state[0]
        has_signal = prob >= ENTRY_THRESHOLD or prob <= (1 - ENTRY_THRESHOLD)
        in_position = abs(state[5]) > 0.5
        if not in_position and not has_signal:
            valid = np.zeros(ACTION_DIM, dtype=bool)
            valid[0] = True
        else:
            valid = np.ones(ACTION_DIM, dtype=bool)
        action = agent.select_action(state, valid)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

    agent.epsilon = old_eps
    trades = env.trades
    if trades:
        wins = sum(1 for t in trades if t['pnl'] > 0)
        wr = wins / len(trades) * 100
    else:
        wr = 0
    return {
        'return': env.total_return,
        'reward': total_reward,
        'trades': len(trades),
        'win_rate': wr,
        'final_capital': env.capital,
    }


def baseline_evaluate(data, threshold=ENTRY_THRESHOLD):
    """Evaluate fixed threshold strategy (no RL) for comparison."""
    env = TradingEnv(data)
    state = env.reset()
    done = False
    while not done:
        prob = state[0]
        has_signal = prob >= threshold or prob <= (1 - threshold)
        in_position = abs(state[5]) > 0.5
        if has_signal and not in_position:
            action = 3  # Default 15%
        else:
            action = 0  # Skip/exit
        next_state, _, done, _ = env.step(action)
        state = next_state
    trades = env.trades
    wins = sum(1 for t in trades if t['pnl'] > 0)
    return {
        'return': env.total_return,
        'trades': len(trades),
        'win_rate': wins / len(trades) * 100 if trades else 0,
        'final_capital': env.capital,
    }


# ===================== MAIN =====================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--threshold', type=float, default=0.60)
    parser.add_argument('--no-train', action='store_true',
                        help='Skip training, just evaluate baseline')
    args = parser.parse_args()

    ENTRY_THRESHOLD = args.threshold
    print(f"\n{'='*65}")
    print(f"  RL POSITION SIZER — {args.symbol}")
    print(f"  Episodes: {args.episodes} | Threshold: {args.threshold}")
    print(f"  State Dim: {STATE_DIM} | Action Dim: {ACTION_DIM}")
    print(f"  PyTorch: {'✅' if TORCH_AVAIL else '⚠️ FALLBACK'}")
    print(f"{'='*65}")

    # 1. Fetch data & train ensemble
    print(f"\n[1/4] Fetching OHLCV...")
    ohlcv = fetch_ohlcv(args.symbol)
    print(f"  {len(ohlcv)} bars")

    print(f"\n[2/4] Loading features & training ensemble...")
    cache = list(CACHE_DIR.glob(f"{args.symbol}_5m_*.parquet"))
    if not cache:
        print(f"  ❌ No cache found for {args.symbol}")
        sys.exit(1)
    feat_df = pd.read_parquet(max(cache, key=lambda p: p.stat().st_mtime))
    split_date = feat_df.index[int(len(feat_df) * 0.80)]
    print(f"  Cache: {len(feat_df)} rows, split at {split_date.date()}")

    predictor = EnsemblePredictor(args.symbol)
    state_df = predictor.train_predict(feat_df, split_date, ohlcv)
    print(f"  State features: {len(state_df)} rows")

    # 3. Split into train/eval RL periods
    split_idx = int(len(state_df) * 0.75)
    train_data = state_df.iloc[:split_idx]
    eval_data = state_df.iloc[split_idx:]
    print(f"\n[3/4] RL: train={len(train_data)} eval={len(eval_data)}")

    # 4. Baseline (no RL)
    print(f"\n[4/4] Training RL agent...")
    print(f"\n  --- BASELINE (fixed 15% sizing) ---")
    base_train = baseline_evaluate(train_data)
    base_eval = baseline_evaluate(eval_data)
    print(f"  Train: ret={base_train['return']:+.2f}% | "
          f"WR {base_train['win_rate']:.1f}% | {base_train['trades']} trades")
    print(f"  Eval:  ret={base_eval['return']:+.2f}% | "
          f"WR {base_eval['win_rate']:.1f}% | {base_eval['trades']} trades")

    # 5. Train RL
    if not args.no_train and TORCH_AVAIL:
        print(f"\n  --- RL DQN TRAINING ---")
        env = TradingEnv(train_data)
        agent = DQNAgent(STATE_DIM, ACTION_DIM)
        history = train_rl(env, agent, episodes=args.episodes, eval_interval=20)

        # Evaluate RL on eval set
        print(f"\n  --- RL EVALUATION (OOS) ---")
        eval_result = evaluate_policy(TradingEnv, eval_data, agent)
        print(f"  Eval: ret={eval_result['return']:+.2f}% | "
              f"WR {eval_result['win_rate']:.1f}% | {eval_result['trades']} trades")

        # Compare with baseline
        print(f"\n  {'='*55}")
        print(f"  COMPARISON: RL vs Baseline (OOS)")
        print(f"{'='*55}")
        print(f"  {'Metric':<20} {'Baseline':<15} {'RL':<15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15}")
        print(f"  {'Return %':<20} {base_eval['return']:<+15.2f} {eval_result['return']:<+15.2f}")
        print(f"  {'Win Rate %':<20} {base_eval['win_rate']:<15.1f} {eval_result['win_rate']:<15.1f}")
        print(f"  {'Trades':<20} {base_eval['trades']:<15d} {eval_result['trades']:<15d}")
        improvement = eval_result['return'] - base_eval['return']
        print(f"  {'Improvement':<20} {'':<15} {improvement:<+15.2f}%")
        print(f"  {'Status':<20} {'':<15} {'✅' if improvement > 0 else '⚠️'}")

        # Save model
        model_path = f"data/rl_models/{args.symbol}_dqn.pt"
        Path("data/rl_models").mkdir(exist_ok=True)
        agent.save(model_path)
        print(f"\n  Model saved: {model_path}")
    elif not TORCH_AVAIL:
        print(f"\n  ⚠️ PyTorch not available — RL training skipped.")
        print(f"  Install with: pip install torch")
    else:
        print(f"\n  ⚠️ --no-train flag set. Skipping RL training.")
