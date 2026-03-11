"""
Autoresearch PPO training script for CarRacing-v3. Single-file.
CNN policy with continuous actions.
Usage: uv run train.py
"""

import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from prepare import (
    TIME_BUDGET, EVAL_EPISODES, ENV_NAME, SEED,
    make_vec_env, evaluate_return, RolloutBuffer,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Network architecture
HIDDEN_SIZE = 256           # FC layer width after CNN
ACTIVATION = "relu"         # activation for CNN layers

# PPO
NUM_ENVS = 8               # number of parallel environments (images are heavier)
NUM_STEPS = 128             # rollout steps per env before each update
NUM_MINIBATCHES = 4         # number of minibatches per update
UPDATE_EPOCHS = 4           # number of passes over rollout data per update
GAMMA = 0.99                # discount factor
GAE_LAMBDA = 0.95           # GAE lambda
CLIP_EPS = 0.2              # PPO clipping epsilon
ENT_COEF = 0.01             # entropy bonus coefficient
VF_COEF = 0.5               # value loss coefficient
MAX_GRAD_NORM = 0.5         # max gradient norm for clipping
LEARNING_RATE = 3e-4        # learning rate
ANNEAL_LR = True            # whether to linearly anneal LR to 0

# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------

def preprocess_obs(obs):
    """Convert (B, 96, 96, 3) uint8 -> (B, 3, 96, 96) float32 in [0,1]."""
    if isinstance(obs, np.ndarray):
        # Transpose HWC -> CHW and normalize
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))
    return obs


# ---------------------------------------------------------------------------
# Actor-Critic Network (CNN + continuous actions)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, act_dim, hidden_size=256):
        super().__init__()

        # CNN encoder: (3, 96, 96) -> features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # -> (32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 8, 8)
            nn.ReLU(),
            nn.Flatten(),  # -> 64*8*8 = 4096
        )

        # Compute CNN output size
        cnn_out_size = 64 * 8 * 8  # 4096

        # Shared feature layer
        self.feature_fc = nn.Sequential(
            nn.Linear(cnn_out_size, hidden_size),
            nn.ReLU(),
        )

        # Policy head: outputs mean for each action dimension
        self.policy_mean = nn.Linear(hidden_size, act_dim)
        # Learnable log standard deviation
        self.policy_log_std = nn.Parameter(torch.zeros(act_dim))

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for CNN
        for module in self.cnn:
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        # Feature FC
        for module in self.feature_fc:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        # Policy head (small init for exploration)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)

        # Value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _encode(self, x):
        """Shared CNN + FC encoding."""
        cnn_features = self.cnn(x)
        return self.feature_fc(cnn_features)

    def forward(self, x):
        features = self._encode(x)
        action_mean = self.policy_mean(features)
        value = self.value_head(features).squeeze(-1)
        return action_mean, value

    def get_action_and_value(self, obs, action=None):
        action_mean, value = self.forward(obs)
        action_std = self.policy_log_std.exp()
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)  # sum across action dims
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value

    def get_deterministic_action(self, obs):
        """For evaluation: use the mean action."""
        action_mean, _ = self.forward(obs)
        return action_mean


def postprocess_action(action_np):
    """Clamp actions to valid CarRacing ranges.
    Action space: [steering, gas, brake]
    steering: [-1, 1], gas: [0, 1], brake: [0, 1]
    """
    action_np = np.clip(action_np, -1.0, 1.0)
    # Gas and brake should be [0, 1]
    action_np[..., 1] = np.clip(action_np[..., 1], 0.0, 1.0)
    action_np[..., 2] = np.clip(action_np[..., 2], 0.0, 1.0)
    return action_np


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create environments
envs = make_vec_env(ENV_NAME, NUM_ENVS, SEED)
act_dim = envs.single_action_space.shape[0]  # 3 for CarRacing
print(f"Environment: {ENV_NAME}")
print(f"Obs shape: {envs.single_observation_space.shape}, Act dim: {act_dim}")

# Create model and optimizer
model = ActorCritic(act_dim, HIDDEN_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

# Rollout buffer — obs stored as preprocessed (3, 96, 96), continuous actions (3,)
batch_size = NUM_ENVS * NUM_STEPS
assert batch_size % NUM_MINIBATCHES == 0
buffer = RolloutBuffer(
    NUM_STEPS, NUM_ENVS,
    obs_shape=(3, 96, 96),
    act_shape=(act_dim,),
    discrete=False,
)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Batch size: {batch_size} ({NUM_ENVS} envs x {NUM_STEPS} steps)")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

ema_model = copy.deepcopy(model)
ema_decay = 0.999

obs_raw, _ = envs.reset(seed=SEED)
obs = preprocess_obs(obs_raw)  # (NUM_ENVS, 3, 96, 96)

total_training_time = 0.0
update = 0
total_timesteps = 0
episode_returns = []

# For tracking episode returns from vectorized envs
running_returns = np.zeros(NUM_ENVS, dtype=np.float64)

while True:
    t0 = time.time()

    # LR annealing
    if ANNEAL_LR:
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lr = LEARNING_RATE * (1.0 - progress)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    # --- Rollout phase ---
    model.eval()
    with torch.no_grad():
        for step in range(NUM_STEPS):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, log_prob, _, value = model.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()

            # Post-process actions for environment
            action_env = postprocess_action(action_np)

            next_obs_raw, reward, terminated, truncated, infos = envs.step(action_env)
            done = np.logical_or(terminated, truncated)

            # Track episode returns
            running_returns += reward
            for i in range(NUM_ENVS):
                if done[i]:
                    episode_returns.append(running_returns[i])
                    running_returns[i] = 0.0

            # Store preprocessed obs and raw (unclamped) actions for PPO
            buffer.store(step, obs, action_np, log_prob_np, value_np, reward, done)
            obs = preprocess_obs(next_obs_raw)

        # Bootstrap value for GAE
        last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        _, last_values = model(last_obs_tensor)
        last_values = last_values.cpu().numpy()
        last_dones = done.astype(np.float32)

    buffer.compute_advantages(last_values, last_dones, GAMMA, GAE_LAMBDA)

    # --- PPO update phase ---
    model.train()
    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_ent_loss = 0.0
    num_updates_this_round = 0

    for epoch in range(UPDATE_EPOCHS):
        for batch in buffer.get_minibatches(NUM_MINIBATCHES, rng=rng):
            mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns, mb_old_values = batch

            mb_obs_t = torch.as_tensor(mb_obs, dtype=torch.float32, device=device)
            mb_actions_t = torch.as_tensor(mb_actions, dtype=torch.float32, device=device)
            mb_old_log_probs_t = torch.as_tensor(mb_old_log_probs, dtype=torch.float32, device=device)
            mb_advantages_t = torch.as_tensor(mb_advantages, dtype=torch.float32, device=device)
            mb_returns_t = torch.as_tensor(mb_returns, dtype=torch.float32, device=device)

            # Normalize advantages
            mb_advantages_t = (mb_advantages_t - mb_advantages_t.mean()) / (mb_advantages_t.std() + 1e-8)

            _, new_log_prob, entropy, new_value = model.get_action_and_value(mb_obs_t, mb_actions_t)

            # Policy loss (clipped surrogate)
            log_ratio = new_log_prob - mb_old_log_probs_t
            ratio = log_ratio.exp()
            pg_loss1 = -mb_advantages_t * ratio
            pg_loss2 = -mb_advantages_t * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            vf_loss = 0.5 * ((new_value - mb_returns_t) ** 2).mean()

            # Entropy loss
            ent_loss = entropy.mean()

            loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            # Update EMA
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_ent_loss += ent_loss.item()
            num_updates_this_round += 1

    update += 1
    total_timesteps += batch_size

    t1 = time.time()
    dt = t1 - t0

    # Only count training time after first update (skip JIT/warmup overhead)
    if update > 1:
        total_training_time += dt

    # Logging
    avg_pg = total_pg_loss / max(num_updates_this_round, 1)
    avg_vf = total_vf_loss / max(num_updates_this_round, 1)
    avg_ent = total_ent_loss / max(num_updates_this_round, 1)
    pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
    remaining = max(0, TIME_BUDGET - total_training_time)

    recent_returns = episode_returns[-50:] if episode_returns else [0.0]
    mean_train_return = np.mean(recent_returns)

    print(
        f"\rupdate {update:04d} ({pct_done:.1f}%) | "
        f"train_return: {mean_train_return:.1f} | "
        f"pg: {avg_pg:.4f} | vf: {avg_vf:.4f} | ent: {avg_ent:.4f} | "
        f"dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    # Time's up
    if update > 1 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log
envs.close()

# ---------------------------------------------------------------------------
# Final evaluation (using EMA model)
# ---------------------------------------------------------------------------

ema_model.eval()

def policy_fn(obs):
    with torch.no_grad():
        # Single obs: (96, 96, 3) -> (1, 3, 96, 96)
        obs_p = obs.astype(np.float32) / 255.0
        obs_p = np.transpose(obs_p, (2, 0, 1))
        obs_t = torch.as_tensor(obs_p, dtype=torch.float32, device=device).unsqueeze(0)
        action = ema_model.get_deterministic_action(obs_t)
    action_np = action.squeeze(0).cpu().numpy()
    return postprocess_action(action_np)

print("Evaluating...")
avg_return = evaluate_return(policy_fn, ENV_NAME, EVAL_EPISODES)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0

print("---")
print(f"avg_return:       {avg_return:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_timesteps:  {total_timesteps}")
print(f"num_updates:      {update}")
print(f"num_params:       {num_params}")
print(f"episodes_seen:    {len(episode_returns)}")
