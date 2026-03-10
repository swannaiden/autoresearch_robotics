"""
Autoresearch PPO training script. Single-file.
Usage: uv run train.py
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from prepare import (
    TIME_BUDGET, EVAL_EPISODES, ENV_NAME, SEED,
    make_vec_env, evaluate_return, RolloutBuffer,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Network architecture
HIDDEN_SIZE = 256           # hidden layer width
NUM_LAYERS = 3              # number of hidden layers
ACTIVATION = "leaky_relu"   # activation function: "tanh", "relu", or "leaky_relu"
SEPARATE_NETWORKS = True    # separate actor and critic networks

# PPO
NUM_ENVS = 16              # number of parallel environments
NUM_STEPS = 256             # rollout steps per env before each update
NUM_MINIBATCHES = 8         # number of minibatches per update
UPDATE_EPOCHS = 10          # number of passes over rollout data per update
GAMMA = 0.999               # discount factor
GAE_LAMBDA = 0.98           # GAE lambda
CLIP_EPS = 0.2              # PPO clipping epsilon
ENT_COEF = 0.005            # entropy bonus coefficient
VF_COEF = 0.5               # value loss coefficient
MAX_GRAD_NORM = 0.5         # max gradient norm for clipping
LEARNING_RATE = 5e-4        # learning rate
ANNEAL_LR = True            # whether to linearly anneal LR to 0

# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

def make_activation(name):
    if name == "tanh":
        return nn.Tanh
    elif name == "relu":
        return nn.ReLU
    elif name == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise ValueError(f"Unknown activation: {name}")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=2, activation="tanh",
                 separate=False):
        super().__init__()
        Act = make_activation(activation)

        def _build_net(in_dim, hidden_size, num_layers):
            layers = []
            d = in_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(d, hidden_size))
                layers.append(nn.LayerNorm(hidden_size))
                layers.append(Act())
                d = hidden_size
            return nn.Sequential(*layers)

        if separate:
            self.actor_net = _build_net(obs_dim, hidden_size, num_layers)
            self.critic_net = _build_net(obs_dim, hidden_size, num_layers)
            self.feature_net = None
        else:
            self.feature_net = _build_net(obs_dim, hidden_size, num_layers)
            self.actor_net = None
            self.critic_net = None

        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for net in [self.feature_net, self.actor_net, self.critic_net]:
            if net is not None:
                for module in net:
                    if isinstance(module, nn.Linear):
                        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                        nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        if self.feature_net is not None:
            features = self.feature_net(x)
            logits = self.policy_head(features)
            value = self.value_head(features).squeeze(-1)
        else:
            logits = self.policy_head(self.actor_net(x))
            value = self.value_head(self.critic_net(x)).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_deterministic_action(self, obs):
        """For evaluation: pick the greedy action."""
        logits, _ = self.forward(obs)
        return logits.argmax(dim=-1)


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
obs_dim = envs.single_observation_space.shape[0]
act_dim = envs.single_action_space.n
print(f"Environment: {ENV_NAME}")
print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

# Create model and optimizer
model = ActorCritic(obs_dim, act_dim, HIDDEN_SIZE, NUM_LAYERS, ACTIVATION,
                    separate=SEPARATE_NETWORKS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

# Rollout buffer
batch_size = NUM_ENVS * NUM_STEPS
assert batch_size % NUM_MINIBATCHES == 0
buffer = RolloutBuffer(
    NUM_STEPS, NUM_ENVS,
    obs_shape=(obs_dim,),
    discrete=True,
)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Batch size: {batch_size} ({NUM_ENVS} envs x {NUM_STEPS} steps)")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

obs, _ = envs.reset(seed=SEED)
total_training_time = 0.0
update = 0
total_timesteps = 0
episode_returns = []  # track completed episode returns during training

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

            next_obs, reward, terminated, truncated, infos = envs.step(action_np)
            done = np.logical_or(terminated, truncated)

            # Track episode returns
            running_returns += reward
            for i in range(NUM_ENVS):
                if done[i]:
                    episode_returns.append(running_returns[i])
                    running_returns[i] = 0.0

            buffer.store(step, obs, action_np, log_prob_np, value_np, reward, done)
            obs = next_obs

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
            mb_actions_t = torch.as_tensor(mb_actions, dtype=torch.long, device=device)
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

            # Value loss (clipped)
            vf_loss = 0.5 * ((new_value - mb_returns_t) ** 2).mean()

            # Entropy loss
            ent_loss = entropy.mean()

            loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

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
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()

def policy_fn(obs):
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = model.get_deterministic_action(obs_t)
    return action.item()

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
