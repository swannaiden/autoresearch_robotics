"""
Fixed RL infrastructure for autoresearch experiments.
Provides environment creation, evaluation, and rollout utilities.

This file is READ-ONLY — the agent must not modify it.

Usage:
    # From train.py:
    from prepare import (
        TIME_BUDGET, EVAL_EPISODES, ENV_NAME, SEED,
        make_env, make_vec_env, evaluate_return, RolloutBuffer,
    )
"""

import numpy as np
import gymnasium as gym

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
EVAL_EPISODES = 100        # number of episodes for evaluation
ENV_NAME = "LunarLander-v3"  # benchmark environment
SEED = 42

# ---------------------------------------------------------------------------
# Environment utilities
# ---------------------------------------------------------------------------

def make_env(env_name=ENV_NAME, seed=SEED):
    """Create a single gym environment."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def make_vec_env(env_name=ENV_NAME, num_envs=8, seed=SEED):
    """Create vectorized parallel environments."""
    def _make(i):
        def _init():
            env = gym.make(env_name)
            env.reset(seed=seed + i)
            return env
        return _init
    return gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_return(policy_fn, env_name=ENV_NAME, num_episodes=EVAL_EPISODES, seed=1000):
    """
    Evaluate a policy by running num_episodes episodes and returning the
    average total return. Uses fixed seeds for reproducibility.

    Args:
        policy_fn: callable(obs_np_array) -> action (int or np array).
                   Receives a single observation (not batched).
        env_name: gymnasium environment name.
        num_episodes: number of evaluation episodes.
        seed: base seed for evaluation episodes.

    Returns:
        float: average episode return (higher is better).
    """
    env = gym.make(env_name)
    total_returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_return = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        total_returns.append(episode_return)

    env.close()
    return float(np.mean(total_returns))

# ---------------------------------------------------------------------------
# Rollout buffer for PPO
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores rollout data from vectorized environments and computes GAE.

    Usage:
        buf = RolloutBuffer(num_steps, num_envs, obs_shape, act_shape, discrete=True)
        for step in range(num_steps):
            buf.store(step, obs, actions, log_probs, values, rewards, dones)
        buf.compute_advantages(last_values, last_dones, gamma, gae_lambda)
        # Then iterate over minibatches:
        for batch in buf.get_minibatches(num_minibatches):
            obs, actions, old_log_probs, advantages, returns = batch
    """

    def __init__(self, num_steps, num_envs, obs_shape, act_shape=(), discrete=True):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.discrete = discrete
        self.obs = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.float32)
        if discrete:
            self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        else:
            self.actions = np.zeros((num_steps, num_envs, *act_shape), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)

    def store(self, step, obs, actions, log_probs, values, rewards, dones):
        self.obs[step] = obs
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.values[step] = values
        self.rewards[step] = rewards
        self.dones[step] = dones

    def compute_advantages(self, last_values, last_dones, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and returns."""
        last_gae = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        self.returns = self.advantages + self.values

    def get_minibatches(self, num_minibatches, rng=None):
        """Yield flattened minibatches for PPO updates."""
        batch_size = self.num_steps * self.num_envs
        assert batch_size % num_minibatches == 0
        minibatch_size = batch_size // num_minibatches

        # Flatten all arrays: (num_steps, num_envs, ...) -> (batch_size, ...)
        flat_obs = self.obs.reshape(batch_size, *self.obs.shape[2:])
        flat_actions = self.actions.reshape(batch_size, *self.actions.shape[2:])
        flat_log_probs = self.log_probs.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)
        flat_values = self.values.reshape(batch_size)

        if rng is None:
            rng = np.random.default_rng()
        indices = rng.permutation(batch_size)

        for start in range(0, batch_size, minibatch_size):
            idx = indices[start:start + minibatch_size]
            yield (
                flat_obs[idx],
                flat_actions[idx],
                flat_log_probs[idx],
                flat_advantages[idx],
                flat_returns[idx],
                flat_values[idx],
            )

# ---------------------------------------------------------------------------
# Main (just prints environment info)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = make_env()
    print(f"Environment: {ENV_NAME}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Eval episodes: {EVAL_EPISODES}")

    # Quick sanity check: run one random episode
    obs, _ = env.reset(seed=SEED)
    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
    env.close()
    print(f"Random policy: {total_reward:.1f} return in {steps} steps")
    print("\nReady to train.")
