# autoresearch

This is an experiment to have the LLM do its own RL research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, environment utilities, evaluation, rollout buffer. Do not modify.
   - `train.py` — the file you modify. Actor-critic network, PPO algorithm, training loop.
4. **Verify environment works**: Run `uv run python -c "import gymnasium; env = gymnasium.make('LunarLander-v3'); print('OK')"` to check that the environment is available.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single machine (GPU optional — PPO on LunarLander runs fine on CPU). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: network architecture (width, depth, activation functions, separate vs shared backbone), PPO hyperparameters (clipping, entropy, learning rate, batch sizes, discount), training loop structure, observation normalization, reward scaling, exploration strategies, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, environment creation, rollout buffer, and training constants (time budget, eval episodes, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_return` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest avg_return.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the network size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
avg_return:       215.432100
training_seconds: 300.1
total_seconds:    310.5
peak_vram_mb:     128.3
total_timesteps:  1228800
num_updates:      300
num_params:       5124
episodes_seen:    2400
```

You can extract the key metric from the log file:

```
grep "^avg_return:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	avg_return	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. avg_return achieved (e.g. 215.432100) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 0.1 — divide peak_vram_mb by 1024) — use 0.0 for crashes (or CPU runs)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	avg_return	memory_gb	status	description
a1b2c3d	125.300000	0.0	keep	baseline
b2c3d4e	185.400000	0.0	keep	increase hidden size to 128
c3d4e5f	110.200000	0.0	discard	switch to relu activation
d4e5f6g	0.000000	0.0	crash	invalid batch size config
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^avg_return:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If avg_return improved (higher), you "advance" the branch, keeping the git commit
9. If avg_return is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

Some ideas to explore (non-exhaustive):
- Network architecture: width, depth, separate actor/critic networks, skip connections
- Activation functions: tanh, relu, leaky relu, elu, gelu
- Observation normalization / reward scaling
- PPO hyperparameters: clip epsilon, entropy coefficient, number of epochs, minibatch size
- Learning rate schedules beyond linear annealing
- Discount factor and GAE lambda tuning
- Larger/smaller rollout buffers (NUM_STEPS)
- More/fewer parallel environments
- Value function clipping
- Gradient accumulation strategies
