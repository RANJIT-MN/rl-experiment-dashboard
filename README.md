# Real-Time Reinforcement Learning Experiment Tracker & Dynamic Controller Dashboard

## Project

A real-time reinforcement learning experiment tracking and dynamic controller dashboard using MuJoCo simulation.

## Technology Stack

### Operating System
- Windows

### Frontend
- React
- Vite
- JavaScript
- CSS

### Backend
- Python
- FastAPI
- SQLAlchemy
- SQLite
- Pydantic

### Reinforcement Learning
- Gymnasium
- Stable-Baselines3
- PPO

### Simulation
- MuJoCo

## Validated Environment

The initial MuJoCo validation environment is:

`InvertedPendulum-v5`

## Foundation Validation

The following components have been validated:

- Python environment: PASS
- MuJoCo installation: PASS
- Gymnasium installation: PASS
- Stable-Baselines3 installation: PASS
- PyTorch installation: PASS
- InvertedPendulum-v5 creation: PASS
- Environment reset: PASS
- Random action execution: PASS
- Observation verification: PASS
- Reward verification: PASS
- MuJoCo rendering: PASS
- PPO initialization: PASS
- Short PPO training: PASS
- PPO model saving: PASS
- PPO model loading: PASS
- Deterministic evaluation: PASS

## Initial PPO Validation

Evaluation results from the short validation run:

- Episode 1: 98.00
- Episode 2: 96.00
- Episode 3: 97.00
- Average reward: 97.00

## Current Status

Foundation validation completed successfully. The FastAPI backend and React/Vite dashboard support experiment creation, cooperative training start/stop controls, persistent PPO metrics, and per-experiment live monitoring.

## Training Metrics and Live Monitoring

PPO training uses Gymnasium episode information from an SB3 `Monitor` wrapper. Completed-episode rewards and lengths are recorded when episodes finish; periodic records capture observed timesteps, elapsed training time, speed, reward averages when available, and status. No episode reward or length is reported before an episode completes.

Metric rows are stored in the configured SQLite database in the additive `training_metrics` table. The existing experiment records are preserved during database initialization.

- `GET /experiments/{experiment_id}/metrics` returns bounded historical metric data (default limit: 1,000; maximum: 10,000), ordered by timestep and timestamp. Optional `from_timestep` and `to_timestep` query parameters filter the range.
- `WS /experiments/{experiment_id}/metrics/ws` streams status and metric updates for that experiment. Per-client queues are bounded; use the history endpoint to recover after reconnects or dropped live updates.
- `RL_METRICS_LOG_INTERVAL` sets the periodic checkpoint interval in timesteps (default: `1000`). Completed episodes are recorded immediately, independent of this interval.

The dashboard uses responsive inline SVG charts and adds no charting dependency.

## PPO Configuration and Reproducibility

New experiments save the PPO configuration alongside the experiment:

| Parameter | Default | Validation |
| --- | ---: | --- |
| `learning_rate` | `0.0003` | Finite and greater than zero |
| `n_steps` | `2048` | Integer greater than one |
| `batch_size` | `64` | Integer greater than one; must evenly divide `n_steps` |
| `gamma` | `0.99` | Finite, greater than zero, and at most one |
| `n_epochs` | `10` | Positive integer |

These defaults match the installed Stable-Baselines3 2.9.0 PPO signature. This project uses a single environment, so the rollout buffer contains `n_steps * 1` samples. SB3 permits a non-dividing batch size but warns that the last minibatch is truncated; the API rejects such combinations to avoid that behavior. SB3's default advantage normalization also requires a batch size greater than one and a rollout buffer greater than one.

SQLite schema initialization adds the five nullable configuration columns without rebuilding the `experiments` table. Existing experiments retain their records and metric history; their new configuration columns remain `NULL` because the historical settings were not stored. The dashboard labels those values "Not recorded (legacy)". Such legacy experiments can still be listed and monitored, but cannot be restarted until a new experiment is created with a recorded configuration.

SB3 collects a complete rollout before checking whether the requested total timestep count has been reached. With one environment, training therefore advances in multiples of `n_steps` and can exceed the requested total by up to `n_steps - 1` timesteps. For example, the default `total_timesteps=1000` and `n_steps=2048` results in one 2048-step rollout. The application preserves SB3's existing `learn(total_timesteps=...)` behavior rather than changing the training loop.

## Experiment Comparison and Export

The dashboard's comparison section lets users select 2–10 distinct experiments and compare their persisted configuration and performance.

- `POST /experiments/compare` accepts `{"experiment_ids": [1, 2]}` and returns selected experiment details, completed-episode summaries, and reward/length curve points. Missing IDs return `404`; duplicate, non-positive, or out-of-range selections are rejected by request validation.
- Mean reward and mean episode length are calculated over all persisted completed-episode samples for each experiment, across its recorded history. The completed episode count is the number of persisted reward samples, not an inferred count. Latest training speed is the newest non-null persisted speed.
- Curve points are capped at 2,000 per experiment, chosen from the latest stored observations. Summary values use all persisted episode observations. Charts display discrete observations without connecting or interpolating them.
- `POST /experiments/compare/export` accepts the same selection and streams UTF-8 CSV for only those experiments, in database batches. Empty CSV cells represent values not recorded. Hyperparameters are included on each metric row; legacy configuration values remain empty.

Comparison summaries are descriptive measurements from recorded runs. They do not imply statistical significance or generalize performance beyond the selected experiments.

## Repeated Training Runs and Policy Evaluation

New experiments persist a base random seed (`42` by default) and a requested run count (`3` by default, maximum `20`). A training start creates distinct child run records. Run `i` uses `(base_seed + i) mod 2^32`; seeds, run indices, lifecycle timestamps, actual timesteps, and failure states are stored per run. A run gets its own model artifact, and metrics carry a nullable `run_id`, so pre-Phase-6 metric history remains associated with the original experiment without being reassigned. Each experiment is capped at 100 total recorded runs.

Runs in a start request execute sequentially; the service admits at most one PPO training or evaluation job at a time and bounds active/queued experiment workers and evaluation workers to 10 each. Stopping an experiment cooperatively stops the active run and marks not-yet-started runs stopped. Failed runs do not prevent later repeated runs from being attempted. An experiment is marked failed if any run failed, stopped if cancellation was requested, and completed only when all its runs complete.

Stable-Baselines3 receives the persisted effective seed for each PPO model. This makes the random seed inputs reproducible, but does not promise bitwise-identical outcomes across hardware, library versions, or nondeterministic operations. Existing experiments receive nullable seed/run-count columns during additive initialization; unknown historical settings remain `NULL` and are never reconstructed.

Training run metrics can be retrieved with `GET /experiments/{experiment_id}/runs/{run_id}/metrics` (maximum 10,000 rows). The existing experiment-level metrics and WebSocket routes remain compatible and include run IDs for new observations.

Policy evaluation is a separate, explicit action: `POST /experiments/runs/{run_id}/evaluate` accepts `episode_count` (1–100, default 10) and `base_seed` (default `100000`). Evaluation is allowed only for a completed run with its application-managed model artifact, and only once per run. The evaluator uses deterministic PPO actions and the same seed sequence (`base_seed + episode_index`, modulo `2^32`) for each evaluated run, creating comparable reset conditions. Each actual episode reward, length, and seed is stored in evaluation tables separately from training metrics. `GET /experiments/runs/{run_id}/evaluation` retrieves the persisted evaluation and episode records.

Experiment comparison continues to report training episode summaries separately. Evaluation reward means are first computed for each completed run; the cross-run mean and sample standard deviation (`ddof=1`) are then calculated over those per-run means. The standard deviation is unavailable for fewer than two evaluated runs. Evaluation episode-length means pool actual episodes from completed evaluations, with the episode count shown. Missing or incomplete evaluations remain missing and are not replaced by training rewards. These are descriptive results, not evidence of statistical significance or universal superiority.

The Phase 6 schema is additive: it creates `training_runs`, `policy_evaluations`, and `evaluation_episodes`, adds nullable `base_seed`, `run_count`, and metric `run_id` columns, and adds an index for run metric history. Existing experiment and metric rows are preserved. Automated tests use isolated temporary SQLite databases.

## Reliability and Restart Recovery

Training and policy evaluation share one process-wide compute semaphore, so only one PPO or MuJoCo job can use the compute slot at a time. The process-local training queue and evaluation queue are each bounded to 10 accepted jobs. Experiment starts require a state transition from `created`, `failed`, or `stopped` to `pending`; workers conditionally transition pending runs to running and only running work can become completed. Repeated stop requests are rejected with `409` after the first request moves an experiment to `stopping`. Queued runs observe cancellation before acquiring the compute slot, and stop without starting. Exceptions preserve already-committed episode metrics and mark the affected run and experiment failed; detailed exceptions are logged server-side while persisted user-facing messages omit exception details.

At startup, after additive schema initialization, persisted training runs left `pending` or `running` are marked failed with an explicit "interrupted by backend restart; outcome unknown" message. Active experiments are reconciled from their child run states; a parent marked completed despite an active child is corrected to failed. Pending/running evaluations are also marked failed. Completed runs and their metrics are not reset or rewritten. No work is automatically resumed because PPO checkpoint resumption is not implemented.

The experiment metrics WebSocket sends current persisted status and the latest metric at connection time. Clients reload bounded metric history on connect, reconnect, and server-requested resynchronization, deduplicating by metric ID. Automatic reconnect attempts use exponential backoff capped at 30 seconds and stop after eight consecutive failures; the UI offers a manual retry while the experiment is active. A closed stream does not hide historical charts. Live streaming is best-effort; persisted metric history remains the recovery source.

## Restrictions

The final simulation must use MuJoCo.

The project does not use:

- Gazebo
- ROS
- ROS2
- mujoco-py
- CartPole as the final RL environment