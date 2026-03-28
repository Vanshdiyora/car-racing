# 🏎️ PPO Autonomous Driving — Human vs AI

A production-ready reinforcement learning project that trains a **PPO agent** to drive in [Gymnasium CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/), then lets you **race against the AI** in a split-screen Pygame interface.

---

## Features

| Feature | Details |
|---------|---------|
| **PPO Agent** | Custom CNN policy (Nature-DQN backbone), GAE, clipped surrogate |
| **Custom Env Wrappers** | Frame-skip, grayscale, resize, frame-stack, normalisation |
| **Reward Shaping** | Speed bonus, grass penalty, standing-still penalty, tile visit bonus |
| **Human vs AI Race** | Split-screen Pygame renderer with keyboard controls |
| **Training Pipeline** | Single-env & vectorised training, TensorBoard + CSV logging |
| **Evaluation** | Multi-episode eval, detailed stats, benchmark vs human baseline |
| **Notebooks** | Training analysis & performance evaluation |

---

## Project Structure

```
ppo-autonomous-driving-human-vs-ai/
├── configs/
│   ├── train_config.yaml      # Training hyper-parameters
│   ├── env_config.yaml        # Environment & preprocessing
│   └── race_config.yaml       # Human vs AI race settings
├── src/
│   ├── main.py                # CLI entry point
│   ├── agent/
│   │   ├── ppo_agent.py       # PPO agent with rollout buffer
│   │   ├── policy.py          # NatureCNN + ActorCritic network
│   │   └── utils.py           # Tensor conversion, device helpers
│   ├── env/
│   │   ├── car_env.py         # Environment factory
│   │   ├── preprocess.py      # Gym wrappers (grayscale, resize, etc.)
│   │   └── reward.py          # Custom reward shaping
│   ├── train/
│   │   ├── train.py           # Training entry point
│   │   ├── trainer.py         # Training orchestrator
│   │   └── callbacks.py       # Checkpoints, TensorBoard, CSV
│   ├── evaluate/
│   │   ├── evaluate.py        # Run evaluation episodes
│   │   └── benchmark.py       # PPO vs human comparison
│   ├── race/
│   │   ├── race_engine.py     # Core race loop
│   │   ├── human_control.py   # Keyboard input handler
│   │   ├── ai_control.py      # Model inference wrapper
│   │   ├── renderer.py        # Pygame split-screen display
│   │   └── metrics.py         # Lap time, speed, penalties
│   └── utils/
│       ├── config_loader.py   # YAML loading & merging
│       ├── logger.py          # Console + CSV logging
│       └── seed.py            # Global seed setting
├── tests/
│   ├── test_env.py
│   ├── test_agent.py
│   └── test_race.py
├── notebooks/
│   ├── training_analysis.ipynb
│   └── performance_eval.ipynb
├── models/                    # Saved checkpoints (gitignored)
├── logs/                      # Training logs (gitignored)
├── outputs/                   # Videos, plots, reports (gitignored)
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Quick Start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
pip install -e .
```

### 2. Train the PPO Agent

```bash
# Default: 4 parallel envs, 2M timesteps
python -m src.main train

# Quick test
python -m src.main train --timesteps 50000 --n-envs 1

# Custom config
python -m src.main train --train-config configs/train_config.yaml --env-config configs/env_config.yaml
```

### 3. Evaluate

```bash
python -m src.main eval models/ppo_best/best_model.pt --episodes 20
python -m src.main eval models/ppo_best/best_model.pt --render
```

### 4. Race Against the AI! 🏁

```bash
python -m src.main race
```

**Controls:**
- ⬆️ Accelerate
- ⬇️ Brake
- ⬅️ Steer left
- ➡️ Steer right
- `ESC` Quit

### 5. Benchmark

```bash
python -m src.main benchmark models/ppo_best/best_model.pt --episodes 50
```

### 6. TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

---

## Configuration

All settings live in `configs/`. Edit the YAML files or override via CLI:

```yaml
# configs/train_config.yaml
training:
  total_timesteps: 3_000_000
  learning_rate: 2.5e-4
  device: cuda
```

```yaml
# configs/env_config.yaml
reward_shaping:
  enabled: true
  grass_penalty: -1.0
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Gymnasium (Box2D)
- Pygame (for race mode)
- TensorBoard, NumPy, Pillow, tqdm, PyYAML, matplotlib

---

## License

MIT
