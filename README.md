# WindGym-Zoo

This repository hosts the **official model zoo** for [WindGym](https://gitlab.windenergy.dtu.dk/sys/windgym): a reinforcement learning environment for wind farm control under sensor uncertainty and noisy flow conditions.

WindGym-Zoo includes:
- 📦 Pretrained RL agents (e.g., PPO, SAC)
- 🧊 Frozen environment configurations for reproducible evaluation
- 📈 Baseline metrics for leaderboard-style comparisons

This repository is designed to support nightly benchmarking pipelines and community-submitted agents.

---

## 🔍 Contents

windgym-zoo/
├── agents/ # Pretrained agents with metadata and evaluation outputs
├── configs/ # Frozen WindGym environment configs (for evaluation)
├── results/ # Leaderboard results (optional, updated nightly)
├── templates/ # Submission templates and metadata guidelines
└── README.md


## 🧪 Getting Started

Clone this repo alongside `windgym`, then evaluate all models:

```bash
git clone https://gitlab.com/kilojoules/windgym-zoo
cd windgym
pixi run python scripts/eval_leaderboard.py --zoo ../windgym-zoo
```

