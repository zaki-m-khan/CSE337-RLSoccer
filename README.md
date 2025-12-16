# CSE337 RL Soccer Kicker

Reinforcement learning project training agents to kick a soccer ball into a goal using MuJoCo physics simulation.

## Methods

- **SARSA** with tile coding for open goal condition
- **DDPG** with neural networks for goalie condition

## How to Run

Open the notebooks in [Google Colab](https://colab.research.google.com) and run all cells.

| Notebook | Description | Runtime |
|----------|-------------|---------|
| `00_mujoco_setup.ipynb` | Environment setup and visualization | ~1 min |
| `20_final_model.ipynb` | SARSA and DDPG experiments | ~10-15 min |

## Requirements

```
pip install mujoco==3.1.6 gymnasium numpy matplotlib pandas torch
```

## Project Structure

```
├── notebooks/
│   ├── 00_mujoco_setup.ipynb    # Environment test + visualization
│   └── 20_final_model.ipynb     # Main experiments (SARSA & DDPG)
├── assets/
│   └── soccer_min.xml           # MuJoCo environment model
├── src/
│   ├── soccer_env.py            # Environment implementation
│   ├── tilecoding.py            # Tile coding for SARSA
│   └── sarsa_agent.py           # SARSA agent
└── requirements.txt
```

## Experiment Settings

| Parameter | Value |
|-----------|-------|
| Actions | 28 discrete kicks (7 yaw × 4 speed) |
| Seeds | 0, 42, 123 |
| Episodes | 500 per seed |
| Noise | ±3° yaw, ±10% speed |
| Goal | 2.4m wide × 1.8m tall |
| Reward | +5 goal, -5 miss/blocked |
