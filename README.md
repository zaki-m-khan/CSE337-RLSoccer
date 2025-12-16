# CSE337 RL Soccer Kicker

Reinforcement learning project training agents to kick a soccer ball into a goal using MuJoCo physics simulation.

## Methods

- **SARSA** with tile coding for open goal condition
- **DDPG** with neural networks for goalie condition

## How to Run

1. Open `notebooks/20_final_model.ipynb` in Google Colab
2. Run all cells (`Runtime` → `Run all`)
3. Results are saved to `notebooks/` folder

## Requirements

```
pip install mujoco==3.1.6 gymnasium numpy matplotlib pandas torch
```

## Project Structure

```
├── notebooks/
│   └── 20_final_model.ipynb   # Main experiment notebook
├── assets/
│   └── soccer_min.xml         # MuJoCo environment model
├── src/
│   ├── soccer_env.py          # Environment implementation
│   ├── tilecoding.py          # Tile coding for SARSA
│   └── sarsa_agent.py         # SARSA agent
└── requirements.txt
```

## Experiment Settings

- **Actions**: 28 discrete kicks (7 yaw angles × 4 speeds)
- **Seeds**: 0, 42, 123 (for reproducibility)
- **Episodes**: 500 per seed
- **Noise**: ±3° yaw, ±10% speed
- **Goal**: 2.4m wide × 1.8m tall
- **Reward**: +5 goal, -5 miss/blocked

