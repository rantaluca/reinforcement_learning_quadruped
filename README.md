# Spot-Inspired Quadruped Reinforcement Learning 🐾

<p align="center">
    <img src="assets/reel_bot.png" alt="Quadruped robot simulation" width="420"/>
  <img src="assets/quadruped_env.png" alt="Quadruped robot simulation" width="420"/>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-≥3.10-blue.svg?logo=python&logoColor=white" alt="Python Version">
  </a>
  <a href="https://pybullet.org/">
    <img src="https://img.shields.io/badge/PyBullet-Physics%20Engine-red.svg?logo=python&logoColor=white" alt="PyBullet">
  </a>
  <a href="https://gymnasium.farama.org/">
    <img src="https://img.shields.io/badge/Gymnasium-Custom%20Env-green.svg?logo=openai&logoColor=white" alt="Gymnasium">
  </a>
  <a href="https://github.com/DLR-RM/stable-baselines3">
    <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg?logo=robotframework&logoColor=white" alt="Stable Baselines3">
  </a>
</p>

---

## Overview

This project implements a **custom reinforcement learning environment** for a **Spot-inspired quadruped robot** (designed by myself), built from scratch in **PyBullet** and compatible with **Gymnasium** and **Stable-Baselines3**.

This work is part of my ongoing effort to make the quadruped robot I personally designed and built learn to walk autonomously.

The robot has **12 actuated joints** and learns to **follow a target twist command** (linear and angular velocity) using **Proximal Policy Optimization (PPO)**.  
Observations include joint states and simulated IMU data, enabling rich proprioceptive feedback for control learning.

Developed as a personal research project in **robotics and embodied AI**, this environment focuses on realistic reward shaping, normalized observations, and reproducible training pipelines.

---

## Environment Structure

```
quadruped_rl/
├── envs/
│   └── simple_quadruped_env.py     # Custom Gymnasium environment
├── ressources/
│   └── urdfs/spot/spot_v2.urdf     # Spot-inspired URDF model
├── models/                         # Saved PPO checkpoints
├── train_quadruped.py              # Training entry point (CLI)
└── assets/
    ├── quadruped_env.png
    └── reward_curve.png
```

---

## Key Features

- **12-Joint Robot** — Simulates a Spot-like quadruped with accurate joint limits  
- **Custom Reward Function** — Encourages forward progress, upright stability, and low slip  
- **Normalized Observations** — 33-dimensional observation vector including joint states, IMU, and target twist  
- **PID control for joint commands** — the policy controls PID-assisted legs, which increases stability
- **Stable-Baselines3** — Uses the PPO algorithm to learn the control policy

---

## Observation Space

| Component | Count | Description |
|------------|--------|-------------|
| Joint positions | 12 | Normalized joint angles |
| Joint velocities | 12 | Normalized joint speeds |
| IMU linear + angular velocity | 6 | Simulated body motion |
| Target twist | 3 | Linear & angular velocity command |

Total: **33-dimensional observation vector**

---

## Reward Design

The reward encourages the robot to walk stably toward the commanded direction:

\[
r = -w_v \|v - v^*\| - w_\omega |\omega - \omega^*| + w_u \cdot \text{uprightness} + w_p \cdot \text{progress} - w_z |v_z|
\]

| Term | Description | Weight |
|------|--------------|---------|
| \( w_v \) | Planar velocity tracking | 1.3 |
| \( w_\omega \) | Yaw rate tracking | 1.3 |
| \( w_u \) | Uprightness | 0.3 |
| \( w_p \) | Progress shaping | 20.0 |
| \( w_z \) | Vertical slip penalty | 0.1 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/rantaluca/Quadruped_RL.git
cd Quadruped_RL

# Install dependencies
pip install gymnasium pybullet stable-baselines3 numpy matplotlib tqdm
```

---

## Training

You can launch training with or without GUI visualization:

```bash

export PYTHONPATH=.

# Headless mode (faster)
python train_quadruped.py

# With GUI
python train_quadruped.py --gui
```

Optional arguments:
```bash
--timesteps 10000000     # Total training steps (default: 1e9)
--save-freq 20000        # Save model every N steps
```

---

## Example Results

<p align="center">
  <img src="assets/reward_curve.png" alt="Reward evolution" width="600"/>
</p>

The PPO agent learns a **stable forward gait**, maintaining upright posture while tracking commanded velocities.  

Future work includes terrain adaptation, sensor fusion (LIDAR, camera), and sim-to-real transfer to the real DIY robot.

---

## Author

**Robert Antaluca**  
ÉTS Montréal / Université de Technologie de Compiègne  
Website: [antaluca.com](https://antaluca.com)

---

## License

MIT License — free to use, modify, and extend for research or educational purposes.

