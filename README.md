# Deep Reinforcement Learning for CarRacing

**Benchmarking PPO, DDPG, and SAC for vision-based continuous control  -  Ecole Polytechnique  -  Jan-Mar 2026**

This project investigates deep reinforcement learning for autonomous control in the Gymnasium **CarRacing** environment. We benchmark three continuous-control algorithms — **PPO, DDPG, and SAC** — and analyze their learning dynamics, stability across random seeds, and generalization to randomly generated tracks.

Under the experimental training budget, **PPO achieved the strongest overall performance**, reaching an average reward of approximately **760**, compared with **687 for DDPG** and **611 for SAC**.

---

## Overview

The project started with a more ambitious goal: learning to drive directly in **TrackMania**.

In practice, TrackMania proved poorly suited for rapid reinforcement learning experimentation:

- the environment ran in real time,
- episodes took roughly 30 seconds,
- only one game instance could easily be run on a machine,
- environment interaction became the main training bottleneck.

We therefore moved to Gymnasium's **CarRacing** environment, which provides a lightweight simulation that can run faster than real time while preserving the main challenges of the original task.

CarRacing is a particularly interesting RL benchmark because it combines:

- **96 × 96 RGB visual observations**
- **continuous actions** for steering, acceleration, and braking
- nonlinear vehicle dynamics
- randomly generated race tracks

The task therefore requires an agent to learn continuous control directly from visual observations.

---

## Algorithms

We compare three widely used deep reinforcement learning approaches for continuous control.

### Proximal Policy Optimization — PPO

PPO is an **on-policy policy-gradient algorithm** that constrains policy updates using a clipped objective.

The clipping mechanism limits destructive policy updates and generally provides stable optimization.

In our experiments, PPO offered the best trade-off between performance and robustness under a limited training budget.

**Mean evaluation reward: ~760**

---

### Deep Deterministic Policy Gradient — DDPG

DDPG is an **off-policy actor-critic algorithm** designed for continuous action spaces.

It combines:

- a deterministic actor,
- a critic estimating action values,
- experience replay,
- noise-based exploration.

DDPG achieved competitive performance, but showed substantially greater sensitivity to initialization and random seed.

**Mean evaluation reward: ~687**

---

### Soft Actor-Critic — SAC

SAC is an off-policy actor-critic method using a **stochastic policy** and an entropy-regularized objective.

Unlike DDPG, SAC explicitly encourages exploration while optimizing expected reward.

Although theoretically attractive, SAC required more training to fully exploit this exploration strategy within our experimental setup.

**Mean evaluation reward: ~611**

---

## Experimental Protocol

To compare the algorithms under consistent conditions, we used:

- **100,000 training timesteps**
- **3 training seeds:** `0`, `1`, and `2`
- randomly generated tracks during training
- continuous steering, acceleration, and braking
- RGB image observations
- evaluation across multiple randomly generated circuits

The analysis focuses on three main questions:

1. **Learning speed** — how quickly does each algorithm improve?
2. **Stability** — how sensitive is performance to the training seed?
3. **Generalization** — how well do trained policies perform on randomly generated evaluation tracks?

---

## Results

| Algorithm | Mean Reward | Main Observation |
|---|---:|---|
| **PPO** | **~760** | Best overall short-budget performance and robustness |
| **DDPG** | **~687** | Competitive but highly seed-dependent |
| **SAC** | **~611** | Strong exploration framework but needs more training |

### PPO

PPO produced the strongest overall policy under the available training budget.

Its clipped policy updates resulted in comparatively stable learning while still reaching the highest evaluation reward.

### DDPG

DDPG was able to reach strong policies but exhibited much higher variance across seeds.

This illustrates one of the challenges of deterministic-policy methods: their exploration strategy and optimization can be sensitive to hyperparameters and initialization.

### SAC

SAC's entropy-based exploration is attractive for continuous-control tasks, but the algorithm was computationally heavier and slower to reach strong policies in our experiments.

A larger training budget would likely be required for a more complete comparison.

---

## Key Takeaways

The experiments highlight several practical lessons about deep reinforcement learning:

- **Environment throughput matters.**  
  Moving from TrackMania to a lightweight simulator dramatically increased the amount of experience that could be collected.

- **Algorithm complexity does not automatically translate into better performance.**  
  Under our training budget, PPO outperformed both DDPG and SAC.

- **Random seeds matter.**  
  DDPG in particular showed large variation between independent training runs.

- **Training budget affects algorithm rankings.**  
  SAC's weaker performance should be interpreted in the context of the relatively short training horizon rather than as a general statement about the algorithm.

- **Generalization should be evaluated explicitly.**  
  Performance on randomly generated circuits provides a stronger signal than evaluating an agent only on one fixed track.

---

## Project Structure

```text
Deep-RL-CarRacing/
│
├── configs/
│   ├── common.yaml
│   ├── ppo.yaml
│   ├── ddpg.yaml
│   └── sac.yaml
│
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── config.py
│   ├── env_factory.py
│   ├── evaluate.py
│   ├── plots.py
│   └── train.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_all_training.py
│   ├── run_all_evaluation.py
│   ├── run_exploration_experiments.py
│   ├── run_full_pipeline.py
│   ├── generate_plots.py
│   └── watch_agent.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── figures/
│   └── ...
│
├── presentation/
│   └── Reinforcement_Learning_CarRacing_Presentation.pdf
│
├── requirements.txt
└── README.md
```

---

## Running the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Training and evaluation are controlled through the configuration files in `configs/`.

For example, individual training runs can be launched through the scripts in:

```text
scripts/
```

The repository also includes scripts for:

- multi-seed training,
- evaluation,
- exploration experiments,
- result visualization,
- running the complete experimental pipeline,
- watching a trained agent interact with the environment.

See the configuration files for the experiment parameters used by each algorithm.

---

## Tech Stack

**Python · PyTorch · Stable-Baselines3 · Gymnasium · NumPy · Pandas · Matplotlib · YAML**

The project covers:

- deep reinforcement learning
- continuous control
- actor-critic methods
- experiment configuration
- multi-seed evaluation
- model benchmarking
- statistical analysis
- scientific visualization

---

## Presentation

The final project presentation contains the motivation, theoretical background, experimental protocol, learning curves, stability analysis, and final comparison of PPO, DDPG, and SAC.

**[View the final presentation](presentation/Reinforcement_Learning_CarRacing_Presentation.pdf)**

---

## Authors

This project was developed collaboratively by:

- **Timothée Bessard**
- **Tom Leon**
- **Cedric Trinh**
- **Henri-Pierre Gawel**
- **Titouan Salin**

as part of the Reinforcement Learning course at **École Polytechnique**.
