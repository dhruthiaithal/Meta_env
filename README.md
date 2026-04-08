---
title: MedSched Pro v2
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - openenv
  - simulation
  - healthcare
---

# MedSched Pro v2 Environment

MedSched Pro v2 is a highly robust reinforcement learning and LLM-compatible environment for temporal clinical scheduling. Designed around the OpenEnv standard, it provides a realistic testbed for scheduling and triage agents.

## Environment Details
- **Time Representation:** 4-hour clinic shift running on a simulated Clock 0 to 240 mins (15-minute steps).
- **Observation Space:** 
  - `clinic_clock`: current integer time.
  - `waiting_room`: strictly-typed list of unassigned `Patient` records.
  - `doctor_schedule`: 8 available appointment slots.
  - `remaining_capacity`: integer count of free slots.
- **Action Space:** Single-step action `Action(patient_id, priority, slot_index)`. For RL, mapped to `Discrete(24)`.
- **Rewards:** Agents are heavily penalized if an Emergency patient's health decays significantly, and rewarded iteratively for appropriate triaging and valid scheduling.

## Tasks
1. **The Morning Queue (Easy):** Triage and schedule 2 routine patients.
2. **The Triage Test (Medium):** Identify 1 Emergency among 4 Routine patients and prioritize them.
3. **The Surge (Hard):** 10 patients arrive, only 8 slots exist. The agent must sacrifice Routine patients to save the Emergency.

## Getting Started

### Local Setup
```bash
pip install -r requirements.txt
```

### Baseline Inference (LLM Evaluator)
This runs the baseline evaluation script using an OpenAI-compatible API.
```bash
export HF_TOKEN="your_token_here"
# Note defaults limit calls to OpenAI, so API_BASE_URL and MODEL_NAME can be overridden
python baseline.py
```

### Reinforcement Learning Training
Train a PPO agent from Stable Baselines 3 using the wrapped OpenEnv:
```bash
python train_rl.py
```

### Docker
```bash
docker build -t medsched_v2 .
docker run -e HF_TOKEN=$HF_TOKEN medsched_v2
```
