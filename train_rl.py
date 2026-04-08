import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import MedSchedProEnv
from models import Action, Priority, Patient

class OpenEnvWrapper(gym.Env):
    """Wraps our MedSched OpenEnv for standard RL libraries"""
    def __init__(self, task_patients=None):
        super().__init__()
        self.internal_env = MedSchedProEnv()
        self.base_task_patients = task_patients or []
        
        # Observation space
        self.observation_space = spaces.Dict({
            "clinic_clock": spaces.Box(low=0, high=240, shape=(1,), dtype=np.int32),
            "waiting_room_count": spaces.Discrete(100),
            "doctor_schedule": spaces.MultiBinary(8),  # 1 if occupied, 0 if free
        })

        # Action space: 24 discrete actions
        # 3 priorities x 8 slots = 24 combinations
        # priority_idx = action // 8
        # slot_idx = action % 8
        self.action_space = spaces.Discrete(24) 

    def _get_obs(self, obs_pydantic):
        schedule_obs = np.zeros(8, dtype=np.int8)
        for i, val in enumerate(obs_pydantic.doctor_schedule):
            if val is not None:
                schedule_obs[i] = 1
                
        return {
            "clinic_clock": np.array([obs_pydantic.clinic_clock], dtype=np.int32),
            "waiting_room_count": min(len(obs_pydantic.waiting_room), 99),
            "doctor_schedule": schedule_obs
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_pydantic = self.internal_env.reset(self.base_task_patients)
        return self._get_obs(obs_pydantic), {}

    def step(self, action):
        pending = self.internal_env.state().waiting_room
        if not pending:
            obs_pydantic, reward, done, info = self.internal_env.step(
                Action(patient_id="dummy", priority=Priority.ROUTINE, slot_index=0)
            )
            return self._get_obs(obs_pydantic), reward.value, done, False, info
            
        # Triage top patient
        top_patient = pending[0]
        
        priority_idx = action // 8
        slot_idx = action % 8
        
        if priority_idx == 0:
            priority = Priority.ROUTINE
        elif priority_idx == 1:
            priority = Priority.URGENT
        else:
            priority = Priority.EMERGENCY
            
        pydantic_action = Action(
            patient_id=top_patient.id,
            priority=priority,
            slot_index=slot_idx
        )
        
        obs_pydantic, reward, done, info = self.internal_env.step(pydantic_action)
        return self._get_obs(obs_pydantic), reward.value, done, False, {}

if __name__ == "__main__":
    print("Setting up OpenEnvWrapper...")
    test_patients = [
        Patient(id="P1", arrival_time=0, symptoms="Fever"),
        Patient(id="P2", arrival_time=5, symptoms="Chest pain")
    ]
    
    env = OpenEnvWrapper(task_patients=test_patients)
    check_env(env)
    
    print("Initializing PPO Model...")
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    print("Starting Training (1000 timesteps)...")
    model.learn(total_timesteps=1000)
    model.save("medsched_agent_v2")
    print("Training Complete! Agent saved as medsched_agent_v2.zip")
