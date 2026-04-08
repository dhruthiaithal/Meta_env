import copy
from typing import Optional, List
from models import Observation, Action, Priority, Reward, Patient

class MedSchedProEnv:
    def __init__(self):
        self.MAX_TIME = 240 # 4 hours
        self.reset()

    def reset(self, task_config=None):
        self.clock = 0
        self.schedule = [None] * 8 # 8 slots in a 4hr shift
        self.patients = [copy.deepcopy(p) for p in (task_config or [])]
        return self.state()

    def state(self) -> Observation:
        # waiting room contains only patients not scheduled
        waiting_room = [p for p in self.patients if p.id not in self.schedule]
        return Observation(
            clinic_clock=self.clock,
            waiting_room=waiting_room,
            doctor_schedule=self.schedule,
            remaining_capacity=self.schedule.count(None)
        )

    def step(self, action: Action):
        # 1. Update Simulation Clock
        self.clock += 15 
        
        info = {}
        
        # 2. Find Patient
        patient = next((p for p in self.patients if p.id == action.patient_id), None)
        if not patient:
            done = self.clock >= self.MAX_TIME or self.schedule.count(None) == 0
            return self.state(), Reward(value=-1.0, reason="Invalid Patient"), done, info

        # 3. Process Triage & Scheduling
        patient.assigned_priority = action.priority
        
        # Reward Logic (The Signal)
        reward_val = 0.0
        reason = "Action processed"

        if action.priority == Priority.EMERGENCY and ("chest pain" in patient.symptoms.lower() or "severe" in patient.symptoms.lower()):
            reward_val += 0.5  # Correct triage
        elif action.priority == Priority.ROUTINE and "routine" in patient.symptoms.lower():
            reward_val += 0.2
            
        if 0 <= action.slot_index < 8:
            if self.schedule[action.slot_index] is None:
                self.schedule[action.slot_index] = patient.id
                reward_val += 0.3
            else:
                reward_val -= 0.5 # Penalty for double-booking
                reason = "Double-booking penalty"

        # 4. Health Decay Logic (The "Robust" part)
        for p in self.patients:
            if p.assigned_priority == Priority.EMERGENCY and p.id not in self.schedule:
                p.health_score -= 0.05 # Emergency patient is getting worse!
                if p.health_score < 0.8: 
                    reward_val -= 0.5
                    reason = "Decay penalty"
        
        done = self.clock >= self.MAX_TIME or self.schedule.count(None) == 0
        return self.state(), Reward(value=reward_val, reason=reason), done, info