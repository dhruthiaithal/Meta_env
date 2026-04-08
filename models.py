from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"

class Specialist(str, Enum):
    GP = "GP"
    CARDIOLOGIST = "CARDIOLOGIST"

class Patient(BaseModel):
    id: str
    arrival_time: int  # Minutes from start
    symptoms: str
    health_score: float = 1.0  # Drops over time if untreated
    assigned_priority: Optional[Priority] = None

class Action(BaseModel):
    # Action space: [Triage Patient, Assign to Slot]
    patient_id: str
    priority: Priority
    slot_index: int  # 0 to 7 (8 slots in a 4hr shift)

class Observation(BaseModel):
    clinic_clock: int  # 0 to 240 minutes
    waiting_room: List[Patient]
    doctor_schedule: List[Optional[str]]  # Patient IDs in slots
    remaining_capacity: int

class Reward(BaseModel):
    value: float
    reason: str