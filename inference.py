import os
import json
import time
from openai import OpenAI
from env import MedSchedProEnv
from models import Action, Priority
import tasks

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

def run_task(task_name, patients, grader):
    env = MedSchedProEnv()
    obs = env.reset(task_config=patients)
    print(f"\n[START] Task: {task_name}")
    
    start_time = time.time()
    
    while True:
        if len(obs.waiting_room) == 0:
            break
            
        patient = obs.waiting_room[0]
        # find first open slot
        first_open_slot = next((i for i, v in enumerate(obs.doctor_schedule) if v is None), 0)
        
        prompt = f"""
        You are a clinical scheduling AI.
        Patient: {patient.model_dump_json()}
        Available capacity: {obs.remaining_capacity}
        First open slot index (0-7): {first_open_slot}
        
        Determine their priority (routine, urgent, emergency) and the best slot index (0 to 7) to schedule them in.
        Provide your response exactly as JSON: {{"priority": "routine", "slot_index": {first_open_slot}}}
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result_json = json.loads(response.choices[0].message.content)
            priority_str = result_json.get("priority", "routine").lower()
            
            if priority_str == "emergency":
                priority = Priority.EMERGENCY
            elif priority_str == "urgent":
                priority = Priority.URGENT
            else:
                priority = Priority.ROUTINE
                
            slot_idx = int(result_json.get("slot_index", first_open_slot))
        except Exception as e:
            # Fallback logic if LLM fails or API errors
            priority = Priority.ROUTINE
            slot_idx = first_open_slot
            
        action = Action(patient_id=patient.id, priority=priority, slot_index=slot_idx)
        obs, reward, done, info = env.step(action)
        
        # Evaluator expected formats
        print(f"[STEP] Action: {action.model_dump_json()} | Reward: {reward.value:.2f}")
        
        if done:
            break
            
    score = grader(env)
    elapsed = time.time() - start_time
    print(f"[END] Task: {task_name} | Score: {score:.2f} | Time: {elapsed:.2f}s")
    return score

def run_inference():
    s1 = run_task("The Morning Queue", tasks.get_easy_task(), tasks.grade_easy_task)
    s2 = run_task("The Triage Test", tasks.get_medium_task(), tasks.grade_medium_task)
    s3 = run_task("The Surge", tasks.get_hard_task(), tasks.grade_hard_task)

if __name__ == "__main__":
    run_inference()
