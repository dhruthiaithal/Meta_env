from models import Patient

def get_easy_task():
    return [
        Patient(id="P1", arrival_time=0, symptoms="Routine checkup", health_score=1.0),
        Patient(id="P2", arrival_time=5, symptoms="Routine physical exam", health_score=1.0)
    ]

def grade_easy_task(env):
    if "P1" in env.schedule and "P2" in env.schedule:
        return 0.95
    return 0.05

def get_medium_task():
    return [
        Patient(id="P1", arrival_time=0, symptoms="Routine checkup", health_score=1.0),
        Patient(id="P2", arrival_time=5, symptoms="Severe chest pain", health_score=1.0),
        Patient(id="P3", arrival_time=10, symptoms="Routine vaccination", health_score=1.0),
        Patient(id="P4", arrival_time=15, symptoms="Routine blood test", health_score=1.0),
        Patient(id="P5", arrival_time=20, symptoms="Routine vision test", health_score=1.0)
    ]

def grade_medium_task(env):
    # P2 is the emergency
    if "P2" in env.schedule:
        idx = env.schedule.index("P2")
        if idx == 0:
            return 0.95
        elif idx < 4:
            return 0.8
        return 0.5
    return 0.05

def get_hard_task():
    patients = [Patient(id=f"P{i}", arrival_time=i*5, symptoms="Routine Follow-up", health_score=1.0) for i in range(1, 10)]
    patients.append(Patient(id="P10", arrival_time=40, symptoms="Severe chest pain and shortness of breath", health_score=1.0))
    return patients

def grade_hard_task(env):
    # 10 patients, 8 slots. P10 must take precedence.
    if "P10" in env.schedule:
        return 0.95
    return 0.05