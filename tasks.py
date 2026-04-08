from models import Patient

def clamp_score(score):
    # ensures strictly between 0 and 1 (never exactly 0 or 1)
    return max(0.01, min(0.99, score))


def get_easy_task():
    return [
        Patient(id="P1", arrival_time=0, symptoms="Routine checkup", health_score=0.95),
        Patient(id="P2", arrival_time=5, symptoms="Routine physical exam", health_score=0.95)
    ]


def grade_easy_task(env):
    score = 0.1
    if "P1" in env.schedule and "P2" in env.schedule:
        score = 0.9
    return clamp_score(score)


def get_medium_task():
    return [
        Patient(id="P1", arrival_time=0, symptoms="Routine checkup", health_score=0.95),
        Patient(id="P2", arrival_time=5, symptoms="Severe chest pain", health_score=0.95),
        Patient(id="P3", arrival_time=10, symptoms="Routine vaccination", health_score=0.95),
        Patient(id="P4", arrival_time=15, symptoms="Routine blood test", health_score=0.95),
        Patient(id="P5", arrival_time=20, symptoms="Routine vision test", health_score=0.95)
    ]


def grade_medium_task(env):
    score = 0.1
    if "P2" in env.schedule:
        idx = env.schedule.index("P2")
        if idx == 0:
            score = 0.9
        elif idx < 4:
            score = 0.8
        else:
            score = 0.5
    return clamp_score(score)


def get_hard_task():
    patients = [
        Patient(id=f"P{i}", arrival_time=i*5, symptoms="Routine Follow-up", health_score=0.95)
        for i in range(1, 10)
    ]
    patients.append(
        Patient(id="P10", arrival_time=40, symptoms="Severe chest pain and shortness of breath", health_score=0.95)
    )
    return patients


def grade_hard_task(env):
    score = 0.1
    if "P10" in env.schedule:
        score = 0.9
    return clamp_score(score)