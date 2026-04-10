from environment.tasks.easy import EasyTaskConfig, get_initial_user as easy_user
from environment.tasks.medium import MediumTaskConfig, get_initial_user as medium_user
from environment.tasks.hard import HardTaskConfig, get_initial_user as hard_user

# Map both short names AND openenv.yaml task IDs
TASK_ALIASES = {
    "easy":                 "easy",
    "easy_recommendation":  "easy",
    "medium":               "medium",
    "diverse_feed":         "medium",
    "hard":                 "hard",
    "trust_preservation":   "hard",
}

def get_task(task_id: str):
    resolved = TASK_ALIASES.get(task_id)
    if resolved is None:
        raise ValueError(f"Unknown task: {task_id}. Choose: {list(TASK_ALIASES.keys())}")
    if resolved == "easy":
        return EasyTaskConfig(), easy_user()
    elif resolved == "medium":
        return MediumTaskConfig(), medium_user()
    elif resolved == "hard":
        return HardTaskConfig(), hard_user()