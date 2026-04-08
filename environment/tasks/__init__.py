"""
tasks/__init__.py — Task registry for the Attention Economy Environment.

Usage:
    from env.tasks import get_task

    task_cfg, initial_user = get_task("medium")
"""

from __future__ import annotations
from typing import Tuple
from tasks import easy, medium, hard


def get_task(task_id: str):
    """
    Retrieve task configuration and initial user state by task ID.

    Parameters
    ----------
    task_id : str
        One of "easy", "medium", "hard"

    Returns
    -------
    (task_config, UserState) — frozen task config + deterministic user state
    """
    if task_id == "easy":
        return easy.get_task_config(), easy.get_initial_user()
    elif task_id == "medium":
        return medium.get_task_config(), medium.get_initial_user()
    elif task_id == "hard":
        return hard.get_task_config(), hard.get_initial_user()
    else:
        raise ValueError(
            f"Unknown task_id: '{task_id}'. "
            f"Choose from 'easy', 'medium', 'hard'."
        )