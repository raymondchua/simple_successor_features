DOMAINS = [
    "walker",
    "quadruped",
    "jaco",
    "cheetah",
    "humanoid",
]

CHEETAH_TASKS = [
    "cheetah_run",
    "cheetah_run_backward",
    "cheetah_flip",
    "cheetah_flip_backward",
]

WALKER_TASKS = [
    "walker_stand",
    "walker_walk",
    "walker_run",
    "walker_run_backward",
    "walker_flip",
]

QUADRUPED_TASKS = [
    "quadruped_walk",
    "quadruped_run",
    "quadruped_stand",
    "quadruped_jump",
]

JACO_TASKS = [
    "jaco_reach_top_left",
    "jaco_reach_top_right",
    "jaco_reach_bottom_left",
    "jaco_reach_bottom_right",
]

HUMANOID_TASKS = [
    "humanoid_stand",
    "humanoid_walk",
    "humanoid_run",
]

TASKS = (
    WALKER_TASKS
    + QUADRUPED_TASKS
    + JACO_TASKS
    + CHEETAH_TASKS
    + HUMANOID_TASKS
)

PRIMAL_TASKS = {
    "cheetah": "cheetah_run",
    "walker": "walker_run",
}

PRIMAL_TASKS_RUN_BACKWARD = {
    "cheetah": "cheetah_run_backward",
    "walker": "walker_run_backward",
}

"""Walker and Cheetah tasks for CRL experiments."""

# =============================== Walker ===============================

CRL_WALKER_DIFF_REWARD_TASKS = [
    "walker_run",
    "walker_run_backward",
]

CRL_WALKER_WALK_RUN_TASKS = [
    "walker_walk",
    "walker_run",
]

CRL_WALKER_STAND_RUN_TASKS = [
    "walker_stand",
    "walker_run",
]

# =============================== Cheetah ===============================

CRL_CHEETAH_DIFF_RUN_SPEED_REWARD_TASKS = [
    "cheetah_run",
    "cheetahfast_run",
]

CRL_CHEETAH_DIFF_REWARD_TASKS = [
    "cheetah_run",
    "cheetah_run_backward",
]


# =============================== Quadruped ===============================
CRL_QUADRUPED_RUN_JUMP_TASKS = [
    "quadruped_run",
    "quadruped_jump",
]

# ================================= CRL Different Domain ======================
CRL_DIFF_DOMAINS_SAME_REWARD = [
    "cheetah_run",
    "walker_run",
]

# =============================== CRL Task Sets ===============================

CRL_TASKS_DIFF_REWARD = {
    'walker': CRL_WALKER_DIFF_REWARD_TASKS,
    'cheetah': CRL_CHEETAH_DIFF_REWARD_TASKS,
}

# walker walk and run
CRL_WALKER_WALK_RUN_TASKS = {
    'walker': CRL_WALKER_WALK_RUN_TASKS,
}

# walker stand and run
CRL_WALKER_STAND_RUN_TASKS = {
    'walker': CRL_WALKER_STAND_RUN_TASKS,
}

# quadruped run and jump
CRL_RUN_JUMP_TASKS = {
    'quadruped': CRL_QUADRUPED_RUN_JUMP_TASKS,
}





