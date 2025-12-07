"""
go2_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from b2_base import train_main




# ---- patch only the bits that change --------------------------------------


env_cfg_patch = {
    "self_collision": False,
    "randomize_rot": True,
    'max_push_vel_xy': 1.0,
    # "base_init_pos": [0.0, 0.0, 0.45],
    "termination_if_roll_greater_than": 90,
    "termination_if_pitch_greater_than": 90,
    "angle_termination_duration": 1.0, #seconds
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 0.9,
    "reward_scales": {
        "tracking_lin_vel": 1.5, #1.5
        "tracking_ang_vel": 0.75, #0.75
        "lin_vel_z": -5.0,
        "relative_base_height": -50.0,
        "orientation": -20.0, #fixed!
        "ang_vel_xy": -0.05, #fixed!
        "collision": -10.0, #fixed!
        "dof_pos_limits": -20.0, #fixed!
        "termination": -30.0,
        "dof_acc": -2.5e-7,
        # "action_rate": -0.01,
        # "feet_contact_forces": -0.0001,
        "default_pose_when_idle": -2.0,
        "all_feet_contact_when_idle": -1.0,
        "feet_stumble": -0.5,
        "alive": 1.0,
        "action_curvature": -0.02,
        "effort_symmetry": -0.01,
        # "front_feet_clearance": 30.0,
        "leg_cross": -1.0,
        "leg_cross_fore_aft": -1.0,
        # "front_feet_forward": 5.0
    },
}



terrain_cfg_patch = {
    "terrain_type": "plane", #plane #single_step
}


command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_iteration_threshold": 2000,
    "mean_reward_threshold": 25,
    "goal_probability": 0.0,
    # "lin_vel_x_range": [-1.5, 1.5],
    # "lin_vel_y_range": [-1.0, 1.0],
    # "ang_vel_range": [-1.0, 1.0],
    "lin_vel_x_range": [-0.0, 0.0],
    "lin_vel_y_range": [-0.0, 0.0],
    "ang_vel_range": [-0.0, 0.0],
}



# leave other five cfgs untouched
CFG_PATCHES = (
    env_cfg_patch,  # env_cfg
    {},  # obs_cfg
    {},  # noise_cfg
    reward_cfg_patch,  # reward_cfg
    command_cfg_patch,  # command_cfg
    terrain_cfg_patch,
)

if __name__ == "__main__":
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="b2_walking")
