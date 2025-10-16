"""
go2_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from go2_base import train_main




# ---- patch only the bits that change --------------------------------------


env_cfg_patch = {
    "self_collision": False,
    "randomize_rot": True,
    "base_init_pos": [0.0, 0.0, 0.45],
    # "termination_if_roll_greater_than": 60,
    # "termination_if_pitch_greater_than": 90,
    "angle_termination_duration": 5.0, #seconds
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 0.9,
    "reward_scales": {
        "tracking_lin_vel": 7.0,
        "tracking_ang_vel": 2.5,
        "termination": -1.0,
        "alive": 1.0,
        "similar_to_default": -0.05,
        "dof_vel": -0.002,
        "dof_acc": -2e-6,
        "ang_vel_xy": -0.2,
        "feet_air_time": -0.05,
        "front_hip": -0.2,
        "rear_hip": -0.5,
        "relative_base_height": -0.1,
        "dof_pos_limits": -0.01, #fixed!
        "torque_limits": -2.0,
        "balance": -2e-5,
        "collision": -10.0, #fixed!
        "action_rate": -0.01,
        "feet_contact_forces": -0.001,
        "stand_still": -0.5,
        # "both_front_feet_airborne": -1.0,
        # "both_rear_feet_airborne": -1.0,
        "foot_xy_compact": -0.5 
    },
}



terrain_cfg_patch = {
    "terrain_type": "plane", #plane
}


command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_iteration_threshold": 2000,
    "mean_reward_threshold": 70,
    "lin_vel_x_range": [-0.5, 0.5],
    "lin_vel_y_range": [-0.5, 0.5],
    "ang_vel_range": [-1.0, 1.0],
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
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="go2_walking")
