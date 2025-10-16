"""
b2_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from b2_base import train_main




# ---- patch only the bits that change --------------------------------------


env_cfg_patch = {
    "self_collision": False,
    "randomize_rot": False,
    "base_init_pos": [0.0, 0.0, 0.55],
    "termination_if_roll_greater_than": 50,
    "termination_if_pitch_greater_than": 90,
    "angle_termination_duration": 2.0, #seconds
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 0.9,
    "reward_scales": {
        "tracking_lin_vel": 1.5,
        "tracking_ang_vel": 0.75,
        # "lin_vel_z": -5.0,
        # "relative_base_height": -30.0,
        # "orientation": -1.0, #fixed!
        # "ang_vel_xy": -0.05, #fixed!
        "collision": -30.0, #fixed!
        "bumper_collision": -50.0,
        "action_rate": -0.01,
        "dof_acc": -2.5e-8,
        "dof_pos_limits": -10.0, #fixed!
        "powers": -2e-6,
        "termination": -30.0,
        "front_hip": -0.2,
        "rear_hip": -0.5,
        "contact": 0.1,
        "feet_air_time": 1.0,
        # "front_feet_clearance": 10.0,
        # "rear_feet_clearance": 5.0,
        # "alive": 1.0,
        # "both_front_feet_airborne": -1.0,
        # "both_rear_feet_airborne": -1.0,
        # "feet_contact_forces": -0.0001,
        "default_pose_when_idle": -2.0,
        # "feet_stumble": -3.0,
        # "similar_to_default": -0.1,
    },
}



terrain_cfg_patch = {
    "terrain_type": "plane", #plane
}


command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_iteration_threshold": 2000,
    "mean_reward_threshold": 60,
    "lin_vel_x_range": [-1.0, 1.0],
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
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="b2_walking")
