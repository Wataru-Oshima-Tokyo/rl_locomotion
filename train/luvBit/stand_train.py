"""
go2_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from luvBit_base import train_main




# ---- patch only the bits that change --------------------------------------


env_cfg_patch = {
    "self_collision": False,
    "randomize_rot": True,
    "base_init_pos": [0.0, 0.0, 0.36],
    # "termination_if_roll_greater_than": 30,
    "max_push_vel_xy": 0.0,
    "angle_termination_duration": 1.0, #seconds
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 1.0,
    "reward_scales": {
        "lin_vel_z": -5.0,
        # "relative_base_height": -30.0,
        "orientation": -30.0, #fixed!
        "ang_vel_xy": -0.05, #fixed!
        "collision": -10.0, #fixed!
        "action_rate": -0.01,
        "dof_acc": -2.5e-7,
        "dof_pos_limits": -10.0, #fixed!
        "powers": -2e-5,
        "termination": -30.0,
        "alive": 5.0,
        # "contact": 0.1,
        "feet_contact_forces": -0.001,
        "default_pose_when_idle": -5.0,
        "similar_to_default": -1.0,
    },
}



terrain_cfg_patch = {
    "terrain_type": "plane", #plane
}


command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_iteration_threshold": 2000,
    "mean_reward_threshold": 30,
    "lin_vel_x_range": [0.0, 0.0],
    "lin_vel_y_range": [0.0, 0.0],
    "ang_vel_range": [0.0, 0.0],
    # "lin_vel_x_range": [-0.0, 0.0],
    # "lin_vel_y_range": [-0.0, 0.0],
    # "ang_vel_range": [-0.0, 0.0],
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
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="luvBit_walking", default_iterations=2000)
