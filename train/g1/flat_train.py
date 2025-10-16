"""
go2_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from g1_base import train_main




# ---- patch only the bits that change --------------------------------------


env_cfg_patch = {
    "self_collision": False,
    "randomize_rot": False,
    # "base_init_pos": [0.0, 0.0, 0.45],
    "termination_if_roll_greater_than": 45,
    "termination_if_pitch_greater_than": 45,
    "angle_termination_duration": 0.00001, #seconds
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 1.0,
    "reward_scales": {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -1.0,
        "relative_base_height": -10.0,
        "dof_acc": -2.5e-7,
        "dof_vel": -1e-3, # ==================================
        "action_rate": -0.01,
        "dof_pos_limits": -5.0,
        "alive": 0.15,
        "hip_pos": -1.0,  # ================================
        "contact_no_vel": -0.2,
        "feet_swing_height": -20.0,
        "contact": 0.18
    },
}



terrain_cfg_patch = {
    "terrain_type": "plane", #plane
}


command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_duration": 2000, #1 calculated 1 iteration is 1 seocnd 2000 =
    "mean_reward_threshold": 15,
    "lin_vel_x_range": [-1.0, 1.0],
    "lin_vel_y_range": [-0.5, 0.5],
    "ang_vel_range": [-0.5, 0.5],
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
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="g1_walking")


"""
# training
cd ts_locomotion/train/g1
python flat_train.py  -B 8192 --vis
"""
