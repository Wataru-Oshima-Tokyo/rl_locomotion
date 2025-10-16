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
    "base_init_pos": [0.0, 0.0, 0.4],
    "angle_termination_duration": 1.0, #seconds
    "dof_lower_limit": [ #order matters!
        -0.2, -0.35, -2.3, -0.8, -0.8, #Left (hip_yaw, hip_roll, hip_pitch, knee, ankle)
        -0.3, -0.25, -1.5, -2.5, -1.0, #Right (hip_yaw, hip_roll, hip_pitch, knee, ankle)
        -0.6, -1.5  #Head (neck, head)
    ],
    "dof_upper_limit": [ #order matters!
        0.3, 0.25, 1.5, 2.5, 1.0, #Left (hip_yaw, hip_roll, hip_pitch, knee, ankle)
        0.2, 0.35, 2.3, 0.8, 0.8, #Right (hip_yaw, hip_roll, hip_pitch, knee, ankle)
        1.5, 1.5  #Head (neck, head)
    ],
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 1.0,
    "reward_scales": {
        "tracking_lin_vel": 1.5,
        "tracking_ang_vel": 0.75,
        "lin_vel_z": -5.0,
        "relative_base_height": -10.0,
        "orientation": -30.0, #fixed!
        "ang_vel_xy": -0.05, #fixed!
        "collision": -10.0, #fixed!
        "front_feet_clearance": 30.0,
        "action_rate": -0.01,
        "dof_acc": -2.5e-6,
        "dof_pos_limits": -10.0, #fixed!
        "powers": -2e-5,
        "termination": -30.0,
        "contact_no_vel": -0.2,
        "contact": 0.1,
        "feet_contact_forces": -0.001,
        "stand_still": -0.5,
    },
}



terrain_cfg_patch = {
    "terrain_type": "plane", #plane
}


command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_iteration_threshold": 2000,
    "mean_reward_threshold": 20,
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
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="luvBit_walking", default_iterations=5000)
