"""
go1_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from go1_base import train_main




# ---- patch only the bits that change --------------------------------------


env_cfg_patch = {
    "self_collision": True,
    "randomize_rot": False,
    "penalized_contact_link_names": ["base", "thigh"],
    "base_init_pos": [0.0, 0.0, 0.45],
    'max_push_vel_xy': 1.0,
}

reward_cfg_patch = {
    "reward_scales": {
        "tracking_lin_vel": 1.5,
        "tracking_ang_vel": 0.75,
        "lin_vel_z": -2.0,
        "relative_base_height": -30.0,
        "orientation": -30.0,
        "ang_vel_xy": -0.05,
        "collision": -5.0,
        "foot_clearance": -0.5,
        "action_rate": -0.01,
        "dof_acc": -2.5e-7,
        "dof_pos_limits": -10.0,
        "dof_vel": 0.0,
        "torques": 0.0,
        "powers": -2e-5,
        "termination": -30.0,
        "similar_to_default": -0.01,
        "feet_contact_forces": -0.001,
    },
}



terrain_patch = {
    "terrain_type": "plane",
}

command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_duration": 0, #1 calculated 1 iteration is 1 seocnd 2000 = 
    "lin_vel_x_range": [-0, 0],
    "lin_vel_y_range": [-0, 0],
    "ang_vel_range": [-0, 0],
}



# leave other five cfgs untouched
CFG_PATCHES = (
    env_cfg_patch,  # env_cfg
    {},  # obs_cfg
    {},  # noise_cfg
    reward_cfg_patch,  # reward_cfg
    {},  # command_cfg
    terrain_patch,
)

if __name__ == "__main__":
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="go1_walking")
