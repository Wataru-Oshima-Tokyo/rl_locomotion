"""
go2_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from luvBit_base import train_main

# ---- patch only the bits that change --------------------------------------

env_cfg_patch = {
    "self_collision": False,
    "randomize_rot": False,
    "base_init_pos": [0.0, 0.0, 0.36],
    "episode_length_s": 30.0,
    "resampling_time_s": 10.0,
    "angle_termination_duration": 1.0,
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 0.9,
    "reward_scales": {
        "tracking_lin_vel": 1.5,
        "tracking_ang_vel": 0.75,
        "lin_vel_z": -1.0,
        "relative_base_height": -10.0,
        "orientation": -5.0, #fixed!
        "ang_vel_xy": -0.05, #fixed!
        "collision": -5.0, #fixed!
        "front_feet_clearance": 50.0,
        "action_rate": -0.01,
        "dof_acc": -2.5e-6,
        "dof_pos_limits": -10.0,
        "powers": -2e-5,
        "termination": -30.0,
        "feet_contact_forces": -0.001,
        "default_pose_when_idle": -1.0,
        "similar_to_default": -0.1,
    },
}



terrain_cfg_patch = {
    "terrain_type": "trimesh", #plane
    "subterrain_size": 6.0,
    "horizontal_scale": 0.05,
    "vertical_scale": 0.005,
    "cols": 6,  #should be more than 5
    "rows": 6,   #should be more than 5
    "selected_terrains":{
        "flat_terrain" : {"probability": 0.3},
        "shallow_blocky_terrain" : {"probability": 0.1},
        "pyramid_sloped_terrain" : {"probability": 0.1},
        "pyramid_down_sloped_terrain" : {"probability": 0.1},
        # "pyramid_shallow_down_stairs_terrain" : {"probability": 0.1},
        # "pyramid_down_stairs_terrain" : {"probability": 0.2},
    }
}

command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_iteration_threshold": 2000, #1 calculated 1 iteration is 1 seocnd 2000 = 
    "mean_reward_threshold": 20,
    "lin_vel_x_range": [-0.8, 0.8],
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
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="luvBit_walking")
