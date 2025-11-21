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
    'max_push_vel_xy': 0.5,
    "episode_length_s": 30.0,
    "resampling_time_s": 10.0,
    "termination_if_relative_height_lower_than": 0.18,
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 1.0,
    "reward_scales": {
        "tracking_lin_vel": 1.5,
        "tracking_ang_vel": 0.75,
        "lin_vel_z": -2.0,
        "relative_base_height": -10.0,
        "orientation": -1.0, 
        "ang_vel_xy": -0.05,
        "collision": -5.0,
        "dof_pos_limits": -10.0, #fixed!
        "termination": -30.0,
        "feet_contact_forces": -0.0001,
        "default_pose_when_idle": -2.0,
        "feet_stumble": -0.5,
        # "alive": 0.3,
        "action_curvature": -0.1,
        "effort_symmetry": -0.1,
    },
}




terrain_cfg_patch = {
    "terrain_type": "trimesh", #plane
    "subterrain_size": 6.0,
    "horizontal_scale": 0.05,
    "vertical_scale": 0.005,
    "cols": 5,  #should be more than 5
    "rows": 5,   #should be more than 5
    "selected_terrains":{
        "flat_terrain" : {"probability": 0.5},
        "stamble_terrain" : {"probability": 0.1},
        "pyramid_sloped_terrain" : {"probability": 0.1},
        "discrete_obstacles_terrain" : {"probability": 0.1},
        "pyramid_shallow_down_stairs_terrain" : {"probability": 0.2},
    }
}

command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_iteration_threshold": 4000, #1 calculated 1 iteration is 1 seocnd 2000 = 
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
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="go2_walking")
