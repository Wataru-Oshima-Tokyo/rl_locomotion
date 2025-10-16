"""
b2_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from b2_base import train_main

# ---- patch only the bits that change --------------------------------------

env_cfg_patch = {
    "self_collision": False,
    "randomize_rot": True,
    "base_init_pos": [0.0, 0.0, 0.3],
    'max_push_vel_xy': 0.0,
    "episode_length_s": 30.0,
    "resampling_time_s": 10.0,
    "termination_if_relative_height_lower_than": 0.15,
    'penalized_contact_link_names': ['base', 'thigh'],

}

reward_cfg_patch = {
    "soft_dof_pos_limit": 1.0,
    "reward_scales": {
        "tracking_lin_vel": 1.5,
        "tracking_ang_vel": 0.75,
        "lin_vel_z": -2.0,
        "relative_base_height": -5.0,
        "orientation": -0.01,
        "ang_vel_xy": -0.05,
        "collision": -2.0,
        "front_feet_clearance": 30.0,
        "rear_feet_clearance": 30.0,
        "action_rate": -0.01,
        "dof_acc": -2.5e-7,
        "dof_pos_limits": -20.0,
        "powers": -2e-5,
        "termination": -30.0,
        "both_front_feet_airborne": -1.0,
        "both_rear_feet_airborne": -1.0,
        "contact_no_vel": -0.2,
        "feet_contact_forces": -0.0001,
        "stand_still": -0.5,
        "swing_stuck": -10.0
    },
}



terrain_cfg_patch = {
    "terrain_type": "trimesh", #plane
    "subterrain_size": 6.0,
    "horizontal_scale": 0.25,
    "vertical_scale": 0.005,
    "cols": 5,  #should be more than 5
    "rows": 5,   #should be more than 5
    "selected_terrains":{
        "flat_terrain" : {"probability": 0.1},
        # "stamble_terrain" : {"probability": 0.1},
        # "pyramid_sloped_terrain" : {"probability": 0.1},
        # "discrete_obstacles_terrain" : {"probability": 0.1},
        # "pyramid_shallow_down_stairs_terrain" : {"probability": 0.1},
        # "pyramid_down_stairs_terrain" : {"probability": 0.2},
    }
}

command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_iteration_threshold": 2000, #1 calculated 1 iteration is 1 seocnd 2000 = 
    "mean_reward_threshold": 30,
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
