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
    'max_push_vel_xy': 0.5,
    "episode_length_s": 30.0,
    "resampling_time_s": 10.0,
    "pitch_range": [-180, 180],  # degrees
    "roll_range": [-180, 180],
    "termination_if_relative_height_lower_than": 0.18,
    "termination_duration": 5.0,
    # 'foot_randomize_friction': True,
    # 'foot_friction_range': [4.5, 5.5]
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 0.9,
    "reward_scales": {
        "tracking_lin_vel": 1.5,
        "untracking_lin_vel_x": -10.0,
        "tracking_ang_vel": 0.75,
        "untracking_ang_vel": -3.0,
        "lin_vel_z": -2.0,
        "relative_base_height": -10.0,
        "orientation": -0.1, #fixed!
        "ang_vel_xy": -0.05, #fixed!
        "collision": -1.0, #fixed!
        "head_collision": -10.0, #fixed!
        "dof_pos_limits": -5.0, #fixed!
        "termination": -30.0,
        "dof_acc": -2.5e-7,
        "feet_contact_forces": -0.0001,
        "default_pose_when_idle": -0.5,
        "feet_stumble": -0.5,
        "action_curvature": -0.02,
        "effort_symmetry": -0.01,
        "leg_cross": -1.0,
        "leg_cross_fore_aft": -1.0,
        "stuck_ema": -0.1,
        "roll_penalty": -0.1,
        "goal_reached": 1.0,
        "both_front_feet_airborne": -0.5,
        "idle_leg_raise": -5.0,
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
        "flat_terrain" : {"probability": 0.2},
        "stamble_terrain" : {"probability": 0.2},
        "pyramid_sloped_terrain" : {"probability": 0.1},
        "pyramid_shallow_up_stairs_terrain" : {"probability": 0.2},
        "pyramid_up_stairs_terrain" : {"probability": 0.3},
    }
}

command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_iteration_threshold": 4000, #1 calculated 1 iteration is 1 seocnd 2000 = 
    "mean_reward_threshold": 20,
    "goal_probability": 0.5,
    "enable_stop_commands": True,
    "stop_command_probability": 0.5,
    "lin_vel_x_range": [-1.5, 1.5],
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
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="b2_walking", default_max_iterations=5000)
