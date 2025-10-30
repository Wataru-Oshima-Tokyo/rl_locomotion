import argparse
import os
import pickle
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from legged_env import LeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from datetime import datetime
import re
import copy
# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _deep_update(dst: dict, patch: dict):
    """Recursively merge *patch* into *dst* in-place."""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "symmetry_cfg":{
                "use_data_augmentation": False,
                "use_mirror_loss" : False,
            }
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic", #"ActorCriticRecurrent",
            # "rnn_type": "lstm",
            # "rnn_hidden_size": 32,
            # "rnn_num_layers": 1,
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 50,
        "empirical_normalization": None,
        "seed": 1,
        "mean_threshold": 20
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "self_collision": False,
        "use_mjcf": True,
        "robot_description": "xml/luvBit/luvBit.xml",
        'links_to_keep': ["imu_link"],
        "default_joint_angles": {  # [rad]
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": 0.0,
            "right_hip_pitch_joint": 0.0,

            "left_knee_joint": 0.0,
            "right_knee_joint": 0.0,

            "left_ankle_joint": 0.0,
            "right_ankle_joint": 0.0,

            "neck_pitch_joint": 0.0,
            "neck_yaw_joint": 0.0,
        },
        # "default_joint_angles": {  # [rad]
        #     "left_hip_yaw_joint": 0.0,
        #     "right_hip_yaw_joint": 0.0,
        #     "left_hip_roll_joint": -0.1,
        #     "right_hip_roll_joint": 0.1,
        #     "left_hip_pitch_joint": 0.4,
        #     "right_hip_pitch_joint": -0.4,

        #     "left_knee_joint": -0.8,
        #     "right_knee_joint": 0.8,

        #     "left_ankle_joint": 0.4,
        #     "right_ankle_joint": -0.4,

        #     "neck_pitch_joint": 0.0,
        #     "neck_yaw_joint": 0.0,
        # },
        "dof_names": [ #order matters!
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_joint",
            "neck_pitch_joint",
            "neck_yaw_joint",
        ],
        "dof_lower_limit": [ #order matters!
            -0.3, -0.35, -2.3, -0.8, -0.8, #Left (hip_yaw, hip_roll, hip_pitch, knee, ankle)
            -0.2, -0.25, -1.5, -2.5, -1.0, #Right (hip_yaw, hip_roll, hip_pitch, knee, ankle)
            -0.6, -0.2  #Head (neck, head)
        ],
        "dof_upper_limit": [ #order matters!
            0.2, 0.25, 1.5, 2.5, 1.0, #Left (hip_yaw, hip_roll, hip_pitch, knee, ankle)
            0.3, 0.35, 2.3, 0.8, 0.8, #Right (hip_yaw, hip_roll, hip_pitch, knee, ankle)
            1.5, 0.2  #Head (neck, head)
        ],
        'PD_stiffness': {'hip_yaw':   10.0,
                         'hip_roll':   10.0,
                         'hip_pitch': 10.0,
                         'knee': 10.0,
                         'ankle': 10.0,
                         'neck': 10.0},
        'PD_damping': { 'hip_yaw':    0.3,
                        'hip_roll': 0.3,
                        'hip_pitch':  0.3,
                        'knee': 0.3,
                        'ankle': 0.3,
                        'neck': 0.3},
        'force_limit': {'hip':    23.7,
                        'knee':  23.7,
                        'ankle':   23.7,
                        'neck':  23.7},
        # termination
        'termination_contact_link_names': ['base_link', 'neck', 'head'],
        'penalized_contact_link_names': ['base_link', 'knee', 'neck', 'head'],
        'calf_link_name': ['knee'],
        'feet_link_name': ['ankle'],
        'thigh_link_name': ['knee'],
        'base_link_name': ['base_link'], 
        "hip_joint_names": [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            # "left_hip_pitch_joint",
            "right_hip_yaw_joint",            
            "right_hip_roll_joint",
            # "right_hip_pitch_joint",
        ],
        "thigh_joint_names": [
            "left_knee_joint",
            "right_knee_joint",     
        ],
        "termination_if_roll_greater_than": 30,  # degree. 
        "termination_if_pitch_greater_than": 30,
        "termination_if_relative_height_lower_than": 0.15,
        "termination_duration": 1.0, #seconds
        "angle_termination_duration": 5.0, #seconds
        # base pose
        "base_init_pos": [0.0, 0.0, 0.36],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": False,
        'send_timeouts': True,
        "clip_actions": 100.0,
        'control_freq': 50,
        'decimation': 5, #5
        'low_level_control': {
            'enabled': True,
            'first_order_hold': True,
            'cutoff_freq_hz': 35.0,
            'max_joint_velocity': 10.0,
        },
        # random push
        'push_interval_s': 10,
        'max_push_vel_xy': 1.0,
        # domain randomization
        'randomize_delay': True,
        'delay_range': [0.015, 0.03], #seconds        
        'motor_randomize_friction': True,
        'motor_friction_range': [0.05, 1.5],
        'foot_randomize_friction': True,
        'foot_friction_range': [1.5, 2.5],
        'randomize_base_mass': True,
        'added_mass_range': [-1., 3.],
        'randomize_com_displacement': True,
        'com_displacement_range': [-0.01, 0.01],
        'randomize_motor_strength': True,
        'motor_strength_range': [0.9, 1.1],
        'randomize_motor_offset': True,
        'motor_offset_range': [-0.02, 0.02],
        'randomize_kp_scale': True,
        'kp_scale_range': [0.98, 1.02],
        'randomize_kd_scale': True,
        'kd_scale_range': [0.98, 1.02],
        "randomize_rot": True,
        "pitch_range": [-15, 15],  # degrees
        "roll_range": [-15, 15],
        "yaw_range": [-180, 180],
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_components": [
            "base_ang_vel",        # 3
            "projected_gravity",   # 3
            "commands",            # 3
            "dof_pos_scaled",      # 12
            "dof_vel_scaled",      # 12
            "actions"              # 12
        ],
        "num_privileged_obs": 50,
        "privileged_obs_components": [
            "base_lin_vel",        # 3
            "base_ang_vel",        # 3
            "projected_gravity",   # 3
            "commands",            # 3
            "dof_pos_scaled",      # 12
            "dof_vel_scaled",      # 12
            "actions",             # 12
            "foot_friction",       # 2
            # "sin_phase",           # 4
            # "cos_phase",           # 4
            # "collision"            # 4
        ],
        "mirror_func": {
            "base_lin_vel": "mirror_linear",
            "base_ang_vel": "mirror_angle",
            "projected_gravity": "mirror_linear",
            "commands": "mirror_commands",
            "dof_pos_scaled": "mirror_go_joint",
            "dof_vel_scaled": "mirror_go_joint",
            "torques_scaled": "mirror_go_joint",
            "actions": "mirror_go_joint",
            "sin_phase": "mirror_go_foot", 
            "cos_phase": "mirror_go_foot",  
            "collision": "mirror_go_foot",
        },
        "obs_scales": {
            "base_lin_vel": 2.0,
            "base_ang_vel": 0.25,
            "dof_pos_scaled": 1.0,
            "dof_vel_scaled": 0.05,
            "torques_scaled": 0.03,
        },
        "clip_observations":100,
    }

    reward_cfg = {
        "tracking_min_sigma": 0.1,
        "tracking_max_sigma": 0.25,
        "base_height_target": 0.36,
        "relative_base_height_target": 0.36,
        "step_period": 0.8, #0.8
        "step_offset": 0.5, #0.5
        "front_feet_relative_height": 0.15,
        "rear_feet_relative_height": 0.15,
        "foot_clearance_height_target": -0.22,
        "calf_clearance_height_target": -0.12,
        "soft_dof_pos_limit": 0.9,
        "soft_torque_limit": 1.0,
        "only_positive_rewards": True,
        "max_contact_force": 200,
        "reward_scales": {},
    }
    command_cfg = {
        "num_commands": 3,
        "curriculum": False,
        "curriculum_iteration_threshold": 0, #1 calculated 1 iteration is 1 seocnd 2000 = 
        "mean_reward_threshold": 20,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
    }
    noise_cfg = {
        "add_noise": True,
        "noise_level": 1.0,
        "noise_scales":{
            "dof_pos": 0.01,
            "dof_vel": 1.5,
            "lin_vel": 0.5,
            "ang_vel": 0.2,
            "gravity": 0.05,
            "torques": 0.5,
        }
    }
    terrain_cfg = {
        "terrain_type": "plane", #plane, trimesh, custom_plane
        "subterrain_size": 4.0,
        "horizontal_scale": 0.05,
        "vertical_scale": 0.005,
        "cols": 5,  #should be more than 5
        "rows": 5,   #should be more than 5
        "selected_terrains":{
            # "flat_terrain" : {"probability": 0.2},
            # # "blocky_terrain" : {"probability": 0.2},
            # "stamble_terrain" : {"probability": 0.2},
            # "discrete_obstacles_terrain" : {"probability": 0.1},
            # "pyramid_stairs_terrain" : {"probability": 0.2},
        }
    }

    return env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg


def train_main(
    cfg_patches=None,
    default_exp_name: str = "luvBit_walking",
    default_iterations: int = 10000,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default=default_exp_name)
    parser.add_argument("-B", "--num_envs", type=int, default=10000) #10000 8192
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if this flag is set")
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--vis", action="store_true", help="If you would like to see how robot is trained")
    parser.add_argument("--wandb_username", type=str, default="wataru-oshima-techshare")
    args = parser.parse_args()

    gs.init(logging_level="warning")
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir_ = os.path.join(BASE_DIR, "logs", args.exp_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir_, timestamp)
    # ------------ build base cfgs & apply patches --------------------------
    env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg = map(
        copy.deepcopy, get_cfgs()
    )

    cfgs = [
        env_cfg,
        obs_cfg,
        noise_cfg,
        reward_cfg,
        command_cfg,
        terrain_cfg,
    ]
    if cfg_patches:
        for dst, patch in zip(cfgs, cfg_patches):
            _deep_update(dst, patch)

    # ------------ train-cfg -----------------------------------------------
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    env_cfg["mirror"] = train_cfg["algorithm"]["symmetry_cfg"]["use_data_augmentation"] or train_cfg["algorithm"]["symmetry_cfg"]["use_mirror_loss"]

    if not env_cfg["mirror"]:
        train_cfg["algorithm"]["symmetry_cfg"] = None
    else:
        from rsl_rl.utils.mirror import mirror_data_augmentation
        train_cfg["algorithm"]["symmetry_cfg"]["data_augmentation_func"] = mirror_data_augmentation
        train_cfg["algorithm"]["symmetry_cfg"]["mirror_loss_coeff"] = 1.0

    
    # pickle
    train_cfg_to_save = copy.deepcopy(train_cfg)
    train_cfg_to_save["algorithm"]["symmetry_cfg"] = None

    env = LeggedEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        noise_cfg=noise_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,        
        terrain_cfg=terrain_cfg,        
        show_camera_=args.vis,
    )

    if args.resume:
        # Get all subdirectories in the base log directory
        subdirs = [d for d in os.listdir(log_dir_) if os.path.isdir(os.path.join(log_dir_, d))]

        # Sort subdirectories by their names (assuming they are timestamped in lexicographical order)
        most_recent_subdir = sorted(subdirs)[-1] if subdirs else None
        most_recent_path = os.path.join(log_dir_, most_recent_subdir)

        if args.ckpt == 0:
            # List all files in the most recent subdirectory
            files = os.listdir(most_recent_path)

            # Regex to match filenames like 'model_100.pt' and extract the number
            model_files = [(f, int(re.search(r'model_(\d+)\.pt', f).group(1)))
                        for f in files if re.search(r'model_(\d+)\.pt', f)]
            model_file = max(model_files, key=lambda x: x[1])[0]
        else:
            model_file = f"model_{args.ckpt}.pt"
        resume_path = os.path.join(most_recent_path,  model_file)

    os.makedirs(log_dir, exist_ok=True)        
    wand_project_name = 'ts_genesis'
    train_cfg.update(
        logger="tensorboard",
        record_interval=50,
        # user_name=args.wandb_username,
        # wandb_project=wand_project_name,
        run_name=args.exp_name,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    if args.resume:
        runner.load(resume_path)

    # wandb.init(project=wand_project_name, name=args.exp_name, dir=log_dir, mode='offline' if args.offline else 'online')

    pickle.dump(
        [env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, train_cfg_to_save, terrain_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )


    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


# if __name__ == "__main__":
#     main()
