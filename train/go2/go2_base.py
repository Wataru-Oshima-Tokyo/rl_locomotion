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
from torch.utils.tensorboard import SummaryWriter

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
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 1.0,
            "class_name": "ActorCriticRecurrent", #"ActorCriticRecurrent", "ActorCriticRecurrentMoE"
            "rnn_type": "lstm",
            "rnn_hidden_size": 32, #256
            "rnn_num_layers": 1,
            "moe":{
                "num_experts": 6,
                "gate_hidden": 128,
                "expert_hidden": 256,   # 各Expertの中間幅
                "log_std_init": -0.5,     # 各Expertの初期logstd
                "gate_entropy_coef": 0.001,   # ゲートのエントロピー正則化
                "prefer_base_expert_coef": 0.0 # 平地で特定expertを優遇する場合に使用(最初は0)
            },

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
        "robot_description": "xml/go2/go2.xml",
        # "robot_description": "urdf/go2/urdf/go2.urdf",
        'links_to_keep': ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',],
        "default_joint_angles": {  # [rad]
            "FR_hip_joint": -0.1,
            "FL_hip_joint": 0.1,
            "RR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,

            "FR_thigh_joint": 0.8,
            "FL_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "RL_thigh_joint": 1.0,

            "FR_calf_joint": -1.5,
            "FL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
        },
        "dof_names": [ #order matters!
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "dof_lower_limit": [ #order matters!
            -0.8, 0.1, -2.5,
            -0.5, 0.1, -2.5,
            -0.8, 0.2,  -2.5,
            -0.5, 0.2,  -2.5,
        ],
        "dof_upper_limit": [ #order matters!
            0.5, 2.6, -0.8,
            0.8, 2.6, -0.8,
            0.5, 2.2, -1.3,
            0.8, 2.2, -1.3,
        ],
        'PD_stiffness': {'hip':   40.0,
                         'thigh': 40.0,
                          'calf': 40.0},
        'PD_damping': {'hip':    1.0,
                        'thigh': 1.0,
                        'calf':  1.0},
        'force_limit': {'hip':    23.7,
                        'thigh':  23.7,
                        'calf':   45.43},
        # termination
        'termination_contact_link_names': ['base'],
        'penalized_contact_link_names': ['base', "hip", 'thigh', 'bumper' ],
        'calf_link_name': ['calf'],
        'feet_link_name': ['foot'],
        'thigh_link_name': ['thigh'],
        'base_link_name': ['base'], 
        "hip_joint_names": [
            "FR_hip_joint",
            "FL_hip_joint",
            "RR_hip_joint",            
            "RL_hip_joint",
        ],
        "thigh_joint_names": [
            "FR_thigh_joint",
            "FL_thigh_joint",
            "RR_thigh_joint",
            "RL_thigh_joint",            
        ],
        "body_half_length": 0.35,
        "body_half_width": 0.16,
        "termination_if_roll_greater_than": 170,  # degree. 
        "termination_if_pitch_greater_than": 180,
        "termination_if_relative_height_lower_than": 0.15,
        "termination_duration": 1.0, #seconds
        "angle_termination_duration": 5.0, #seconds
        # base pose
        "base_init_pos": [0.0, 0.0, 0.55],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        'send_timeouts': True,
        "clip_actions": 100.0,
        'control_freq': 50,
        'decimation': 4,
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
        "pitch_range": [-30, 30],  # degrees
        "roll_range": [-30, 30],
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
        "num_privileged_obs": 52,
        "privileged_obs_components": [
            "base_lin_vel",        # 3
            "base_ang_vel",        # 3
            "projected_gravity",   # 3
            "commands",            # 3
            "dof_pos_scaled",      # 12
            "dof_vel_scaled",      # 12
            "actions",             # 12
            "foot_friction",       # 4
            # "sin_phase",           # 4
            # "cos_phase",           # 4
            # "collision"            # 4
        ],
        "mirror_func": {
            "base_lin_vel": "mirror_linear",
            "base_ang_vel": "mirror_angle",
            "projected_gravity": "mirror_linear",
            "commands": "mirror_commands",
            "dof_pos_scaled": "mirror_go_leg_joint",
            "dof_vel_scaled": "mirror_go_leg_joint",
            "torques_scaled": "mirror_go_leg_joint",
            "actions": "mirror_go_leg_joint",
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
        "tracking_min_sigma": 0.05,
        "tracking_max_sigma": 0.25,
        "base_height_target": 0.36,
        "relative_base_height_target": 0.36,
        "step_period": 1.0, #0.8
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
            "base_lin_vel": 0.5,
            "base_ang_vel": 0.2,
            "projected_gravity": 0.05,
            "dof_pos_scaled": 0.01,
            "dof_vel_scaled": 1.5,
            "torques_scaled": 0.5,
        }
    }
    terrain_cfg = {
        "terrain_type": "plane", #plane, trimesh, custom_plane
        "subterrain_size": 4.0,
        "horizontal_scale": 0.05,
        "vertical_scale": 0.005,
        "step_height": 0.075, # this is for custom plane
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
    default_exp_name: str = "go2_walking",
):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default=default_exp_name)
    parser.add_argument("-B", "--num_envs", type=int, default=8192) #10000 8192
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if this flag is set")
    parser.add_argument("--show_viewer", action="store_true", help="Wether to visualize simulation env")
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
        show_viewer_ = args.show_viewer,
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
    # runner.tb_writer = SummaryWriter(log_dir)
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