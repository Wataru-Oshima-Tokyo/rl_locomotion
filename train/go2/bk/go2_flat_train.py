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
# import wandb

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
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
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
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "self_collision": True,
        "use_mjcf": True,
        "robot_description": "xml/go2/go2.xml",
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
        'penalized_contact_link_names': ['base', 'thigh', 'calf'],
        'calf_link_name': ['calf'],
        'feet_link_name': ['foot'],
        'thigh_link_name': ['thigh'],
        'base_link_name': ['base'], 
        "hip_joint_names": [
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",            
        ],
        "termination_if_roll_greater_than": 170,  # degree. 
        "termination_if_pitch_greater_than": 180,
        "termination_if_height_lower_than": -40,
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
        'randomize_friction': True,
        'friction_range': [0.05, 4.5],
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
        "randomize_rot": False,
        "pitch_range": [-40, 40],  # degrees
        "roll_range": [-50, 50],
        "yaw_range": [-180, 180],
    }
    obs_cfg = {
        "num_obs": 45,
        "num_privileged_obs": 60,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "torques": 0.03,
        },
        "clip_observations":100,
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.32,
        "relative_base_height_target": 0.32,
        "step_period": 1.0, #0.8
        "step_offset": 0.5, #0.5
        "front_feet_relative_height": 0.15,
        "rear_feet_relative_height": 0.15,
        "clearance_height_target": -0.22,
        "soft_dof_pos_limit": 0.9,
        "soft_torque_limit": 1.0,
        "only_positive_rewards": True,
        "max_contact_force": 200,
        "reward_scales": {
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.75,
            "lin_vel_z": -2.0, #-5.0
            "relative_base_height": -30.0, # -30.0
            "orientation": -30.0,
            "ang_vel_xy": -0.05,
            "collision": -5.0,
            # "roll_penalty": -0.1,
            # "front_leg_retraction": -1.0,
            # "thigh_retraction": 1.0,
            "foot_clearance": -0.5,
            # "front_feet_clearance": 10.0,
            # "rear_feet_clearance": 10.0,
            "action_rate": -0.01,
            # "rear_feet_level_with_front": 1.0,
            # "hip_pos": -.1, #-1.0
            # "contact_no_vel": -0.2,
            "dof_acc": -2.5e-7,
            # "contact": 0.01,
            "dof_pos_limits": -10.0,
            "dof_vel": 0.0,
            "torques": 0.0,
            'powers': -2e-5,
            "termination": -30.0,
            # "base_upward_progress": 2.0,
            # "calf_collision_low_clearance": -5.0,
            "similar_to_default": -0.01,
            "feet_contact_forces": -0.001,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "curriculum": False,
        "curriculum_duration": 0, #1 calculated 1 iteration is 1 seocnd 2000 = 
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
            "lin_vel": 0.2,
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
            "flat_terrain" : {"probability": 0.2},
            # "blocky_terrain" : {"probability": 0.2},
            "stamble_terrain" : {"probability": 0.2},
            "discrete_obstacles_terrain" : {"probability": 0.1},
            "pyramid_stairs_terrain" : {"probability": 0.2},
        }
    }

    return env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096) #10000
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
    env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    
    env = LeggedEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        noise_cfg=noise_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,        
        terrain_cfg=terrain_cfg,        
        show_viewer=args.vis,

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
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    if args.resume:
        runner.load(resume_path)

    wand_project_name = 'ts_genesis'
    # wandb.init(project=wand_project_name, name=args.exp_name, dir=log_dir, mode='offline' if args.offline else 'online')

    pickle.dump(
        [env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, train_cfg, terrain_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )


    train_cfg.update(
        logger="wandb",
        record_interval=50,
        user_name=args.wandb_username,
        wandb_project=wand_project_name,
        run_name=args.exp_name,
    )
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()