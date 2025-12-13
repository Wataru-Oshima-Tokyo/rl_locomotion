import argparse
import os
import pickle
from typing import TYPE_CHECKING

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import copy
import re

if TYPE_CHECKING:
    from legged_env import LeggedEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--export", action="store_true", help="Export policy artifacts (TorchScript).")
    parser.add_argument("--export-onnx", action="store_true", help="Export the loaded policy to ONNX (CPU).")
    parser.add_argument("--onnx-path", type=str, default=None, help="Override ONNX export path.")
    parser.add_argument("--onnx-opset", type=int, default=17, help="ONNX opset version to use.")
    args = parser.parse_args()

    gs.init()
    # Delayed import to ensure Genesis is initialized first.
    from legged_env import LeggedEnv

    log_dir = f"logs/{args.exp_name}"
    # Get all subdirectories in the base log directory
    subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

    # Sort subdirectories by their names (assuming they are timestamped in lexicographical order)
    most_recent_subdir = sorted(subdirs)[-1] if subdirs else None
    log_dir = os.path.join(log_dir, most_recent_subdir)
    env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, train_cfg, terrain_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    # train_cfg["policy"]["class_name"] = "ActorCritic"      # or "ActorCriticRecurrent"
    # train_cfg["algorithm"]["class_name"] = "PPO"          # ← add this line
    env_cfg["randomize_rot"] = True
    command_cfg["curriculum"] = False
    env = LeggedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        noise_cfg=noise_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        terrain_cfg=terrain_cfg,
        show_viewer_= False,
        eval_=True,
        show_camera_ = True
    )

    
    print(train_cfg)
    class_name = train_cfg["policy"]["class_name"]

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    # List all files in the most recent subdirectory
    files = os.listdir(log_dir)

    # Regex to match filenames like 'model_100.pt' and extract the number
    model_files = [(f, int(re.search(r'model_(\d+)\.pt', f).group(1)))
                for f in files if re.search(r'model_(\d+)\.pt', f)]
    model_file = max(model_files, key=lambda x: x[1])[0]

    resume_path = os.path.join(log_dir,  model_file)
    runner.load(resume_path)
    # resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    # runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    # export policy as a jit module (used to run it from C++)
    if args.export:
        # 保存先パスを作成
        save_dir = os.path.join(log_dir, 'exported', 'policies')
        os.makedirs(save_dir, exist_ok=True)
            
        if class_name == "ActorCritic":
            save_path = os.path.join(save_dir, 'policy_mlp.pt')
            model = copy.deepcopy(runner.alg.actor_critic.actor).to('cpu')
            traced_script_module = torch.jit.script(model)
            traced_script_module.save(save_path)
            print('Exported policy as jit script to: ', save_path)
            # Convert the policy to a version-less format
            versionless_path = os.path.join(save_dir, "policy_mlp_safe.pt")
            loaded_model = torch.jit.load(save_path)
            loaded_model.eval()
            loaded_model.save(versionless_path)
            print("Model successfully converted to version-less format: ", versionless_path)

        elif "ActorCriticRecurrent" in  class_name  and train_cfg["policy"]["rnn_type"] == "lstm":
            from rsl_rl.modules import InferenceActorLSTM, InferenceActorLSTMWrapper

            # Actor-Critic から MLP と RNN を取得
            actor_critic = runner.alg.actor_critic
            actor_mlp = actor_critic.actor
            rnn_module = actor_critic.memory_a.rnn

            # 推論用 Actor-LSTM を作成
            inference_actor = InferenceActorLSTMWrapper(
                inference_actor=InferenceActorLSTM(actor_mlp, rnn_module),
                num_envs=1,
                rnn_hidden_size=train_cfg["policy"]["rnn_hidden_size"])
            inference_actor.inference_actor.actor.load_state_dict(actor_critic.actor.state_dict())

            # RNN の state_dict を変換してロード ("rnn." プレフィックスを削除)
            rnn_state_dict = actor_critic.memory_a.state_dict()
            new_rnn_state_dict = {k.replace("rnn.", ""): v for k, v in rnn_state_dict.items()}
            inference_actor.inference_actor.lstm.load_state_dict(new_rnn_state_dict)

            save_path = os.path.join(save_dir, "actor_lstm_wrapper.pt")

            # TorchScript 化して保存
            scripted_actor = torch.jit.script(inference_actor)
            print("Succeeded to script inference_actor!")
            scripted_actor.save(save_path)
            print("Success to save policy!!")
        
        else:
            print("This version does not support this network architecture.")
            print("Could not save policy...")

    # Optional: export to ONNX for CPU-only inference (can be requested independently)
    if True:
        save_dir = os.path.join(log_dir, 'exported', 'policies')
        os.makedirs(save_dir, exist_ok=True)

        class OnnxPolicyWrapper(torch.nn.Module):
            def __init__(self, actor_module: torch.nn.Module, normalizer: torch.nn.Module):
                super().__init__()
                self.normalizer = normalizer
                self.actor = actor_module

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                obs = self.normalizer(obs)
                return self.actor(obs)

        # Deepcopy to detach from training runner and force everything to CPU
        actor_cpu = copy.deepcopy(runner.alg.actor_critic.actor).to("cpu").eval()
        normalizer_cpu = copy.deepcopy(runner.obs_normalizer).to("cpu").eval()
        wrapper = OnnxPolicyWrapper(actor_cpu, normalizer_cpu).cpu().eval()

        onnx_path = args.onnx_path or os.path.join(save_dir, "policy_mlp.onnx")
        dummy_obs = torch.zeros(1, obs_cfg["num_obs"], dtype=torch.float32, device="cpu")
        export_kwargs = dict(
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=args.onnx_opset,
            do_constant_folding=True,
        )
        try:
            torch.onnx.export(wrapper, dummy_obs, onnx_path, dynamo=True, **export_kwargs)
            print(f"Exported policy to ONNX with dynamo exporter: {onnx_path}")
        except ModuleNotFoundError as e:
            # onnxscript not installed; fall back to legacy exporter
            print(f"onnxscript not found ({e}); falling back to legacy torch.onnx.export without dynamo.")
            torch.onnx.export(wrapper, dummy_obs, onnx_path, dynamo=False, **export_kwargs)
            print(f"Exported policy to ONNX with legacy exporter: {onnx_path}")

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python legged_eval.py -e go2_walking --ckpt 100
"""
