import argparse
import os
import pickle
import re
import copy
from typing import Tuple

import torch
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticRecurrentMoE,
    EmpiricalNormalization,
)


def _resolve_log_dir(exp_name: str, log_root: str) -> str:
    base_dir = os.path.join(log_root, exp_name)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Log root not found: {base_dir}. Use --log-root to point to your logs directory.")

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found under {base_dir}")
    return os.path.join(base_dir, sorted(subdirs)[-1])


def _resolve_ckpt_path(log_dir: str, ckpt: int | None) -> str:
    if ckpt is not None:
        path = os.path.join(log_dir, f"model_{ckpt}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Requested checkpoint not found: {path}")
        return path

    files = os.listdir(log_dir)
    model_files = [
        (f, int(match.group(1)))
        for f in files
        if (match := re.search(r"model_(\d+)\.pt", f)) is not None
    ]
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt files found in {log_dir}")
    latest = max(model_files, key=lambda x: x[1])[0]
    return os.path.join(log_dir, latest)


def _build_policy(train_cfg: dict, obs_cfg: dict, env_cfg: dict, device: torch.device):
    cls_map = {
        "ActorCritic": ActorCritic,
        "ActorCriticRecurrent": ActorCriticRecurrent,
        "ActorCriticRecurrentMoE": ActorCriticRecurrentMoE,
    }
    policy_cfg = copy.deepcopy(train_cfg["policy"])
    class_name = policy_cfg.pop("class_name")
    actor_cls = cls_map.get(class_name)
    if actor_cls is None:
        raise ValueError(f"Unsupported policy class: {class_name}")

    num_obs = obs_cfg["num_obs"]
    num_critic_obs = obs_cfg.get("num_privileged_obs", 0) or num_obs
    num_actions = env_cfg["num_actions"]

    actor_critic = actor_cls(num_obs, num_critic_obs, num_actions, **policy_cfg).to(device)
    if train_cfg.get("empirical_normalization", False):
        normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(device)
    else:
        normalizer = torch.nn.Identity().to(device)
    return actor_critic, normalizer


def _export_onnx(
    actor_module: torch.nn.Module,
    normalizer: torch.nn.Module,
    obs_dim: int,
    onnx_path: str,
    opset: int,
):
    class OnnxPolicyWrapper(torch.nn.Module):
        def __init__(self, actor_module: torch.nn.Module, normalizer: torch.nn.Module):
            super().__init__()
            self.normalizer = normalizer
            self.actor = actor_module

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            obs = self.normalizer(obs)
            return self.actor(obs)

    actor_module = actor_module.to("cpu").eval()
    normalizer = normalizer.to("cpu").eval()
    wrapper = OnnxPolicyWrapper(actor_module, normalizer).cpu().eval()

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32, device="cpu")
    export_kwargs = dict(
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrapper, dummy_obs, onnx_path, dynamo=True, **export_kwargs)
        print(f"Exported policy to ONNX with dynamo exporter: {onnx_path}")
    except ModuleNotFoundError as e:
        print(f"onnxscript not found ({e}); falling back to legacy torch.onnx.export without dynamo.")
        torch.onnx.export(wrapper, dummy_obs, onnx_path, dynamo=False, **export_kwargs)
        print(f"Exported policy to ONNX with legacy exporter: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export a trained policy checkpoint to ONNX without running simulation.")
    parser.add_argument("-e", "--exp_name", type=str, default="b2_walking", help="Experiment name under logs/")
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint number (e.g., 100). Defaults to latest.")
    parser.add_argument("--log-root", type=str, default="logs", help="Root directory containing experiment logs.")
    parser.add_argument("--onnx-path", type=str, default=None, help="Output ONNX path. Defaults to logs/.../exported/policies/policy_mlp.onnx")
    parser.add_argument("--onnx-opset", type=int, default=17, help="ONNX opset version to use.")
    args = parser.parse_args()

    log_dir = _resolve_log_dir(args.exp_name, args.log_root)
    ckpt_path = _resolve_ckpt_path(log_dir, args.ckpt)
    cfg_path = os.path.join(log_dir, "cfgs.pkl")

    env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, train_cfg, terrain_cfg = pickle.load(open(cfg_path, "rb"))

    device = torch.device("cpu")
    actor_critic, obs_normalizer = _build_policy(train_cfg, obs_cfg, env_cfg, device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    if isinstance(obs_normalizer, EmpiricalNormalization) and "obs_norm_state_dict" in checkpoint:
        obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])

    save_dir = os.path.join(log_dir, "exported", "policies")
    os.makedirs(save_dir, exist_ok=True)
    onnx_path = args.onnx_path or os.path.join(save_dir, "policy_mlp.onnx")

    _export_onnx(actor_critic.actor, obs_normalizer, obs_cfg["num_obs"], onnx_path, args.onnx_opset)


if __name__ == "__main__":
    main()
