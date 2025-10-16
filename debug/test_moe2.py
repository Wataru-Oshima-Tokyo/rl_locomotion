from __future__ import annotations
import torch

import os
import statistics
import time
import torch
from collections import deque
import matplotlib.pyplot as plt

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticRecurrentMoE, EmpiricalNormalization
from rsl_rl.utils import store_code_state
import numpy as np
B, H, A, K = 8, 16, 12, 4
z = torch.randn(1, B, H)

# モデルを適当に作る
moe = ActorCriticRecurrentMoE(num_actor_obs=H, num_critic_obs=H, num_actions=A, moe={"num_experts": K})
# 1) rollout APIで通るか
obs = torch.randn(B, H)
a = moe.act(obs)                  # -> [B, A] / 有限チェック
assert torch.isfinite(a).all()

z = moe.memory_a(obs)             # [1,B,256]
dist, base = moe._moe_build_distribution(z)
smp = dist.sample()               # [1,B,A]
lp  = dist.log_prob(smp)          # [1,B]
assert torch.isfinite(smp).all() and torch.isfinite(lp).all()

# 3) ゲート統計
gp = moe.get_last_gate_probs()          # [1,B,K]
print("gate probs row-sum:", gp.sum(-1))# ≈1
print("gate entropy ~", (-(gp*gp.clamp_min(1e-9).log()).sum(-1)).mean().item())