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
B,H,A = 4096, 16, 12
obs_a = torch.randn(B, H)
obs_c = torch.randn(B, H)

pi = ActorCriticRecurrentMoE(H, H, A, moe={"num_experts":6, "min_std":0.05})

a = pi.act(obs_a)           # [B, A]
v = pi.evaluate(obs_c)      # [B, 1]  ← ここが重要
assert a.shape == (B, A) and v.shape == (B, 1)