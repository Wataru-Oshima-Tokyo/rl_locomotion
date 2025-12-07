# teleop_with_policy.py  ────────────────────────────────────────────────
import argparse, threading, time, os, pickle, re, torch
import numpy as np
from typing import TYPE_CHECKING
from pynput import keyboard
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

if TYPE_CHECKING:
    from legged_env import LeggedEnv

# ---------- helper -----------------------------------------------------
def load_last_run(base_dir):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"No runs found: directory '{base_dir}' does not exist.")

    candidates = [d for d in os.listdir(base_dir)
                  if os.path.isdir(f"{base_dir}/{d}")]
    if not candidates:
        raise FileNotFoundError(f"No runs found under '{base_dir}'.")

    latest = sorted(candidates)[-1]
    return f"{base_dir}/{latest}"

def load_cfgs(exp_name):
    run_dir = load_last_run(f"logs/{exp_name}")
    return pickle.load(open(f"{run_dir}/cfgs.pkl","rb")), run_dir

class KeyboardDevice:
    def __init__(self):
        self.pressed = set()
        self.lock    = threading.Lock()
        self.listener = keyboard.Listener(
            on_press=self._on, on_release=self._off)

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
        self.listener.join()

    # ⬇︎ FIXED: each statement on its own line
    def _on(self, key):
        with self.lock:
            self.pressed.add(key)

    def _off(self, key):
        with self.lock:
            self.pressed.discard(key)


# ---------- main -------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-e","--exp_name",default="go2_walking")
    ap.add_argument("-n","--num_envs",type=int,default=1)
    args=ap.parse_args()

    gs.init()
    # Delay import until after gs.init()
    from legged_env import LeggedEnv
    (env_cfg,obs_cfg,noise_cfg,reward_cfg,command_cfg,
     train_cfg,terrain_cfg), run_dir = load_cfgs(args.exp_name)
    reward_cfg["reward_scales"] = {}
    env_cfg["episode_length_s"] = 3600
    # train_cfg["policy"]["class_name"] = "ActorCriticRecurrent"      # or "ActorCriticRecurrent"
    # train_cfg["algorithm"]["class_name"] = "PPO"          # ← add this line
    command_cfg["curriculum"] = False
    env_cfg["randomize_rot"] = False
    env = LeggedEnv(args.num_envs, env_cfg, obs_cfg, noise_cfg,
                    reward_cfg, command_cfg, terrain_cfg,
                    show_viewer_=True, eval_=True, control_=True, show_camera_=False)

    # --- load trained actor ---
    runner = OnPolicyRunner(env, train_cfg, run_dir, device="cuda:0")
    # pick newest checkpoint
    ckpt = max((f for f in os.listdir(run_dir) if f.startswith("model_")),
               key=lambda n:int(re.search(r'\d+',n).group()))
    runner.load(os.path.join(run_dir, ckpt))
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    # obs = obs.to(device)           # first frame on correct device

    # --- keyboard ---
    kb=KeyboardDevice(); kb.start()
    base_cmd = torch.zeros(3,device=env.device)
    # ---  speeds  -------------------------------------------------
    lin_step   = 0.5                    # [m/s] at 100 % scale
    ang_step   = 0.8                    # [rad/s] at 100 % scale
    speed_lin  = 1.0                    # scaling factors (changed by q/z, w/x)
    speed_ang  = 1.0

    # printable help once
    help_msg = """
Classic teleop-twist keys
  u  i  o        +x plus yaw (u/o) or straight +x (i)
  j  k  l        +yaw — stop — -yaw
  m  ,  .        -x plus yaw (m/.) or straight -x (,)

Holonomic (Shift held):
  U  I  O        +x  +y  (strafe)  +x -y
  J  K  L        +y  stop  -y
  M  <  >        -x +y      -x -y

t / b            up / down  (+z / -z)   [ignored for quad]
q/z              inc/dec linear & angular by 10 %
w/x              inc/dec linear only
e/c              inc/dec angular only
u                reset robot
Esc              quit
"""
    print(help_msg)

    try:
        while True:
            time.sleep(env.sim_dt)                        # keep real‑time pace
            keys = kb.pressed.copy()                 # non‑blocking

            # ─── quit & reset ───────────────────────────────────────────
            if keyboard.Key.esc in keys:
                break
            if keyboard.KeyCode.from_char("Q") in keys:
                env.reset()
                base_cmd.zero_()

            # ─── speed scaling (q/z w/x e/c) ────────────────────────────
            if keyboard.KeyCode.from_char("q") in keys:   # both +
                speed_lin *= 1.1; speed_ang *= 1.1
            if keyboard.KeyCode.from_char("z") in keys:   # both -
                speed_lin *= 0.9; speed_ang *= 0.9
            if keyboard.KeyCode.from_char("w") in keys:   # lin +
                speed_lin *= 1.1
            if keyboard.KeyCode.from_char("x") in keys:   # lin -
                speed_lin *= 0.9
            if keyboard.KeyCode.from_char("e") in keys:   # ang +
                speed_ang *= 1.1
            if keyboard.KeyCode.from_char("c") in keys:   # ang -
                speed_ang *= 0.9

            # ─── build command from keymap ──────────────────────────────
            cmd = torch.zeros_like(base_cmd)

            # lowercase set (no shift)  → differential drive style
            if keyboard.KeyCode.from_char('i') in keys:  cmd[0] = +lin_step
            if keyboard.KeyCode.from_char(',') in keys:  cmd[0] = -lin_step
            if keyboard.KeyCode.from_char('j') in keys:  cmd[2] = +ang_step
            if keyboard.KeyCode.from_char('l') in keys:  cmd[2] = -ang_step
            if keyboard.KeyCode.from_char('u') in keys:  cmd[0] = +lin_step; cmd[2] = +ang_step
            if keyboard.KeyCode.from_char('o') in keys:  cmd[0] = +lin_step; cmd[2] = -ang_step
            if keyboard.KeyCode.from_char('m') in keys:  cmd[0] = -lin_step; cmd[2] = +ang_step
            if keyboard.KeyCode.from_char('.') in keys:  cmd[0] = -lin_step; cmd[2] = -ang_step

            # uppercase set (Shift)  → holonomic strafing
            if keyboard.KeyCode.from_char('I') in keys:  cmd[0] = +lin_step                  # fwd
            if keyboard.KeyCode.from_char('<') in keys or ',' in keys and keyboard.Key.shift in keys:
                             cmd[0] = -lin_step                  # back (Shift+comma)
            if keyboard.KeyCode.from_char('J') in keys:  cmd[1] = +lin_step                  # left strafe
            if keyboard.KeyCode.from_char('L') in keys:  cmd[1] = -lin_step                  # right strafe
            if keyboard.KeyCode.from_char('U') in keys:  cmd[0] = +lin_step; cmd[1] = +lin_step
            if keyboard.KeyCode.from_char('O') in keys:  cmd[0] = +lin_step; cmd[1] = -lin_step
            if keyboard.KeyCode.from_char('M') in keys:  cmd[0] = -lin_step; cmd[1] = +lin_step
            if keyboard.KeyCode.from_char('>') in keys:  cmd[0] = -lin_step; cmd[1] = -lin_step  # Shift+period

            # apply scaling
            cmd[0:2] *= speed_lin
            cmd[2]   *= speed_ang
            base_cmd = cmd

            # feed into env
            env.commands[:] = base_cmd.unsqueeze(0)

            # ─── policy inference & sim step ────────────────────────────
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    finally:
        kb.stop()
        print("Tele‑op with policy finished.")

if __name__=="__main__":
    main()
