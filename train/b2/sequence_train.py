"""
Run the B2 training stages back-to-back.

Stages:
1) flat_train.py
2) easy_rough_train.py
3) rough_train.py

Usage examples:
  python run_b2_sequence.py
  python run_b2_sequence.py --exp_name my_exp --flat_iters 6000 --resume_after_first
  python run_b2_sequence.py --start_stage easy_rough --no-resume --extra --vis
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).parent


def build_cmd(script_name: str, iterations: int, exp_name: str, num_envs: int, resume: bool, extra: List[str]):
    """Assemble the command to launch a single stage."""
    cmd = [
        sys.executable,
        str(BASE_DIR / script_name),
        "--max_iterations",
        str(iterations),
        "--exp_name",
        exp_name,
        "-B",
        str(num_envs),
    ]
    if resume:
        cmd.append("--resume")
    if extra:
        cmd.extend(extra)
    return cmd


def run_stage(title: str, cmd: List[str]) -> None:
    print(f"\n=== Starting {title} ===")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{title} failed with exit code {result.returncode}")
    print(f"=== Finished {title} ===\n")


def main():
    parser = argparse.ArgumentParser(description="Run B2 flat -> easy_rough -> rough training sequentially.")
    parser.add_argument("--exp_name", default="b2_walking", help="Experiment name forwarded to each stage.")
    parser.add_argument("-B", "--num_envs", type=int, default=8192, help="Number of envs forwarded to each stage.")
    parser.add_argument("--flat_iters", type=int, default=8000, help="Max iterations for flat_train.py.")
    parser.add_argument("--easy_iters", type=int, default=5000, help="Max iterations for easy_rough_train.py.")
    parser.add_argument("--rough_iters", type=int, default=10000, help="Max iterations for rough_train.py.")
    parser.add_argument(
        "--start_stage",
        choices=["flat", "easy_rough", "rough"],
        default="flat",
        help="Start from this stage (useful if you already finished earlier ones).",
    )
    parser.add_argument(
        "--resume_after_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the latest checkpoint for every stage after the first one you run.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Extra args appended to every stage command (prefix them with '--').",
    )
    args = parser.parse_args()

    stage_order = [
        ("flat", "flat_train.py", args.flat_iters),
        ("easy_rough", "easy_rough_train.py", args.easy_iters),
        ("rough", "rough_train.py", args.rough_iters),
    ]

    start_idx = next(i for i, (name, _, _) in enumerate(stage_order) if name == args.start_stage)

    extra_args = args.extra or []
    stages_to_run = stage_order[start_idx:]

    for idx, (name, script, iters) in enumerate(stages_to_run):
        resume_flag = args.resume_after_first and idx > 0
        cmd = build_cmd(script, iters, args.exp_name, args.num_envs, resume_flag, extra_args)
        run_stage(name, cmd)


if __name__ == "__main__":
    main()
