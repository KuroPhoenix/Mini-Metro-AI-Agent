#!/usr/bin/env python3
"""evaluate_agent.py  –  Rollout trained policies in Mini Metro

This script plays one or more full games using a checkpoint produced by
``dagger_train.py`` or any other script that saves a compatible
``MiniMetroRLAgent``.  It prints summary statistics and, if requested,
writes a GIF of the first episode so you can visually sanity‑check what
the agent is doing.

Example usage
-------------
$ python evaluate_agent.py \
      --model models/policy_latest.pt \
      --episodes 10 \
      --mode pixels \
      --gif runs/2025‑05‑30_demo.gif
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import List

import imageio.v2 as imageio  # «pip install imageio» if missing
import numpy as np

# ---------------------------------------------------------------------------
#  Project‑local imports – adjust to your package layout
# ---------------------------------------------------------------------------

from mini_metro_env import MiniMetroEnv  # your environment wrapper
from mini_metro_rl_agent import MiniMetroRLAgent, preprocess

# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Mini Metro policy.")
    p.add_argument("--model", type=str, required=True, help="Checkpoint path ( .pt )")
    p.add_argument("--episodes", type=int, default=5, help="How many games to play")
    p.add_argument("--mode", choices=["pixels", "symbolic"], default="pixels")
    p.add_argument("--feature-dim", type=int, default=None, help="Needed if mode=symbolic and not stored in ckpt")
    p.add_argument("--gif", type=str, default=None, help="Save a GIF of the FIRST episode to this file")
    p.add_argument("--epsilon", type=float, default=0.0, help="ε‑greedy exploration during evaluation")
    p.add_argument("--fps-log", action="store_true", help="Print instantaneous FPS each step (spammy)")
    return p.parse_args()

# ---------------------------------------------------------------------------
#  Rollout helpers
# ---------------------------------------------------------------------------

def run_episode(env: MiniMetroEnv, agent: MiniMetroRLAgent, *, record_frames: bool = False, epsilon: float = 0.0):
    """Play one episode; return (total_reward, n_steps, frames | None)."""
    obs = env.reset()
    done = False
    total_reward = 0.0
    frames: List[np.ndarray] | None = [] if record_frames else None

    start_time = time.perf_counter()
    steps = 0
    while not done:
        action = agent.act(obs, epsilon=epsilon)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if record_frames:
            frames.append(env.render_to_rgb())  # you must implement render_to_rgb() in env
    wall_time = time.perf_counter() - start_time
    return total_reward, steps, wall_time, frames

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Environment ------------------------------------------------------
    env = MiniMetroEnv(headless=False)  # or True if you don’t need the window visible

    # 2. Agent ------------------------------------------------------------
    agent = MiniMetroRLAgent.load(args.model, device="cpu")  # → uses mode stored in ckpt
    # Override mode/feature_dim only if explicitly requested
    if agent.mode != args.mode:
        print(f"[warn] overriding ckpt mode {agent.mode!r} → {args.mode!r}")
        agent.mode = args.mode
    if args.feature_dim and agent.mode == "symbolic":
        # sanity‑check: SymbolicPolicy expects fixed input dim.  If mismatch, rebuild net.
        if agent.net.net[0].in_features != args.feature_dim:
            from mini_metro_rl_agent import SymbolicPolicy  # local import to avoid circular

            agent.net = SymbolicPolicy(args.feature_dim, agent.n_actions)
            agent.net.load_state_dict(torch.load(args.model)["state_dict"], strict=False)
            agent.net.eval()

    # 3. Evaluation loop --------------------------------------------------
    rewards, steps_taken, times = [], [], []

    for ep in range(1, args.episodes + 1):
        rec = (ep == 1 and args.gif is not None)
        r, n, t, frames = run_episode(env, agent, record_frames=rec, epsilon=args.epsilon)
        rewards.append(r)
        steps_taken.append(n)
        times.append(t)
        fps = n / t if t > 0 else 0.0
        print(f"Episode {ep:02d}  reward={r:.1f}  steps={n}  wall‑time={t:.1f}s  FPS={fps:.1f}")

        if rec and frames is not None:
            print(f"  – Writing GIF to {args.gif} …")
            Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.gif, frames, fps=15)

    # 4. Aggregate summary ----------------------------------------------
    print("\n=== Summary ===")
    print(f"episodes   : {args.episodes}")
    print(f"mean reward: {statistics.mean(rewards):.1f} ± {statistics.stdev(rewards):.1f}")
    print(f"mean steps : {statistics.mean(steps_taken):.1f}")
    print(f"mean FPS   : {statistics.mean(s/ t for s, t in zip(steps_taken, times)):.1f}")


if __name__ == "__main__":
    main()
