import argparse
import numpy as np
import torch

from dqn_agent import DQNAgent
from mini_metro_env import MiniMetroEnv

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained DQN agent on Mini Metro")
    p.add_argument("--model", type=str, required=True, help="Checkpoint path (.pt)")
    p.add_argument("--episodes", type=int, default=5, help="Number of rollout episodes")
    p.add_argument("--mode", choices=["pixels", "symbolic"], default="pixels")
    p.add_argument("--feature-dim", type=int, default=None)
    p.add_argument("--epsilon", type=float, default=0.0, help="ε-greedy during evaluation")
    return p.parse_args()

def main():
    args = parse_args()

    env = MiniMetroEnv()
    env.perceive()
    try:
        n_actions = len(env.available_actions)
    except:
        n_actions = 10_000

    agent = DQNAgent(
        n_actions=n_actions,
        mode=args.mode,
        feature_dim=args.feature_dim,
        replay_capacity=1,     # dummy
        lr=1e-4, gamma=0.99,
        target_update_freq=1,
        device=None
    )
    agent.load(args.model)
    agent.q_online.eval()

    for ep in range(1, args.episodes + 1):
        try:
            state = env.reset()
        except AttributeError:
            env.perceive()
            state = np.asarray(env.screenshot, dtype=np.uint8).transpose(2, 0, 1)

        total_reward = 0.0
        done = False
        t = 0
        while not done:
            t += 1
            # Select greedy action (ε can be nonzero if you want randomization)
            action_idx = agent.select_action(state, args.epsilon)
            next_state, reward, done, info = env.step(action_idx)
            total_reward += reward
            state = next_state
            if t > 500:
                break  # safety

        print(f"[Eval] Episode {ep} | Reward: {total_reward:7.2f} | Steps: {t}")

if __name__ == "__main__":
    main()
