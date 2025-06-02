# train_dqn.py

import argparse
import os
import time
import numpy as np
import torch

from dqn_agent import DQNAgent
from mini_metro_env import MiniMetroEnv  # your existing env wrapper

def parse_args():
    p = argparse.ArgumentParser(description="Train a DQN agent on Mini Metro")
    p.add_argument("--mode", choices=["pixels"], default="pixels",
                   help="Observation mode: only 'pixels' is supported now")
    p.add_argument("--episodes", type=int, default=500,
                   help="Number of episodes to train for")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Max timesteps per episode (to avoid infinite loops)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for DQN updates")
    p.add_argument("--replay-capacity", type=int, default=100_000,
                   help="Max capacity of replay buffer")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate for Adam optimizer")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor")
    p.add_argument("--target-update-freq", type=int, default=1000,
                   help="Number of gradient steps between target network updates")
    p.add_argument("--start-epsilon", type=float, default=1.0,
                   help="Initial ε for ε‐greedy")
    p.add_argument("--end-epsilon", type=float, default=0.01,
                   help="Final ε after linear decay")
    p.add_argument("--epsilon-decay-episodes", type=int, default=300,
                   help="Episodes over which ε decays")
    p.add_argument("--save-dir", type=str, default="checkpoints",
                   help="Directory to save checkpoints (auto‐created if missing)")
    p.add_argument("--save-freq", type=int, default=50,
                   help="Save the model every N episodes")
    p.add_argument("--device", type=str, default=None,
                   help="Device for training (e.g. 'cuda' or 'cpu'). If None, auto‐detect.")
    return p.parse_args()


def linear_epsilon(episode: int, start_eps: float, end_eps: float, decay_episodes: int):
    """
    Linearly decay epsilon from start_eps→end_eps over the first decay_episodes.
    After that, keep it at end_eps.
    """
    if episode >= decay_episodes:
        return end_eps
    slope = (end_eps - start_eps) / float(decay_episodes)
    return float(start_eps + slope * episode)


def main():
    args = parse_args()

    # 1. Create save directory if needed
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. Instantiate the Mini Metro environment
    env = MiniMetroEnv()  # your existing class; ensure it has reset() + step()

    # 3. Determine the *actual* number of valid actions = len(env.actions)
    print("Initializing environment to get action size...")
    env.perceive()   # Let the environment build `self.actions`

    try:
        n_actions = len(env.actions)
        print(f"Detected n_actions = {n_actions} from env.actions")
    except Exception as e:
        raise RuntimeError(
            "Failed to read `env.actions`. Make sure `mini_metro_env.py` populates `self.actions` inside perceive()."
        ) from e

    # 4. Instantiate DQNAgent with the correct n_actions
    agent = DQNAgent(
        n_actions=n_actions,
        mode=args.mode,
        feature_dim=None,               # we only support pixels now
        replay_capacity=args.replay_capacity,
        lr=args.lr,
        gamma=args.gamma,
        target_update_freq=args.target_update_freq,
        device=args.device
    )

    total_steps = 0
    for episode in range(1, args.episodes + 1):
        # 5a. Reset the environment (new game)
        try:
            state = env.reset()
        except AttributeError:
            # If no reset() is implemented, fallback to just perceiving again:
            env.perceive()
            frame = env.screenshot  # (H,W,3) uint8
            state = np.asarray(frame, dtype=np.uint8).transpose(2, 0, 1)
        episode_reward = 0.0
        losses = []

        # 5b. Compute ε for this episode
        epsilon = linear_epsilon(episode, args.start_epsilon, args.end_epsilon, args.epsilon_decay_episodes)

        # 5c. Step loop
        for t in range(1, args.max_steps + 1):
            total_steps += 1

            # 1) Select action via ε‐greedy
            action_idx = agent.select_action(state, epsilon)

            # 2) Step the environment
            next_state, reward, done, info = env.step(action_idx)

            # 3) Store transition
            agent.store_transition(state, action_idx, reward, next_state, float(done))

            # 4) DQN update
            loss = agent.update(args.batch_size)
            if loss is not None:
                losses.append(loss)

            # 5) Accumulate reward and set up next state
            episode_reward += reward
            state = next_state

            if done:
                break

        # 6. Episode end: log stats
        avg_loss = np.mean(losses) if losses else 0.0
        print(
            f"Episode {episode:4d} | Reward: {episode_reward:7.2f} | "
            f"Epsilon: {epsilon:.3f} | Loss: {avg_loss:.4f} | Steps: {t}"
        )

        # 7. Save model if needed
        if episode % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_dir, f"dqn_ckpt_ep{episode:04d}.pt")
            agent.save(ckpt_path)
            print(f" → Saved checkpoint to {ckpt_path}")

    # 8. Final save
    final_path = os.path.join(args.save_dir, "dqn_final.pt")
    agent.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
