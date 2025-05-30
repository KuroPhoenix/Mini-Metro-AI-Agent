"""
DAGGER training – aggregates expert actions on states visited by the current
policy.  Uses `expert_demo.wait_for_human_drag` as the oracle.
"""
import random, json, time, copy
from pathlib import Path
from mini_metro_env import MiniMetroEnv
from mini_metro_rl_agent import MiniMetroRLAgent
from vision.preprocess import preprocess
from models_and_dataset import (
    CNNPolicy, SymbolicPolicy,
    behavioural_clone, make_loader
)

DATA   = Path("dagger_dataset")
DATA.mkdir(exist_ok=True)

env     = MiniMetroEnv()
agent = MiniMetroRLAgent(n_actions=len(env.action_space), mode="symbolic")
N_ITER  = 5          # total DAgger iterations
HORIZON = 120        # steps per rollout

for k in range(N_ITER):
    print(f"\n🔁  DAgger iter {k+1}/{N_ITER}")
    episode_data = []

    raw = env.reset()  # or env.perceive() if you prefer frame-grabs
    state = preprocess(raw, mode="symbolic")
    for t in range(HORIZON):
        # 1. agent acts
        a_pred = agent.choose_action(state)

        # 2. expert corrects that state → oracle action
        try:
            a_expert_idx = env.action_index(*env.wait_for_human_drag())
        except TimeoutError:
            a_expert_idx = a_pred          # fall back: trust agent

        # 3. env transitions
        raw_next, reward, done, _ = env.step(a_pred)
        next_state = preprocess(raw_next, mode="symbolic")

        # 4. record (s, a_expert)
        episode_data.append((raw, a_expert_idx))
        state = next_state
        if done:
            break

    # 5. add to dataset & retrain policy
    with (DATA / f"iter_{k:02d}.json").open("w") as fp:
        json.dump(episode_data, fp)

    # Flatten all JSON files → supervised train
    all_pairs = []
    for p in DATA.glob("*.json"):
        all_pairs.extend(json.load(p.open()))

    agent.behavioural_clone(all_pairs)   # implement in your agent class
