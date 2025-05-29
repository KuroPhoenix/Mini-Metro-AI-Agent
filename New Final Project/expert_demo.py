import json
from pathlib import Path
from mini_metro_env import MiniMetroEnv
from mini_metro_rl_agent import MiniMetroRLAgent

DATA_DIR   = Path("expert_dataset")
DATA_DIR.mkdir(exist_ok=True)

env       = MiniMetroEnv()
agent     = MiniMetroRLAgent(actions=list(range(10_000)))   # large upper bound
ep_length = 120                                             # 2 minutes-ish

for episode in range(3):                                    # you decide how many
    print(f"\n🎮  Human Episode {episode+1}")
    env.reset()
    trajectory = []                                         # [(s, a, r, s')]
    state = env.perceive()

    for step in range(ep_length):
        # --------------------------------------------------------------
        # let the human perform exactly one drag-action on screen
        # the helper *must* translate that drag into the two station
        # centres (station_A, station_B) that were involved
        # --------------------------------------------------------------
        station_A, station_B = env.wait_for_human_drag()    # ← you implement

        action_idx = env.action_index(station_A, station_B)
        next_state, reward, done = env.step(action_idx)

        # real-time Q-update (pure imitation learning)
        agent.update(state, action_idx, reward, next_state)

        trajectory.append((state, action_idx, reward, next_state))
        state = next_state
        if done:
            break

    # save the demonstration for possible offline re-training later
    with (DATA_DIR / f"episode_{episode:03d}.json").open("w") as fp:
        json.dump(trajectory, fp)
    print("saved trajectory")

agent.save("q_table_from_human.npy")
print("✅  Q-table updated with your demonstrations")