
import time
from mini_metro_env import MiniMetroEnv

env = MiniMetroEnv()
EPISODES = 10

for episode in range(EPISODES):
    print(f"\n🎮 Episode {episode + 1}")
    env.reset()
    for step in range(4):
        action_idx = 0
        next_state, reward, done = env.step(action_idx)
        time.sleep(1)
