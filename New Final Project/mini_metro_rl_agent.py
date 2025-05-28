import numpy as np
import random
from collections import defaultdict
class MiniMetroRLAgent:
    def __init__(self, actions):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.actions = actions
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.2

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(len(self.actions)))
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - predict)


    def save(self, filename="q_table.npy"):
        np.save(filename, dict(self.q_table))

    def load(self, filename="q_table.npy"):
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), np.load(filename, allow_pickle=True).item())
