# dqn_agent.py

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Only import CNNPolicy—there is no SymbolicPolicy in RL/policy.py
from RL.policy import CNNPolicy

# ----------------------------------------------
#  Simple Replay Buffer
# ----------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer.
        - state, next_state: numpy arrays (or Tensors) of same shape
        - action: int
        - reward: float
        - done: float (0.0 or 1.0)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Returns: five tensors (states, actions, rewards, next_states, dones),
        each of shape (batch_size, …).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert lists → PyTorch tensors
        states      = torch.stack([torch.from_numpy(s) for s in states], dim=0)
        actions     = torch.tensor(actions, dtype=torch.long, device=states.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=states.device)
        next_states = torch.stack([torch.from_numpy(s) for s in next_states], dim=0)
        dones       = torch.tensor(dones, dtype=torch.float32, device=states.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------
#  DQN Agent (pixels‐only)
# ----------------------------------------------
class DQNAgent:
    def __init__(
        self,
        n_actions: int,
        *,
        mode: str = "pixels",          # only "pixels" is supported
        feature_dim: int = None,       # not used
        replay_capacity: int = 100_000,
        lr: float = 1e-4,
        device: torch.device | str = None,
        gamma: float = 0.99,
        target_update_freq: int = 1000
    ):
        """
        - n_actions: the size of your discrete action space
        - mode: only "pixels" is supported in this version
        - replay_capacity: max # of transitions stored
        - lr: learning rate
        - gamma: discount factor
        - target_update_freq: how many gradient steps between copying online→target network
        """

        if mode != "pixels":
            raise ValueError("Only mode='pixels' is supported. Remove any reference to 'symbolic' in your call.")

        self.mode = mode
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        # Create online Q‐network and target Q‐network
        self.q_online = CNNPolicy(n_actions).to(self.device)
        self.q_target = CNNPolicy(n_actions).to(self.device)

        # Copy weights online → target
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=replay_capacity)

        # Internal step counter (to know when to update target)
        self.update_count = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Given a state (NumPy), choose an ε‐greedy action index.
        - state: a (3,H,W) uint8 array (pixels)
        - epsilon: float ∈ [0,1], probability of random action
        Returns: int action_idx in [0, n_actions)
        """
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        else:
            # Convert state → Tensor (add batch dimension)
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            # Normalize to [0,1] if needed (CNNPolicy expects floats in [0,1])
            if state_tensor.max() > 1.0:
                state_tensor = state_tensor / 255.0
            q_values = self.q_online(state_tensor)              # shape: (1, n_actions)
            action = q_values.argmax(dim=1).item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store (s,a,r,s',done) in replay buffer.
        - state, next_state: (3,H,W) numpy arrays
        - action: int
        - reward: float
        - done: float (0.0 or 1.0)
        """
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, batch_size: int):
        """
        Sample a minibatch from replay buffer and do one gradient step on
        MSE( Q_online(s,a), target ). Return loss (float) for logging, or None
        if not enough samples yet.
        """
        if len(self.buffer) < batch_size:
            return None

        # 1) Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # Move everything to the device
        states      = states.to(self.device).float()
        next_states = next_states.to(self.device).float()
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        dones       = dones.to(self.device)

        # Normalize pixel states if needed
        if states.max() > 1.0:
            states = states / 255.0
        if next_states.max() > 1.0:
            next_states = next_states / 255.0

        # 2) Compute Q(s,a) using online network
        q_values      = self.q_online(states)                                      # (B, n_actions)
        q_value_s_a   = q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)  # (B,)

        # 3) Compute target: r + γ * (1 - done) * max_a' Q_target(s', a')
        with torch.no_grad():
            q_target_next = self.q_target(next_states)                              # (B, n_actions)
            max_q_target_next, _ = q_target_next.max(dim=1)                         # (B,)
            y = rewards + (1.0 - dones) * self.gamma * max_q_target_next           # (B,)

        # 4) MSE loss between Q_online(s,a) and y
        loss = F.mse_loss(q_value_s_a, y)

        # 5) Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6) Maybe update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

        return loss.item()

    def save(self, filepath: str):
        """
        Save only the online network’s state_dict (no need to save replay buffer).
        """
        torch.save(self.q_online.state_dict(), filepath)

    def load(self, filepath: str):
        """
        Load weights into both online and target networks.
        """
        state_dict = torch.load(filepath, map_location=self.device)
        self.q_online.load_state_dict(state_dict)
        self.q_target.load_state_dict(state_dict)
        self.q_online.to(self.device)
        self.q_target.to(self.device)
        self.q_online.eval()
        self.q_target.eval()
