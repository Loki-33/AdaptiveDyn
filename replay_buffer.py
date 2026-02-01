import torch 
import numpy as np 
from collections import deque


class Trajectory:
    def __init__(self, max_len=50, num_actions=4):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.mcts_probs = []
        self.done = []
        self.max_len = max_len
    
    def add(self, obs, action, reward, mcts_probs, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.mcts_probs.append(mcts_probs)
        self.done.append(done)

        if len(self.obs) > self.max_len:
            self.obs.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.mcts_probs.pop(0)
            self.done.pop(0)


    def to_batch(self):
        obs_batch = torch.stack(self.obs)
        actions_batch = torch.tensor(self.actions, dtype=torch.long)
        rewards_batch = torch.tensor(self.rewards, dtype=torch.float32)
        probs_batch = torch.tensor(self.mcts_probs, dtype=torch.float32)
        done_batch = torch.tensor(self.done, dtype=torch.float32)
        return obs_batch, actions_batch, probs_batch, done_batch


class ReplayBuffe:
    def __init__(self, capacity=1000, max_len-50, num_actions=4):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.max_len=max_len
        self.num_actions = num_actions

    def store(self, obs, action, reward, mcts_probs, done):
        traj = Trajectory(self.max_len, self.num_actions)
        traj.add(obs, action, reward, mcts_probs, done)
        self.buffer.append(traj)

    def sample(self, batch_size=16, n_step=5, fresh_threshold=10):
        sampled = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch_obs, batch_actions, batch_rewards, batch_probs, batch_done = [], [], [], [], []

        for idx in sampled:
            traj = self.buffer[idx]
            obs, actions, rewards, probs, done = traj.to_batch()

            # Determine n-step clipping based on trajectory age
            if len(traj.obs) <= fresh_threshold:
                n = n_step
            else:
                n = 1  # old trajectory, clip n-step return

            # Compute n-step returns
            returns = []
            for t in range(len(rewards)):
                n_return = 0.0
                for k in range(n):
                    if t + k < len(rewards):
                        n_return += rewards[t+k]
                returns.append(n_return)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

            batch_obs.append(obs)
            batch_actions.append(actions)
            batch_rewards.append(returns)
            batch_probs.append(probs)
            batch_done.append(done)

        # Stack along batch dimension
        return {
            'obs': torch.stack(batch_obs),
            'actions': torch.stack(batch_actions),
            'n_step_returns': torch.stack(batch_rewards),
            'mcts_probs': torch.stack(batch_probs),
            'done': torch.stack(batch_done)
        }
