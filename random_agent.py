# baseline_agents.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nca import NCA
from world import World

class RandomAgent:
    """Baseline 1: Purely random actions"""
    def __init__(self, num_actions=4):
        self.num_actions = num_actions
    
    def select_action(self, obs):
        return np.random.randint(0, self.num_actions)

class ReactiveAgent(nn.Module):
    """Baseline 2: Simple CNN policy, no planning, no memory"""
    def __init__(self, in_channels=5, num_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_actions)
    
    def forward(self, obs):
        # obs shape: [batch, channels, patch_h, patch_w]
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        logits = self.fc(x)
        return logits
    
    def select_action(self, obs, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(0, 4)
        with torch.no_grad():
            logits = self.forward(obs.unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

def compute_local_entropy(patch):
    """Reward = -ΔH(local patch)"""
    x = patch[0, 0].flatten()  # visible channel only
    hist = torch.histc(x, bins=32, min=-2.0, max=2.0)
    prob = hist / hist.sum()
    p = prob[prob > 0]
    entropy = -(p * torch.log(p)).sum()
    return entropy.item()

def run_episode(world, agent, num_steps=50, patch_size=8):
    """Run one episode and compute total reward"""
    total_reward = 0.0
    
    for t in range(num_steps):
        # Random position on grid
        x = np.random.randint(patch_size//2, world.grid_size - patch_size//2)
        y = np.random.randint(patch_size//2, world.grid_size - patch_size//2)
        
        # Get observation
        obs = world.observe(x, y)
        
        # Compute entropy before action
        h_before = compute_local_entropy(obs)
        
        # Select action
        action = agent.select_action(obs)
        
        # Step world
        world.step(x, y, action)
        
        # Get new observation
        obs_new = world.observe(x, y)
        h_after = compute_local_entropy(obs_new)
        
        # Reward = entropy reduction (creating order)
        reward = h_before - h_after
        total_reward += reward
    
    return total_reward

if __name__ == '__main__':
    print("=== BASELINE COMPARISON ===\n")
    
    num_episodes = 10
    
    # Test Random Agent
    print("Testing Random Agent...")
    random_rewards = []
    for ep in range(num_episodes):
        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        agent = RandomAgent(num_actions=4)
        reward = run_episode(world, agent, num_steps=50)
        random_rewards.append(reward)
        print(f"Episode {ep+1}: {reward:.3f}")
    
    print(f"\nRandom Agent Mean Reward: {np.mean(random_rewards):.3f} ± {np.std(random_rewards):.3f}")
    
    # Test Reactive Agent (untrained)
    print("\n" + "="*50)
    print("Testing Reactive Agent (untrained)...")
    reactive_rewards = []
    reactive_agent = ReactiveAgent(in_channels=5, num_actions=4)
    
    for ep in range(num_episodes):
        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        reward = run_episode(world, reactive_agent, num_steps=50)
        reactive_rewards.append(reward)
        print(f"Episode {ep+1}: {reward:.3f}")
    
    print(f"\nReactive Agent Mean Reward: {np.mean(reactive_rewards):.3f} ± {np.std(reactive_rewards):.3f}")
    
    print("\n" + "="*50)
    print("BASELINE RESULTS:")
    print(f"Random:   {np.mean(random_rewards):.3f} ± {np.std(random_rewards):.3f}")
    print(f"Reactive: {np.mean(reactive_rewards):.3f} ± {np.std(reactive_rewards):.3f}")
    print("\nNext: Train reactive agent to see if task is learnable at all")
