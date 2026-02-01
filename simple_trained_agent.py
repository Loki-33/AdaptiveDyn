# train_reactive.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from world import World
from random_agent import ReactiveAgent, compute_local_entropy

def train_reactive_agent(num_episodes=500, lr=1e-3):
    agent = ReactiveAgent(in_channels=5, num_actions=4)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        
        log_probs = []
        rewards = []
        
        # Collect trajectory
        for t in range(50):
            x = np.random.randint(4, 60)
            y = np.random.randint(4, 60)
            
            obs = world.observe(x, y)
            h_before = compute_local_entropy(obs.unsqueeze(0))
            
            # Get action probabilities
            logits = agent(obs.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            # Step world
            world.step(x, y, action.item())
            
            obs_new = world.observe(x, y)
            h_after = compute_local_entropy(obs_new.unsqueeze(0))
            
            reward = h_before - h_after
            rewards.append(reward)
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)
        
        if episode % 50 == 0:
            avg = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            print(f"Episode {episode}: Reward = {ep_reward:.3f}, Avg(50) = {avg:.3f}")
    
    return agent, episode_rewards

if __name__ == '__main__':
    print("Training Reactive Agent with REINFORCE...\n")
    agent, rewards = train_reactive_agent(num_episodes=500)
    
    print("\n=== FINAL EVALUATION ===")
    eval_rewards = []
    for ep in range(20):
        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        total_reward = 0
        for t in range(50):
            x = np.random.randint(4, 60)
            y = np.random.randint(4, 60)
            obs = world.observe(x, y)
            h_before = compute_local_entropy(obs.unsqueeze(0))
            
            action = agent.select_action(obs, epsilon=0.0)  # greedy
            world.step(x, y, action)
            
            obs_new = world.observe(x, y)
            h_after = compute_local_entropy(obs_new.unsqueeze(0))
            total_reward += (h_before - h_after)
        
        eval_rewards.append(total_reward)
    
    print(f"\nTrained Reactive Agent: {np.mean(eval_rewards):.3f} ± {np.std(eval_rewards):.3f}")
    print(f"Random Baseline:        -0.686 ± 0.957")
    
    if np.mean(eval_rewards) > -0.686:
        print("\n Task is LEARNABLE - reactive policy beats random")
    else:
        print("\n Task might be TOO HARD - reactive policy can't learn")
    
    # Save agent
    torch.save(agent.state_dict(), 'reactive_agent.pt')
    print("\nSaved trained agent to reactive_agent.pt")
