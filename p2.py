import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from world import World
from random_agent import ReactiveAgent

def compute_structure_reward(patch):
    visible = patch[0]  # only first channel
    
    # Compute spatial gradients
    grad_x = torch.abs(visible[1:, :] - visible[:-1, :])
    grad_y = torch.abs(visible[:, 1:] - visible[:, :-1])
    
    # Mean absolute gradient = structure score
    structure = (grad_x.mean() + grad_y.mean()) / 2.0
    
    return structure.item()

def train_reactive_agent_v2(num_episodes=500, lr=1e-3):
    agent = ReactiveAgent(in_channels=5, num_actions=4)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    episode_rewards = []
    structure_scores = []
    
    for episode in range(num_episodes):
        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        
        log_probs = []
        rewards = []
        structures = []
        
        # Collect trajectory
        for t in range(50):
            x = np.random.randint(4, 60)
            y = np.random.randint(4, 60)
            
            obs = world.observe(x, y)
            
            # Get action probabilities
            logits = agent(obs.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            # Step world
            world.step(x, y, action.item())
            
            # NEW REWARD: structure after action
            obs_new = world.observe(x, y)
            structure = compute_structure_reward(obs_new.unsqueeze(0))
            
            rewards.append(structure)
            structures.append(structure)
        
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
        avg_structure = np.mean(structures)
        episode_rewards.append(ep_reward)
        structure_scores.append(avg_structure)
        
        if episode % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            avg_s = np.mean(structure_scores[-50:]) if len(structure_scores) >= 50 else np.mean(structure_scores)
            print(f"Ep {episode}: Reward={ep_reward:.3f}, Structure={avg_structure:.4f}, Avg(50)={avg_r:.3f}")
    
    return agent, episode_rewards, structure_scores


if __name__ == '__main__':
    print("=== Training Reactive Agent v2 (Structure Reward) ===\n")
    agent, rewards, structures = train_reactive_agent_v2(num_episodes=500)
    
    # Plot learning curve
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(structures)
    plt.xlabel('Episode')
    plt.ylabel('Avg Structure Score')
    plt.title('Structure Maintenance')
    
    plt.tight_layout()
    plt.savefig('training_v2.png')
    print("Saved training_v2.png")
    
    # Evaluate
    print("\n=== FINAL EVALUATION ===")
    eval_rewards = []
    eval_structures = []
    
    for ep in range(20):
        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        total_reward = 0
        structures_ep = []
        
        for t in range(50):
            x = np.random.randint(4, 60)
            y = np.random.randint(4, 60)
            obs = world.observe(x, y)
            
            action = agent.select_action(obs, epsilon=0.0)
            world.step(x, y, action)
            
            obs_new = world.observe(x, y)
            structure = compute_structure_reward(obs_new.unsqueeze(0))
            total_reward += structure
            structures_ep.append(structure)
        
        eval_rewards.append(total_reward)
        eval_structures.append(np.mean(structures_ep))
    
    print(f"\nTrained Agent: {np.mean(eval_rewards):.3f} ± {np.std(eval_rewards):.3f}")
    print(f"Avg Structure: {np.mean(eval_structures):.4f} ± {np.std(eval_structures):.4f}")
    
    # Compare to random baseline
    print("\n=== Random Baseline ===")
    random_rewards = []
    random_structures = []
    
    for ep in range(20):
        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        total_reward = 0
        structures_ep = []
        
        for t in range(50):
            x = np.random.randint(4, 60)
            y = np.random.randint(4, 60)
            
            action = np.random.randint(0, 4)
            world.step(x, y, action)
            
            obs = world.observe(x, y)
            structure = compute_structure_reward(obs.unsqueeze(0))
            total_reward += structure
            structures_ep.append(structure)
        
        random_rewards.append(total_reward)
        random_structures.append(np.mean(structures_ep))
    
    print(f"Random Agent:  {np.mean(random_rewards):.3f} ± {np.std(random_rewards):.3f}")
    print(f"Avg Structure: {np.mean(random_structures):.4f} ± {np.std(random_structures):.4f}")
    
    print("\n" + "="*60)
    improvement = np.mean(eval_structures) - np.mean(random_structures)
    if improvement > 0:
        print(f"✓ Agent LEARNED to create structure (+{improvement:.4f})")
        print("  → Task is solvable, proceed to planning")
    else:
        print(f"✗ Agent didn't learn ({improvement:.4f})")
        print("  → May need better architecture or more training")
    
    torch.save(agent.state_dict(), 'reactive_agent_v2.pt')
