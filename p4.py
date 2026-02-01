# train_reactive_v3.py
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from world import World
from random_agent import ReactiveAgent
from p2 import compute_structure_reward

def train_reactive_agent_v3(num_episodes=500, lr=1e-3):
    agent = ReactiveAgent(in_channels=5, num_actions=4)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        
        # Let world settle first (important!)
        for _ in range(10):
            world.step()
        
        log_probs = []
        rewards = []
        
        # Collect trajectory
        for t in range(50):
            x = np.random.randint(4, 60)
            y = np.random.randint(4, 60)
            
            obs = world.observe(x, y)
            structure_before = compute_structure_reward(obs.unsqueeze(0))
            
            # Get action
            logits = agent(obs.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            # Apply action
            world.step(x, y, action.item())
            
            # Measure structure change AT SAME LOCATION
            obs_after = world.observe(x, y)
            structure_after = compute_structure_reward(obs_after.unsqueeze(0))
            
            # REWARD = CHANGE caused by action
            reward = structure_after - structure_before
            rewards.append(reward)
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        
        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)
        
        if episode % 50 == 0:
            avg = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            print(f"Ep {episode}: Reward={ep_reward:.4f}, Avg(50)={avg:.4f}")
    
    return agent, episode_rewards


if __name__ == '__main__':
    print("=== Training Reactive Agent v3 (Delta Structure Reward) ===\n")
    agent, rewards = train_reactive_agent_v3(num_episodes=1000, lr=2e-3)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    
    # Moving average
    window = 50
    if len(rewards) > window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label='Avg(50)')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Δ Structure')
    plt.title('Learning Curve (Structure Change Reward)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_v3.png')
    print("\nSaved training_v3.png")
    
    # Evaluate
    print("\n=== EVALUATION ===")
    
    def evaluate_agent(agent, name, num_episodes=20, epsilon=0.0):
        rewards = []
        for _ in range(num_episodes):
            world = World(hidden_channels=4, grid_size=64, patch_size=8)
            for _ in range(10):
                world.step()
            
            total_delta = 0
            for t in range(50):
                x = np.random.randint(4, 60)
                y = np.random.randint(4, 60)
                
                obs = world.observe(x, y)
                s_before = compute_structure_reward(obs.unsqueeze(0))
                
                if agent is None:
                    action = np.random.randint(0, 4)
                else:
                    action = agent.select_action(obs, epsilon=epsilon)
                
                world.step(x, y, action)
                
                obs_after = world.observe(x, y)
                s_after = compute_structure_reward(obs_after.unsqueeze(0))
                
                total_delta += (s_after - s_before)
            
            rewards.append(total_delta)
        
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"{name:20s}: {mean_r:.4f} ± {std_r:.4f}")
        return mean_r
    
    trained_score = evaluate_agent(agent, "Trained Agent", epsilon=0.0)
    random_score = evaluate_agent(None, "Random Baseline")
    
    print("\n" + "="*60)
    improvement = trained_score - random_score
    if improvement > 0.05:  # meaningful threshold
        print(f"✓✓✓ AGENT LEARNED! (+{improvement:.4f} structure vs random)")
        print("    → Task is solvable")
        print("    → Ready to test if PLANNING helps even more")
    elif improvement > 0:
        print(f"✓ Small improvement (+{improvement:.4f})")
        print("  → May need more training or better architecture")
    else:
        print(f"✗ No improvement ({improvement:.4f})")
        print("  → Check network architecture or learning rate")
    
    torch.save(agent.state_dict(), 'reactive_agent_v3.pt')
