# diagnostic_test.py
import torch
import numpy as np
from world import World
from random_agent import compute_local_entropy

def test_manual_control():
    """Can a fixed strategy reduce entropy better than random?"""
    
    print("=== DIAGNOSTIC: Testing Fixed Strategies ===\n")
    
    strategies = {
        "Random": lambda: np.random.randint(0, 4),
        "Always Inject (action=1)": lambda: 1,
        "Always Kill (action=2)": lambda: 2,
        "Always Seed (action=3)": lambda: 3,
        "Always Nothing (action=0)": lambda: 0,
    }
    
    num_episodes = 20
    
    for name, strategy in strategies.items():
        rewards = []
        for ep in range(num_episodes):
            world = World(hidden_channels=4, grid_size=64, patch_size=8)
            total_reward = 0
            
            for t in range(50):
                x = np.random.randint(4, 60)
                y = np.random.randint(4, 60)
                
                obs = world.observe(x, y)
                h_before = compute_local_entropy(obs.unsqueeze(0))
                
                action = strategy()
                world.step(x, y, action)
                
                obs_new = world.observe(x, y)
                h_after = compute_local_entropy(obs_new.unsqueeze(0))
                
                total_reward += (h_before - h_after)
            
            rewards.append(total_reward)
        
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"{name:25s}: {mean_r:6.3f} ± {std_r:.3f}")
    
    print("\n" + "="*60)
    print("If ALL strategies get negative reward:")
    print("  → Task might be impossible (chaos too strong)")
    print("If ANY strategy beats random significantly:")
    print("  → Task is learnable, just need better training")

if __name__ == '__main__':
    test_manual_control()
