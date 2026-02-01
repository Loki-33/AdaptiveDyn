import torch
import numpy as np
from world import World
from p2 import compute_structure_reward

def test_action_effects():
    
    print("=== Testing What Each Action Does to Structure ===\n")
    
    num_trials = 100
    actions = {0: "Nothing", 1: "Inject Energy", 2: "Kill", 3: "Seed Hidden"}
    
    for action_id, action_name in actions.items():
        structure_before = []
        structure_after = []
        
        for _ in range(num_trials):
            world = World(hidden_channels=4, grid_size=64, patch_size=8)
            
            # Let world evolve a bit first
            for _ in range(20):
                world.step()
            
            # Random location
            x = np.random.randint(4, 60)
            y = np.random.randint(4, 60)
            
            # Measure structure BEFORE action
            obs_before = world.observe(x, y)
            s_before = compute_structure_reward(obs_before.unsqueeze(0))
            structure_before.append(s_before)
            
            # Apply action
            world.step(x, y, action_id)
            
            # Measure structure AFTER action
            obs_after = world.observe(x, y)
            s_after = compute_structure_reward(obs_after.unsqueeze(0))
            structure_after.append(s_after)
        
        mean_before = np.mean(structure_before)
        mean_after = np.mean(structure_after)
        delta = mean_after - mean_before
        
        effect = "INCREASES" if delta > 0 else "DECREASES"
        print(f"Action {action_id} ({action_name:15s}): {mean_before:.4f} → {mean_after:.4f} (Δ={delta:+.4f}) {effect}")
    
    print("\n" + "="*70)
    print("If all actions DECREASE structure:")
    print("  → Your actions are destructive, need to redesign them")
    print("If some actions INCREASE structure:")
    print("  → Task is learnable, but network/training might be the issue")

if __name__ == '__main__':
    test_action_effects()
