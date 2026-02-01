import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from world import World
from random_agent import ReactiveAgent 
from p2 import compute_structure_reward
from main import EZNet

def run_and_record(agent, agent_name, num_steps=200, seed=42):
    """Run agent and record full history"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    world = World(hidden_channels=4, grid_size=64, patch_size=8)
    
    # Let world settle
    for _ in range(10):
        world.step()
    
    history = {
        'grids': [],
        'positions': [],
        'actions': [],
        'structures': [],
        'rewards': []
    }
    
    for t in range(num_steps):
        # Record current state
        history['grids'].append(world.grid[0, 0].clone().detach().cpu().numpy())
        
        # Choose position and action
        x = np.random.randint(4, 60)
        y = np.random.randint(4, 60)
        
        obs = world.observe(x, y)
        s_before = compute_structure_reward(obs.unsqueeze(0))
        
        if agent is None:  # Random
            action = np.random.randint(0, 4)
        elif isinstance(agent, EZNet):
            with torch.no_grad():
                if obs.dim()==3:
                    obs_batch = obs.unsqueeze(0)
                else:
                    obs_batch = obs 

                _, policy_logits, _ = agent.initial_inference(obs_batch)
                action = torch.argmax(policy_logits, dim=-1).item()
        else:  # Trained
            action = agent.select_action(obs, epsilon=0.0)
        
        history['positions'].append((x, y))
        history['actions'].append(action)
        
        # Step
        world.step(x, y, action)
        
        obs_after = world.observe(x, y)
        s_after = compute_structure_reward(obs_after.unsqueeze(0))
        
        reward = s_after - s_before
        history['structures'].append(s_after)
        history['rewards'].append(reward)
    
    # Final grid
    history['grids'].append(world.grid[0, 0].clone().detach().cpu().numpy())
    
    print(f"{agent_name}: Avg Structure = {np.mean(history['structures']):.4f}, "
          f"Avg Reward = {np.mean(history['rewards']):.4f}")
    
    return history

def create_comparison_video(random_history, trained_history, ez_history, filename='comparison.mp4'):
    """Create side-by-side video with 3 agents and metrics"""
    
    fig = plt.figure(figsize=(20, 10))
    
    # Grid layout: 2 rows, 6 columns
    gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)
    
    # Top row: 3 video panels (2 columns each)
    ax_random = fig.add_subplot(gs[0, :2])
    ax_trained = fig.add_subplot(gs[0, 2:4])
    ax_ez = fig.add_subplot(gs[0, 4:])
    
    # Bottom row: 2 metric plots (3 columns each)
    ax_structure = fig.add_subplot(gs[1, :3])
    ax_reward = fig.add_subplot(gs[1, 3:])
    
    # Initialize images
    im_random = ax_random.imshow(random_history['grids'][0], 
                                  cmap='viridis', vmin=-0.5, vmax=0.5)
    im_trained = ax_trained.imshow(trained_history['grids'][0], 
                                    cmap='viridis', vmin=-0.5, vmax=0.5)
    im_ez = ax_ez.imshow(ez_history['grids'][0],
                         cmap='viridis', vmin=-0.5, vmax=0.5)
    
    ax_random.set_title('Random Agent', fontsize=14, fontweight='bold')
    ax_trained.set_title('Trained Agent', fontsize=14, fontweight='bold')
    ax_ez.set_title('EZ Agent', fontsize=14, fontweight='bold')
    ax_random.axis('off')
    ax_trained.axis('off')
    ax_ez.axis('off')
    
    # Add colorbars
    plt.colorbar(im_random, ax=ax_random, fraction=0.046, pad=0.04)
    plt.colorbar(im_trained, ax=ax_trained, fraction=0.046, pad=0.04)
    plt.colorbar(im_ez, ax=ax_ez, fraction=0.046, pad=0.04)
    
    # Intervention markers
    rect_random = Rectangle((0, 0), 8, 8, fill=False, edgecolor='red', linewidth=2)
    rect_trained = Rectangle((0, 0), 8, 8, fill=False, edgecolor='red', linewidth=2)
    rect_ez = Rectangle((0, 0), 8, 8, fill=False, edgecolor='red', linewidth=2)
    ax_random.add_patch(rect_random)
    ax_trained.add_patch(rect_trained)
    ax_ez.add_patch(rect_ez)
    
    # Metrics setup
    num_steps = len(random_history['structures'])
    
    # Structure plot
    line_struct_random, = ax_structure.plot([], [], 'b-', alpha=0.7, label='Random', linewidth=2)
    line_struct_trained, = ax_structure.plot([], [], 'r-', alpha=0.7, label='Trained', linewidth=2)
    line_struct_ez, = ax_structure.plot([], [], 'g-', alpha=0.7, label='EZ', linewidth=2)
    
    max_struct = max(max(random_history['structures']), 
                     max(trained_history['structures']),
                     max(ez_history['structures']))
    
    ax_structure.set_xlim(0, num_steps)
    ax_structure.set_ylim(0, max_struct * 1.1)
    ax_structure.set_xlabel('Time Step', fontsize=12)
    ax_structure.set_ylabel('Structure Score', fontsize=12)
    ax_structure.set_title('Structure Over Time', fontsize=12, fontweight='bold')
    ax_structure.legend()
    ax_structure.grid(True, alpha=0.3)
    
    # Reward plot
    line_reward_random, = ax_reward.plot([], [], 'b-', alpha=0.7, label='Random', linewidth=2)
    line_reward_trained, = ax_reward.plot([], [], 'r-', alpha=0.7, label='Trained', linewidth=2)
    line_reward_ez, = ax_reward.plot([], [], 'g-', alpha=0.7, label='EZ', linewidth=2)
    
    reward_min = min(min(random_history['rewards']), 
                     min(trained_history['rewards']),
                     min(ez_history['rewards']))
    reward_max = max(max(random_history['rewards']), 
                     max(trained_history['rewards']),
                     max(ez_history['rewards']))
    
    ax_reward.set_xlim(0, num_steps)
    ax_reward.set_ylim(reward_min * 1.1, reward_max * 1.1)
    ax_reward.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_reward.set_xlabel('Time Step', fontsize=12)
    ax_reward.set_ylabel('Reward (Δ Structure)', fontsize=12)
    ax_reward.set_title('Reward Over Time', fontsize=12, fontweight='bold')
    ax_reward.legend()
    ax_reward.grid(True, alpha=0.3)
    
    # Text for cumulative stats
    text_random = ax_random.text(0.02, 0.98, '', transform=ax_random.transAxes,
                                 fontsize=9, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    text_trained = ax_trained.text(0.02, 0.98, '', transform=ax_trained.transAxes,
                                   fontsize=9, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    text_ez = ax_ez.text(0.02, 0.98, '', transform=ax_ez.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def update(frame):
        # Update grids
        im_random.set_array(random_history['grids'][frame])
        im_trained.set_array(trained_history['grids'][frame])
        im_ez.set_array(ez_history['grids'][frame])
        
        # Update intervention markers and text
        if frame > 0:
            x_r, y_r = random_history['positions'][frame-1]
            x_t, y_t = trained_history['positions'][frame-1]
            x_e, y_e = ez_history['positions'][frame-1]
            
            rect_random.set_xy((y_r - 4, x_r - 4))
            rect_trained.set_xy((y_t - 4, x_t - 4))
            rect_ez.set_xy((y_e - 4, x_e - 4))
            
            # Action labels
            action_names = ['Nothing', 'Inject', 'Kill', 'Seed']
            action_r = action_names[random_history['actions'][frame-1]]
            action_t = action_names[trained_history['actions'][frame-1]]
            action_e = action_names[ez_history['actions'][frame-1]]
            
            # Update text
            avg_struct_r = np.mean(random_history['structures'][:frame])
            avg_reward_r = np.mean(random_history['rewards'][:frame])
            text_random.set_text(f'Step: {frame}\nAction: {action_r}\n'
                                f'Avg Struct: {avg_struct_r:.3f}\n'
                                f'Avg Reward: {avg_reward_r:.3f}')
            
            avg_struct_t = np.mean(trained_history['structures'][:frame])
            avg_reward_t = np.mean(trained_history['rewards'][:frame])
            text_trained.set_text(f'Step: {frame}\nAction: {action_t}\n'
                                 f'Avg Struct: {avg_struct_t:.3f}\n'
                                 f'Avg Reward: {avg_reward_t:.3f}')
            
            avg_struct_e = np.mean(ez_history['structures'][:frame])
            avg_reward_e = np.mean(ez_history['rewards'][:frame])
            text_ez.set_text(f'Step: {frame}\nAction: {action_e}\n'
                            f'Avg Struct: {avg_struct_e:.3f}\n'
                            f'Avg Reward: {avg_reward_e:.3f}')
        
        # Update metrics
        if frame > 0:
            line_struct_random.set_data(range(frame), random_history['structures'][:frame])
            line_struct_trained.set_data(range(frame), trained_history['structures'][:frame])
            line_struct_ez.set_data(range(frame), ez_history['structures'][:frame])
            
            line_reward_random.set_data(range(frame), random_history['rewards'][:frame])
            line_reward_trained.set_data(range(frame), trained_history['rewards'][:frame])
            line_reward_ez.set_data(range(frame), ez_history['rewards'][:frame])
        
        return [im_random, im_trained, im_ez,
                rect_random, rect_trained, rect_ez,
                line_struct_random, line_struct_trained, line_struct_ez,
                line_reward_random, line_reward_trained, line_reward_ez,
                text_random, text_trained, text_ez]
    
    anim = animation.FuncAnimation(fig, update, frames=len(random_history['grids']),
                                   interval=50, blit=True)
    
    print(f"Saving video to {filename}...")
    anim.save(filename, writer='ffmpeg', fps=20, dpi=100)
    print("Done!")
    
    plt.close()

if __name__ == '__main__':
    print("=== Creating 3-Way Comparison Video ===\n")
    
    # Load trained agent
    print("Loading Trained Agent...")
    agent = ReactiveAgent(in_channels=5, num_actions=4)
    agent.load_state_dict(torch.load('reactive_agent_v3.pt'))
    agent.eval()
    
    # Load EZ agent
    print("Loading EZ Agent...")
    ez_agent = EZNet(observation_shape=(5,8,8),
                     num_actions=4,
                     num_channels=64,
                     num_blocks=2)
    ez_agent.load_state_dict(torch.load('ez_model.pt', map_location=torch.device('cpu')))
    ez_agent.eval()
    
    # Run all agents with same seed
    print("\nRunning Random Agent...")
    random_history = run_and_record(None, "Random Agent", num_steps=200, seed=42)
    
    print("Running Trained Agent...")
    trained_history = run_and_record(agent, "Trained Agent", num_steps=200, seed=42)
    
    print("Running EZ Agent...")
    ez_history = run_and_record(ez_agent, "EZ Agent", num_steps=200, seed=42)
    
    # Create video
    print("\nCreating comparison video...")
    create_comparison_video(random_history, trained_history, ez_history,
                           filename='random_vs_trained_vs_ez.mp4')
    
    print("\n✓ Video saved: random_vs_trained_vs_ez.mp4")
