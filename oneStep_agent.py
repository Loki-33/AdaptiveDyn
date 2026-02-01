import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt 
from random_agent import ReactiveAgent 
from world import World 
from p2 import compute_structure_reward 
import torch.optim as optim 

class Model(nn.Module):
    def __init__(self, in_channels=5, num_actions=4):
        super().__init__()
        self.num_actions = num_actions
        
        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU()
                )
        self.action_embd = nn.Embedding(num_actions, 64)

        self.decoder = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, in_channels, 3, padding=1)
                )

    def forward(self, obs, action):
        batch_size = obs.size(0)
        h,w = obs.size(2), obs.size(3)

        obs_feat = self.encoder(obs)

        action_feat = self.action_embd(action)
        action_feat = action_feat.view(batch_size, 64, 1, 1)
        action_feat = action_feat.expand(batch_size, 64, h, w)

        combined = obs_feat + action_feat 

        delta = self.decoder(combined)
        next_obs = obs+delta 
        return next_obs


class Agent(nn.Module):
    def __init__(self, in_channels=5, num_actions=4):
        super().__init__()

        self.dynamics = Model(in_channels, num_actions)
        self.num_actions = num_actions

    def select_action(self, obs, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)

        with torch.no_grad():
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)

            best_action = 0
            best_structure = -float('inf')

            for action_id in range(self.num_actions):
                action = torch.tensor([action_id], dtype=torch.long)
                predicted_next_obs = self.dynamics(obs, action)

                predicted_structure = compute_structure_reward(predicted_next_obs)

                if predicted_structure > best_structure:
                    best_structure = predicted_structure
                    best_action = action_id 

            return best_action


def collect_data(num_episodes=100, steps_per_episode=50):
    dataset = {
            'obs': [],
            'actions': [],
            'next_obs': []
            }

    for ep in range(num_episodes):
        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        
        # warmup 
        for _ in range(10):
            world.step()

        for t in range(steps_per_episode):
            x = np.random.randint(4, 60)
            y = np.random.randint(4, 60)

            obs = world.observe(x,y).detach()
            action = np.random.randint(0, 4)
            world.step(x, y, action)
            next_obs = world.observe(x, y).detach()
            dataset['obs'].append(obs)
            dataset['actions'].append(action)
            dataset['next_obs'].append(next_obs)

        if (ep+1) % 20 == 0:
            print(f"Collected {ep+1}/{num_episodes} episodes")

    dataset['obs'] = torch.stack(dataset['obs'])
    dataset['actions'] = torch.tensor(dataset['actions'], dtype=torch.long)
    dataset['next_obs'] = torch.stack(dataset['next_obs'])

    return dataset 

def train_model(model, dataset, num_epochs=50, batch_size=64, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_samples = len(dataset['obs'])

    indices = list(range(num_samples))
    losses = []

    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]

            obs = dataset['obs'][batch_idx]
            actions = dataset['actions'][batch_idx]
            next_obs = dataset['next_obs'][batch_idx]

            pred_next_obs = model(obs, actions)

            loss = F.mse_loss(pred_next_obs, next_obs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss/num_batches
        losses.append(avg_loss)

        if (epoch+1) %10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")

    return losses 


def train_planning(agent, num_episodes=500, lr=1e-3):
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    episode_rewards = []
    for episode in range(num_episodes):
        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        
        for _ in range(10):
            world.step()


        transitions = {
                'obs': [],
                'actions': [],
                'next_obs': [],
                'rewards': []
                }

        for t in range(50):
            x = np.random.randint(4, 60)
            y = np.random.randint(4, 60)

            obs = world.observe(x,y)
            s_before = compute_structure_reward(obs.unsqueeze(0))
            epsilon = max(0.1, 1.0-episode/300)
            action = agent.select_action(obs, epsilon=epsilon)

            world.step(x, y, action)

            next_obs = world.observe(x, y)
            s_after = compute_structure_reward(next_obs.unsqueeze(0))

            reward = s_after - s_before 

            transitions['obs'].append(obs)
            transitions['actions'].append(action)
            transitions['next_obs'].append(next_obs)
            transitions['rewards'].append(reward)


        if len(transitions['obs']) > 0:
            obs_batch = torch.stack(transitions['obs'])
            actions_batch = torch.tensor(transitions['actions'], dtype=torch.long)
            next_obs_batch = torch.stack(transitions['next_obs'])

            pred_next_obs = agent.dynamics(obs_batch, actions_batch)
            losss = F.mse_loss(pred_next_obs, next_obs_batch)

            optimizer.zero_grad()
            losss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()


        ep_reward = sum(transitions['rewards'])
        episode_rewards.append(ep_reward)

        if episode%50==0:
            avg = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            print(f"Ep {episode}: Reward={ep_reward:.4f}, Avg(50)={avg:.4f}, ε={epsilon:.3f}")
        
    return episode_rewards

if __name__ == '__main__':
    dataset = collect_data(num_episodes=50, steps_per_episode=50)
    
    model = Model(in_channels=5, num_actions=4)
    losses = train_model(model, dataset, num_epochs=100, batch_size=8)

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Dynamics Model Training')
    plt.grid(True, alpha=0.3)
    plt.savefig('dynamics_training.png')
    print("Saved dynamics_training.png")

    planning_agent = Agent(in_channels=5, num_actions=4)
    planning_agent.dynamics = model 

    rewards = train_planning(planning_agent, num_episodes=500, lr=5e-4)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3)
    window = 50
    if len(rewards) > window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('One-Step Planning Agent Training')
    plt.grid(True, alpha=0.3)
    plt.savefig('planning_training.png')
    print("Saved planning_training.png")

    def evaluate(agent, name, num_eps=20):
        rewards = []
        for _ in range(num_eps):
            world = World(hidden_channels=4, grid_size=64, patch_size=8)
            for _ in range(10):
                world.step()
            
            total = 0
            for t in range(50):
                x = np.random.randint(4, 60)
                y = np.random.randint(4, 60)
                
                obs = world.observe(x, y)
                s_before = compute_structure_reward(obs.unsqueeze(0))
                
                if agent is None:
                    action = np.random.randint(0, 4)
                else:
                    action = agent.select_action(obs, epsilon=0.0)
                
                world.step(x, y, action)
                
                obs_after = world.observe(x, y)
                s_after = compute_structure_reward(obs_after.unsqueeze(0))
                
                total += (s_after - s_before)
            
            rewards.append(total)
        
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"{name:20s}: {mean_r:.4f} ± {std_r:.4f}")
        return mean_r
    
    planning_score = evaluate(planning_agent, "Planning Agent")
    
    # Load reactive agent for comparison
    reactive_agent = ReactiveAgent(in_channels=5, num_actions=4)
    reactive_agent.load_state_dict(torch.load('reactive_agent_v3.pt'))
    reactive_agent.eval()
    reactive_score = evaluate(reactive_agent, "Reactive Agent")
    
    random_score = evaluate(None, "Random Baseline")
    
    print("\n" + "="*70)
    print("RESULTS:")
    print(f"  Random:   {random_score:.4f}")
    print(f"  Reactive: {reactive_score:.4f} (+{reactive_score - random_score:.4f})")
    print(f"  Planning: {planning_score:.4f} (+{planning_score - random_score:.4f})")
    print("="*70)
    
    
    improvement = planning_score - reactive_score
    if improvement > 0.05:
        print(f"\n PLANNING HELPS! (+{improvement:.4f} over reactive)")
    elif improvement > 0:
        print(f"\n Small improvement (+{improvement:.4f})")
    else:
        print(f"\n Planning doesn't help ({improvement:.4f})")
    
    torch.save(planning_agent.state_dict(), 'planning_agent.pt')
