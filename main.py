import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from collections import deque 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import trange 
from world import World 
from nca import NCA


def compute_structure_reward(patch):
    """
    Reward = spatial structure (gradients)
    High gradients = patterns/edges = what we want
    Low gradients = uniform or noise = boring
    """
    visible = patch[0]  # only first channel
    
    # Compute spatial gradients
    grad_x = torch.abs(visible[1:, :] - visible[:-1, :])
    grad_y = torch.abs(visible[:, 1:] - visible[:, :-1])
    
    # Mean absolute gradient = structure score
    structure = (grad_x.mean() + grad_y.mean()) / 2.0
    
    return structure.item()

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x 
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out + identity
        out = F.relu(out)
        return out 

class RepresentationNet(nn.Module):
    def __init__(self, observation_shape=(5, 8, 8), num_blocks=2, num_channels=64):
        super().__init__()
        in_channel = observation_shape[0]
        self.conv = nn.Conv2d(in_channel, num_channels, 3, padding=1)
        self.resblock = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])
        self.bn = nn.BatchNorm2d(num_channels)
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))

        for block in self.resblock:
            x = block(x)
        return x 


class DynamicsNet(nn.Module):
    def __init__(self, num_channels=64, num_actions=4, num_blocks=2):
        super().__init__()
        self.num_acitons = num_actions 

        self.action_encoder = nn.Linear(num_actions, num_channels)

        self.conv = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)

        self.resblock = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])
        self.reward_conv = nn.Conv2d(num_channels, 1, 1)
        self.reward_fc = nn.Linear(64, 1)

    def forward(self, latent, action_onehot):
        B, C, H, W = latent.shape 

        action_embd = self.action_encoder(action_onehot)
        action_embd = action_embd.view(B, C, 1, 1).expand(B, C, H, W)

        x = latent + action_embd
        x = F.relu(self.bn(self.conv(x)))

        for block in self.resblock:
            x = block(x)

        next_latent = x 
        reward_feat = F.relu(self.reward_conv(x))
        reward_feat = reward_feat.view(B, -1)
        reward = self.reward_fc(reward_feat)

        return next_latent, reward


class PredictionNet(nn.Module):
    def __init__(self, num_channels=64, num_actions=4):
        super().__init__()
        self.policy_conv = nn.Conv2d(num_channels, 2, 1)
        self.policy_fc = nn.Linear(2*8*8, num_actions)

        self.value_conv = nn.Conv2d(num_channels, 1, 1)
        self.value_fc = nn.Linear(1*8*8, 1)

    def forward(self, latent):
        B = latent.size(0)

        policy_feat = F.relu(self.policy_conv(latent))
        policy_feat = policy_feat.view(B, -1)
        policy_logits = self.policy_fc(policy_feat)

        value_features = F.relu(self.value_conv(latent))
        value_features = value_features.view(B, -1)
        value = self.value_fc(value_features)

        return policy_logits, value 


class ProjectionNet(nn.Module):
    def __init__(self, num_channels=64, proj_dim=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projector = nn.Sequential(
                nn.Linear(num_channels, proj_dim),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
                )

        self.predictor = nn.Sequential(
                nn.Linear(proj_dim, proj_dim//2),
                nn.BatchNorm1d(proj_dim//2),
                nn.ReLU(),
                nn.Linear(proj_dim//2, proj_dim)
                )

    def forward(self, latent):
        x = self.pool(latent).squeeze(-1).squeeze(-1)

        z = self.projector(x)
        p = self.predictor(z)
        return z, p 


class EZNet(nn.Module):
    def __init__(self, observation_shape=(5, 8, 8), num_actions=4, num_channels=64, num_blocks=2):
        super().__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.representation = RepresentationNet(observation_shape, num_blocks, num_channels)
        self.dynamics = DynamicsNet(num_channels, num_actions, num_blocks)
        self.prediction = PredictionNet(num_channels, num_actions)
        self.projection = ProjectionNet(num_channels, proj_dim=64)

    def initial_inference(self, obs):
        latent = self.representation(obs)
        policy_logits, value = self.prediction(latent)
        return latent, policy_logits, value 


    def recurrent_inference(self, latent, action):
        action_onehot = F.one_hot(action, self.num_actions).float()
        next_latent, reward = self.dynamics(latent, action_onehot)
        policy_logits, value = self.prediction(next_latent)
        return next_latent, reward, policy_logits, value 



class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')
    
    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    
    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTSNode:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.hidden_state = None 
        self.reward = 0 
        
    def expanded(self):
        return len(self.children)>0

    def value(self):
        if self.visit_count==0:
            return 0 
        return self.value_sum/self.visit_count


class MCTS:
    def __init__(self, network, num_simulations=50, discount=0.99):
        self.network = network
        self.num_simulations = num_simulations
        self.discount = discount
        self.device = next(network.parameters()).device
        self.pb_c_base = 19652
        self.pb_c_init = 1.25 

    def run(self, observation, add_exploration_noise=False, temperature=1.0):
        root = MCTSNode(0)
        # print('YO')
        with torch.no_grad():
            if observation.dim() == 3:
                observation = observation.unsqueeze(0)
            observation = observation.to(self.device)
            latent, policy_logits, value = self.network.initial_inference(observation)

            root.hidden_state = latent 

            policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

            if add_exploration_noise:
                noise = np.random.dirichlet([0.3]*self.network.num_actions)
                policy_probs=0.75*policy_probs + 0.25 * noise 

            self._expand_node(root, policy_probs, value.item())

        min_max_stats = MinMaxStats()
        min_max_stats.update(value.item())
        # print('YO2')
        for i in range(self.num_simulations):
            # print(f'SIM: {i+1}')
            node = root 
            search_path = [node]
            current_latent = root.hidden_state

            while node.expanded():
                action, node = self._select_child(node, min_max_stats)
                search_path.append(node)

            with torch.no_grad():
                action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
                next_latent, reward, policy_logits, value = self.network.recurrent_inference(
                        current_latent, action_tensor
                        )

            node.hidden_state = next_latent 
            node.reward = reward.item()
            current_latent = next_latent


            policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            self._expand_node(node, policy_probs, value.item())

            self._backpropagate(search_path, value.item(), min_max_stats)


        visit_counts = np.array([
            root.children[a].visit_count if a in root.children else 0
            for a in range(self.network.num_actions)
            ])

        if temperature == 0:
            # Greedy
            action = np.argmax(visit_counts)
            action_probs = np.zeros(self.network.num_actions)
            action_probs[action] = 1.0
        else:
            # Sample proportional to visit^(1/T)
            visit_counts_temp = visit_counts ** (1.0 / temperature)
            action_probs = visit_counts_temp / visit_counts_temp.sum()
            action = np.random.choice(self.network.num_actions, p=action_probs)

        
        root_value = root.value()
        # if np.random.rand() < 0.05:  # don't spam every step
        #     print("\n[MCTS]")
        #     print("Visit counts:", visit_counts)
        #     print("Action probs:", action_probs)
        #     print("Root value:", root_value)
        return action, action_probs, root_value


    def _expand_node(self, node, policy_probs, value):
        if node.expanded():
            return 
            
        for action in range(self.network.num_actions):
            node.children[action] = MCTSNode(prior=policy_probs[action])
        
    def _select_child(self, node, min_max_stats):
        """Select child using PUCT formula"""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        # Total visit count for exploration term
        total_visits = sum(child.visit_count for child in node.children.values())
        
        for action, child in node.children.items():
            # UCB score (official formula from paper)
            ucb_score = self._ucb_score(node, child, min_max_stats, total_visits)
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _ucb_score(self, parent, child, min_max_stats, total_visits):
        """Compute UCB score (official EfficientZero formula)"""
        # Normalized Q value
        q_value = child.reward + self.discount * child.value()
        normalized_q = min_max_stats.normalize(q_value)
        
        # Exploration term
        pb_c = np.log((total_visits + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        exploration = pb_c * child.prior * np.sqrt(total_visits) / (child.visit_count + 1)
        
        return normalized_q + exploration
    
    def _backpropagate(self, search_path, value, min_max_stats):
        """Backpropagate value through search path"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            
            # Update min-max stats
            q_value = node.reward + self.discount * value
            min_max_stats.update(q_value)
            
            # Discount for parent
            value = node.reward + self.discount * value


class ReplayBuffer:
    def __init__(self, capacity=1000, k_steps=5, discount=0.99):
        self.buffer = deque(maxlen=capacity)
        self.k_steps = k_steps
        self.discount = discount
        self.total_steps = 0

    def add(self, trajectory):
        self.buffer.append({
            'trajectory': trajectory,
            'age': self.total_steps 
            })

        self.total_steps += 1 

    def sample(self, batch_size):
        if len(self.buffer)<batch_size:
            indices = range(len(self.buffer))
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        batch = []
        for idx in indices:
            entry = self.buffer[idx]
            trajectory = entry['trajectory']
            age = self.total_steps - entry['age']

            max_start = len(trajectory['observations']) - self.k_steps-1
            if max_start <= 0:
                continue 

            start_pos = np.random.randint(0, max_start)
            observations = trajectory['observations'][start_pos:start_pos + self.k_steps + 1]
            actions = trajectory['actions'][start_pos:start_pos + self.k_steps]
            rewards = trajectory['rewards'][start_pos:start_pos + self.k_steps]
            root_values = trajectory['root_values'][start_pos:start_pos + self.k_steps + 1]
            policies = trajectory['policies'][start_pos:start_pos + self.k_steps]

            value_prefixes = self._compute_value_prefixes(rewards, root_values, age)
            
            batch.append({
                'observations': torch.stack(observations),
                'actions': torch.tensor(actions, dtype=torch.long),
                'rewards': torch.tensor(rewards, dtype=torch.float32),
                'value_prefixes': value_prefixes,
                'root_values': torch.tensor(root_values, dtype=torch.float32),
                'policies': torch.stack(policies)
            })
        
        return batch


    def _compute_value_prefixes(self, rewards, root_values, age):
        if age>1000:
            n=1 
        else:
            n = min(5, len(rewards))

        value_prefixes = []
        for t in range(len(rewards)):
            G = 0
            for i in range(min(n, len(rewards)-t)):
                G += (self.discount ** i) * rewards[t+i]

            if t+n < len(root_values):
                G += (self.discount**n) * root_values[t+n]

            value_prefixes.append(G)

        return torch.tensor(value_prefixes, dtype=torch.float32)

    def __len__(self):
        return len(self.buffer)



def augment_obs(obs):
    if torch.rand(1).item()<0.5:
        obs = torch.flip(obs, [-1])

    return obs 



def consistency_loss(z1, p1, z2, p2):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    p1 = F.normalize(p1, dim=-1)
    p2 = F.normalize(p2, dim=-1)

    loss = -(F.cosine_similarity(p1, z2.detach(), dim=-1).mean()+
             F.cosine_similarity(p2, z1.detach(), dim=-1).mean())*0.5 

    return loss 


def collect_episode(network, mcts, world, num_steps=50, temperature=1.0, add_noise=False):
    trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'root_values': [],
            'policies': []
            }

    for t in range(num_steps):
        x = np.random.randint(4, 60)
        y = np.random.randint(4, 60)
        obs = world.observe(x, y).detach().to(next(network.parameters()).device)


        s_before = compute_structure_reward(obs.unsqueeze(0))

        action, policy, root_value = mcts.run(obs, add_exploration_noise=add_noise,
                                              temperature=temperature)
        world.step(x, y, action)

        obs_after = world.observe(x,y).detach()

        s_after = compute_structure_reward(obs_after.unsqueeze(0))
        reward = s_after - s_before 
       
        if t % 10 == 0:
            print(f"[STEP {t}] Reward: {reward:.4f} | Structure Δ: {(s_after - s_before):.4f}")

        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['root_values'].append(root_value)
        trajectory['policies'].append(torch.tensor(policy, dtype=torch.float32))


    final_obs = world.observe(x, y).detach().to(next(network.parameters()).device)

    with torch.no_grad():
        _, _, final_value = network.initial_inference(final_obs.unsqueeze(0))
        trajectory['root_values'].append(final_value.item())
        trajectory['observations'].append(final_obs)


    return trajectory

def train(network, num_episodes=300, num_sims=10, buffer_size=200, 
          batch_size=4, k_steps=5, lr=1e-3, weight_decay=1e-4):
    device = next(network.parameters()).device
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    replay_buffer = ReplayBuffer(capacity=buffer_size, k_steps=k_steps)
    mcts = MCTS(network, num_simulations=num_sims)

    episode_rewards = []
    for episode in trange(num_episodes):
        print(f"EPISODE {episode+1}")
        if episode < 100:
            temperature = 1.0

        elif episode < 200:
            temperature=0.5

        else:
            temperature = 0.25 


        world = World(hidden_channels=4, grid_size=64, patch_size=8)
        # print('WORLD CREATED')
        for _ in range(10):
            world.step()
        # print('WORLD WARMUP DONE')
        
        trajectory = collect_episode(network, mcts, world, num_steps=50,
                                     temperature=temperature, add_noise=(episode<100))
        # print("TRAJECTORY COLLECTED")
        replay_buffer.add(trajectory)
        # print("TRAJECTORY ADDED TO BUFFER")
        ep_reward = sum(trajectory['rewards'])
        episode_rewards.append(ep_reward)
        # print("EPISODE REWARD APPENDED")
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            # print(f"[BUFFER] Sampling batch | Batch size: {len(batch)}")
            total_loss = 0
            for sample in batch:
                observations = sample['observations'].to(device)
                actions = sample['actions'].to(device)
                value_prefixes = sample['value_prefixes'].to(device)
                policies = sample['policies'].to(device)
                rewards = sample['rewards'].to(device)

                latent, policy_logits, value = network.initial_inference(observations[:1])
                
                # Losses for initial step
                policy_loss = F.cross_entropy(policy_logits, policies[:1])
                value_loss = F.mse_loss(value, value_prefixes[:1].unsqueeze(-1))
                reward_loss =0.0
                # Unroll k steps
                for k in range(min(k_steps, len(actions))):
                    # Recurrent inference
                    action = actions[k:k+1]
                    latent, reward_pred, policy_logits, value = network.recurrent_inference(latent, action)
                    
                    # Target
                    target_policy = policies[k:k+1] if k < len(policies) else policies[-1:]
                    idx = min(k + 1, value_prefixes.size(0) - 1)
                    target_value = value_prefixes[idx:idx+1].unsqueeze(-1)

                    target_reward = rewards[k:k+1].unsqueeze(-1)
                    
                    # Accumulate losses
                    policy_loss += F.cross_entropy(policy_logits, target_policy)
                    value_loss += F.mse_loss(value, target_value)
                    reward_loss += F.mse_loss(reward_pred, target_reward)
                    
                    total_loss += policy_loss + value_loss + reward_loss
                
                # Consistency loss (augmentation)
                obs_aug1 = torch.stack([augment_obs(o) for o in sample['observations'][:2]])
                obs_aug2 = torch.stack([augment_obs(o) for o in sample['observations'][:2]])
                
                latent1 = network.representation(obs_aug1)
                latent2 = network.representation(obs_aug2)
                
                z1, p1 = network.projection(latent1)
                z2, p2 = network.projection(latent2)
                
                consist_loss = consistency_loss(z1, p1, z2, p2)
                
                total_loss += 0.1 * consist_loss
            
            # Optimize
            total_loss = total_loss / len(batch)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            optimizer.step()
        
        if episode % 20 == 0:
            avg = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            print(f"Ep {episode}: Reward={ep_reward:.4f}, Avg(20)={avg:.4f}, "
                  f"Buffer={len(replay_buffer)}, Temp={temperature:.2f}")
    
    return episode_rewards


if __name__ == '__main__':
    network = EZNet(
            observation_shape=(5, 8, 8),
            num_actions=4,
            num_channels=64,
            num_blocks=2
            )

    print('\nTraining...')
    network = network.to('cpu')
    rewards = train(
            network,
            num_episodes=300,
            num_sims=10,
            buffer_size=150,
            batch_size=5,
            k_steps=5,
            lr=1e-3
            )
        
    torch.save(network.state_dict(), 'ez_model.pt')
    print('MODEL SAVED!!!!!!')
        
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    window=20
    if len(rewards)>window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 'r--', linewidth=2, label='Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Efficient Zero Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ez_training.png')
    print('\nSaved!!')
