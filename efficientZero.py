import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class RepresentationNet(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size,=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, latent_dim, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.latent_dim = latent_dim
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)

class DynamicsNet(nn.Module):
    def __init__(self, latent_dim=128, num_actions=4):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim+num_actions, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.reward_head = nn.Linear(latent_dim, 1)
        self.value_head = nn.Linear(latent_dim, 1)

    def forward(self, latent, a_onehot):
        x = torch.cat([latent, a_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        next_latent = F.relu(self.fc2(x))
        reward = self.reward_head(next_latent)
        value_prefix = self.value_head(next_latent)
        return next_latent, reward, value_prefix

class PredictionNet(nn.Module):
    def __init__(self, latent_dim=128, num_actions=4):
        super().__init__()
        self.fc_policy = nn.Linear(latent_dim, num_actions)
        self.fc_value = nn.Linear(latent_dim, 1)

    def forward(self, latent):
        policy_logits = self.fc_policy(latent)
        policy = F.softmax(policy_logits, dim=1)
        value = self.fc_value(latent_dim)
        return policy, value


class SimSiamHead(nn.Module):
    def __init__(self, latent_dim=128, proj_dim=64, pred_dim=32):
        super().__init__()
        self.projector = nn.Sequential(
                nn.Linear(latent_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
                )
        self.predictor = nn.Sequential(
                nn.Linear(proj_dim, pred_dim),
                nn.ReLU(),
                nn.Linear(pred_dim, proj_dim)
                )

    def forward(self, latent):
        z=self.projector(latent)
        p=self.predictor(z)
        return z, p 


class EZNet(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, num_actions=4):
        super().__init__()
        self.representation=RepresentationNet(in_channels, latent_dim)
        self.dynamics = DynamicsNet(latent_dim, num_actions)
        self.prdiction = PredictionNet(latent_dim, num_actions)
        self.siam = SimSiamHead(latent_dim)

    def initial_inference(self, obs):
        latent = self.representation(obs)
        policy, value = self.prediction(latent)
        return latent, policy, value


    def recurrent_inference(self, latent, a_onehot):
        next_latent, reward, value_prefix = self.dynamics(latent, a_onehot)
        policy, value = self.prediction(next_latent)
        return next_latent, reward, value_prefix, policy, value

