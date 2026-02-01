import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 


class MCTSNode:
    def __init__(self, latent, prior, reward=0.0):
        self.latent = latent
        self.reward = reward 
        self.prior = prior
        self.visit_count = 0
        self.value_sum - 0.0 
        self.children = {}
        self.virtual_loss = 0.0 

    def value(self):
        if self.visit_count==0:
            return 0.0
        return self.value_sum/self.visit_count

class MCTS:
    def __init__(self, networks, num_actions=4, c_puct=1.0, virtual_loss=1.0):
        self.networks =networks
        self.num_actions=num_actions
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss
    
    def run(self, root_obs, num_simulations=50):
        with torch.no_grad():
            latent = self.networks.representation(root_obs)
            policy, value = self.networks.prediction(latent)
            policy = policy.squeeze(0).cpu().numpy()
            root = MCTSNode(latent, prior)
            for _ in range(num_simulations):
                self._simulate(root)
        return self._select_action(root)

    def _simulate(self, node):
        path = [node]
        current=node 
        while current.children:
            max_ucb = -float('inf')
            best_action = None 
            for a, child in current.children.items():
                ucb = self._ucb_score(current, child)
                if ucb>max_ucb:
                    max_ucb = ucb
                    best_action=a 
            current = current.children[best_action]
            path.append(current)
            current.virtual_loss += self.virtual_loss


        latent = current.latent 
        policy, value = self.networks.prediction(latent)
        policy = policy.squeeze(0).cpu().numpy()

        onehot = torch.zeros(1, self.num_actions)
        for a in range(self.num_actions):
            onehot[0, a] = 1.0

            next_latent, reward, value_prefix, _, _ = self.networks.recurrent_inference(latent, onehot)
            current_children[a] = MCTSNode(next_latent, policy[a], reward.item())

        self._backup(path, value.item())
        for n in path:
            n.virtual_loss -= self.virtual_loss

    def _ucb_score(self, parent, child):
        q = child.value()
        u = self.c_puct * child.prior * np.sqrt(parent.visit_count+1)/(1+child.visit_count)
        return q + u - child.virtual_loss

    def _backup(self, path, value):
        for node in reversed(path):
            node.visit_count +=1 
            node.value_sum += value
            value = node.reward + value 

    def _select_action(self, root):
        visits = np.array([root.children[a].visit_count for a in range(self.num_actions)])
        probs = visits/visits.sum()
        action = np.random.choice(self.num_actions, p=probs)
        return action, probs 



