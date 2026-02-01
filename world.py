import torch 
import torch.nn.functional as F 
import numpy as np 
from nca import NCA 
from metrics_fn import compute_lyapunov, spatial_entropy, temporal_entropy

class World:
    def __init__(self,  hidden_channels, grid_size=64, patch_size=8, fire_rate=0.1):
        self.nca = NCA(grid_size, hidden_channels, fire_rate)
        self.grid_size = grid_size
        self.patch_size = patch_size
        
        self.grid = self.nca.initialize_grid(1)
        self.grid_hist = [self.grid.clone()]

    def observe(self, x, y):
        half = self.patch_size//2 
        xs = (torch.arange(x-half, x+half)%self.grid_size)
        ys = (torch.arange(y-half, y+half)%self.grid_size)

        patch = self.grid[:, :, xs[:, None], ys[None, :]]
        return patch.squeeze(0)

    def apply_action(self, x, y, action):
        half = 1
        xs = (torch.arange(x-half, x+half)%self.grid_size)
        ys = (torch.arange(y-half, y+half)%self.grid_size)

        if action==1: # inject Energy
            self.grid[:, 0, xs[:, None], ys[None, :]] += 1
        elif action == 2:  # kill cells
            self.grid[:, 0, xs[:, None], ys[None, :]] *= 0.0
        elif action == 3:  # plant seeds in hidden channels
            num_hidden = self.nca.hidden_channels
            self.grid[:, 1:1+num_hidden, xs[:, None], ys[None, :]] += torch.randn(1, num_hidden, len(xs), len(ys)) * 0.05
        
        self.grid = torch.clamp(self.grid, -2.0, 2.0)

    def step(self, x=None, y=None, action=None):
        if x is not None and action is not None:
            self.apply_action(x, y, action)

        self.grid = self.nca(self.grid)
        self.grid_hist.append(self.grid.clone())

        return self.grid.clone()

#if __name__ == '__main__':
#    world = World(hidden_channels=4, grid_size=64, patch_size=8)
#    for t in range(50):
#        x, y = np.random.randint(0, 64), np.random.randint(0, 64)
#        action = np.random.randint(0, 4)
#        world.step(x, y, action)
#    print("Temporal Entropy: ", temporal_entropy(world.grid_hist))
#    print("Spatial Entropy: ", spatial_entropy(world.grid_hist[-1]))
