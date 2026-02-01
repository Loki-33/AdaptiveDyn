import torch 
import torch.nn as nn 
import torch.nn.functional as F
from metrics_fn import compute_lyapunov, spatial_entropy, temporal_entropy 

class NCA(nn.Module):
    def __init__(self, grid_size, hidden_channels, fire_rate):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_channels = hidden_channels
        self.fire_rate = fire_rate
        self.total_channels = hidden_channels+1
        # (identity, sobel_x, sobel_y) * total_channels
        self.perception_channels = self.total_channels * 3
        
        self.update_net = nn.Sequential(
            nn.Conv2d(self.perception_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, self.total_channels, kernel_size=1)
        )

        self.reg_perception_filters()
        self.chaos_scale = 0.05

    def reg_perception_filters(self):
        identity = torch.tensor([[0., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 0.]])
        sobel_x = torch.tensor([[1., 0., -1.],
                                [2., 0., -2.],
                                [1., 0., -1.]]) / 8.0

        sobel_y = torch.tensor([[1.,  2.,  1.],
                                [0.,  0.,  0.],
                               [-1., -2., -1.]]) / 8.0
        filters = torch.stack([identity, sobel_x, sobel_y])[:, None, :, :]
        self.register_buffer('perception_filters', filters)

    def initialize_grid(self, batch_size):
        grid = torch.zeros((batch_size, self.total_channels, self.grid_size, self.grid_size))
        center = self.grid_size//2 
        grid[:,:,  center-1:center+2, center-1:center+2] = torch.randn(batch_size, self.total_channels, 3, 3)*0.1 
        return grid 

    def percieve(self, grid):
        batch_size = grid.size(0)
        perception = []
        for i in range(self.total_channels):
            channel = F.pad(grid[:, i:i+1, :, :], (1,1,1,1), mode='circular')
            filtered = F.conv2d(channel, self.perception_filters, padding=0)
            perception.append(filtered)

        perception = torch.cat(perception, dim=1)
        return perception

    def apply_stochastic_mask(self, update):
        mask = (torch.rand(update.shape[0], 1, update.shape[2], update.shape[3], device=update.device)<self.fire_rate).float()
        return update*mask 

    def forward(self, grid):
        perception = self.percieve(grid)
        update = self.update_net(perception)
        update = self.apply_stochastic_mask(update)

        x= torch.norm(grid, dim=1, keepdim=True)
        x = F.pad(x, (1,1,1,1), mode='circular')
        grad_x = F.conv2d(x, self.perception_filters[1:2], padding=0)
        grad_y = F.conv2d(x, self.perception_filters[2:3], padding=0)
        grad_mag=torch.sqrt(grad_x**2+grad_y**2+1e-8)
        
        noise = torch.randn_like(update) * grad_mag * self.chaos_scale
        update = update + noise 
        new_grid = grid + update
        norm = torch.norm(new_grid, dim=1, keepdim=True)
        new_grid = new_grid/(1+0.1*norm)
        return new_grid

    def get_visible_channel(self, grid):
        return grid[:, 0:1, :, :]

#if __name__ == '__main__':
#    import matplotlib.pyplot as plt 
#    import matplotlib.cm as cm 
#    import cv2
#    import numpy as np 
#    HIDDEN_CHANNELS = 4 
#    GRID_SIZE = 64 
#    FIRE_RATE = 0.1
#    batch_size = 1
#
#    nca = NCA(GRID_SIZE, HIDDEN_CHANNELS, FIRE_RATE)
#    grid = nca.initialize_grid(batch_size)
#    optimizer = torch.optim.Adam(nca.parameters(), lr=1e-3)
#    grid_hist = []
#    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for mp4
#    out = cv2.VideoWriter('grid_animation.mp4', fourcc, 20.0, (64,64), isColor=False)
#    for t in range(200):
#        grid = nca(grid)
#        grid_hist.append(grid.clone().cpu())
#        
#        frame = (grid[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)        
#        out.write(frame)
#        if t%20 ==0:
#            plt.imshow(grid[0, 0].detach().cpu(), cmap='viridis')
#            plt.title(f'Step {t}')
#            plt.colorbar()
#            plt.savefig(f'figures/fig_{t}.png', dpi=150)
#            plt.close() 
#    out.release()
#    print('Video Saved')
#    temp_entropy = temporal_entropy(grid_hist)
#    print("Temporal entropy:", temp_entropy)
#
#    spat_ent = spatial_entropy(grid_hist[-1])
#    print("Spatial entropy:", spat_ent)
#
#    lyap, divergence = compute_lyapunov(nca, grid_hist[0])
#    print("Lyapunov exponent:", lyap)
#
#    plt.plot(divergence)
#    plt.xlabel('Time Step')
#    plt.ylabel('Divergence')
#    plt.title('Lyapunov Divergence')
#    plt.savefig('figures/divergence.png')
#    plt.close()
