import torch 

def compute_lyapunov(nca, grid, steps=50, epsilon=1e-5):
    nca.eval()

    grid_ref = grid.clone().detach()
    perturbations = torch.randn_like(grid_ref) * epsilon
    grid_pert = grid_ref+perturbations
    lyapunov_sum = 0.0
    divergence_hist = []

    with torch.no_grad():
        for _ in range(steps):
            grid_ref = nca(grid_ref)
            grid_pert = nca(grid_pert)

            diff = grid_pert - grid_ref
            distance = torch.norm(diff) 
            divergence_hist.append(distance.item())

            lyapunov_sum += torch.log(distance/epsilon)
            diff = diff/distance
            grid_pert = grid_ref + epsilon*diff 
    lyapunov_exponent = lyapunov_sum/steps 
    return lyapunov_exponent, divergence_hist


def spatial_entropy(grid, num_bins=64):
    x = grid[:, 0].flatten()
    hist = torch.histc(x, bins=num_bins, min=-2.0, max=2.0)
    prob = hist/hist.sum() 
    p = prob[prob>0]
    return -(p * torch.log(p)).sum()


def temporal_entropy(grids, num_bins=64):
    diffs = []
    for t in range(1, len(grids)):
        diffs.append((grids[t][:, 0]-grids[t-1][:, 0]).flatten())
    diffs = torch.cat(diffs)
    hist = torch.histc(diffs, bins=num_bins, min=-1.0, max=1.0)
    p = hist/hist.sum()
    p =p[p>0]
    return -(p*torch.log(p)).sum()

