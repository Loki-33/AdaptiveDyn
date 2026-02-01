import torch
import matplotlib.pyplot as plt
from nca import NCA
from metrics_fn import compute_lyapunov, spatial_entropy, temporal_entropy
import numpy as np

# Step 1: Run NCA with NO interventions
nca = NCA(grid_size=64, hidden_channels=4, fire_rate=0.5)
grid = nca.initialize_grid(1)
grid_hist = [grid.clone()]

print("Running NCA for 200 steps...")
for t in range(200):
    grid = nca(grid)
    grid_hist.append(grid.clone())
    
    if t % 50 == 0:
        plt.imshow(grid[0, 0].detach().cpu(), cmap='viridis')
        plt.title(f'Step {t}')
        plt.colorbar()
        plt.savefig(f'plots/step_{t}.png')
        plt.close()
        print(f"Step {t}: saved visualization")

# Step 2: Measure chaos
print("\n=== CHAOS METRICS ===")
lyap, divergence = compute_lyapunov(nca, grid_hist[0], steps=50)
print(f"Lyapunov exponent: {lyap}")

temp_ent = temporal_entropy(grid_hist)
print(f"Temporal entropy: {temp_ent}")

spat_ent = spatial_entropy(grid_hist[-1])
print(f"Spatial entropy: {spat_ent}")

# Step 3: Plot divergence
plt.plot(divergence)
plt.xlabel('Time Step')
plt.ylabel('Log Divergence')
plt.title('Lyapunov Divergence Over Time')
plt.savefig('plots/divergence.png')
plt.close()
print("\nSaved divergence.png")

