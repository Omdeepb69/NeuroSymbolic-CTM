import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

print("==================================================")
print(" GCF PROTOTYPE: DIFFERENTIABLE COST FIELDS ")
print("==================================================")

# Ensure we are in a dedicated research directory
os.makedirs("results", exist_ok=True)

class GCFManifold(nn.Module):
    def __init__(self, size=64):
        super().__init__()
        self.size = size
        # Initialize the raw parameters. We use a base value of 1.0
        self.raw_cost = nn.Parameter(torch.ones(size, size) * 0.5)
        
    def get_cost_field(self):
        # Softplus ensures the metric (friction) is always strictly positive
        # Add a tiny epsilon to prevent divide-by-zero or pure 0 cost
        return F.softplus(self.raw_cost) + 0.01

def softmin(tensors, tau=0.1):
    """ Differentiable minimum using LogSumExp """
    stacked = torch.stack(tensors, dim=0)
    return -tau * torch.logsumexp(-stacked / tau, dim=0)

def solve_geodesic(cost_field, start_x, start_y, iters=120, tau=0.1):
    """ Runs Differentiable Soft-Bellman to find the geodesic landscape """
    H, W = cost_field.shape
    device = cost_field.device
    
    V = torch.full((H, W), 100.0, device=device)
    V[start_x, start_y] = 0.0
    
    # Pad cost field to prevent wrapping around the edges (torus topology)
    pad_cost = F.pad(cost_field, (1, 1, 1, 1), value=100.0)
    
    for _ in range(iters):
        # We need V to be padded to shift safely
        V_pad = F.pad(V, (1, 1, 1, 1), value=100.0)
        
        V_up    = V_pad[:-2, 1:-1]
        V_down  = V_pad[2:, 1:-1]
        V_left  = V_pad[1:-1, :-2]
        V_right = V_pad[1:-1, 2:]
        
        # Softmin among neighbors and self
        V_min = softmin([V_up, V_down, V_left, V_right, V], tau=tau)
        
        # Add the local topological cost
        V_new = V_min + cost_field
        
        # Source must remain 0
        mask = torch.ones_like(V_new)
        mask[start_x, start_y] = 0.0
        V = V_new * mask
        
    return V

# ==========================================
# EXPERIMENT: TRANSITIVE LOGIC (A -> B -> C)
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCFManifold(size=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Semantic Coordinate Mapping
coords = {
    "A": (10, 10),
    "B": (32, 32),
    "C": (54, 54)
}

print("\nTraining Manifold (Injecting A->B and B->C constraints)...")
epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()
    
    cost_field = model.get_cost_field()
    
    # Compute Geodesic Landscape from A
    V_from_A = solve_geodesic(cost_field, *coords["A"], iters=120)
    # The loss is the path distance from A to B
    loss_AB = V_from_A[coords["B"][0], coords["B"][1]]
    
    # Compute Geodesic Landscape from B
    V_from_B = solve_geodesic(cost_field, *coords["B"], iters=120)
    # The loss is the path distance from B to C
    loss_BC = V_from_B[coords["C"][0], coords["C"][1]]
    
    # Regularization to prevent trivial collapse (all costs -> 0)
    # Force the average cost of the grid to remain around 1.0
    reg = F.mse_loss(cost_field.mean(), torch.tensor(1.0, device=device)) * 100.0
    
    total_loss = loss_AB + loss_BC + reg
    total_loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:02d} | Loss A->B: {loss_AB.item():.2f} | Loss B->C: {loss_BC.item():.2f} | Mean Cost: {cost_field.mean().item():.2f}")

print("\n==================================================")
print(" INFERENCE TEST: A -> C (Transitive Emergence) ")
print("==================================================")

with torch.no_grad():
    final_cost = model.get_cost_field()
    
    # Test 1: The Emergent Transitive Path
    V_A = solve_geodesic(final_cost, *coords["A"], iters=120)
    ans_AC = V_A[coords["C"][0], coords["C"][1]].item()
    
    # Test 2: Control Path (A -> Random empty space (54, 10))
    ans_ARandom = V_A[54, 10].item()
    
    print(f"Distance A -> C (Transitive Deduction) : {ans_AC:.2f}")
    print(f"Distance A -> Random Empty Space       : {ans_ARandom:.2f}")
    
    if ans_AC < ans_ARandom:
        print("\n✅ SUCCESS: The manifold naturally deduced A->C via the geometric valley formed by B, without any neural layers or explicit training on A->C!")
    else:
        print("\n❌ FAILURE: The manifold failed to connect A to C.")

# Save a visualization of the cost field
cost_np = final_cost.cpu().detach().numpy()
plt.figure(figsize=(8,8))
plt.imshow(cost_np, cmap='hot', interpolation='nearest')
plt.title("Learned Geometric Manifold (Cost Field)")
plt.plot(coords["A"][1], coords["A"][0], 'bo', markersize=10, label='A')
plt.plot(coords["B"][1], coords["B"][0], 'go', markersize=10, label='B')
plt.plot(coords["C"][1], coords["C"][0], 'yo', markersize=10, label='C')
plt.legend()
plt.savefig("results/manifold_visualization.png")
print("\nVisualization saved to results/manifold_visualization.png")
