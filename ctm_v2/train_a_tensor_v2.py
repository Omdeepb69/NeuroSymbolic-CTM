"""
CTM v2 A-Tensor Training (Validating Transitivity)
===================================================
Bug fixes from critic review:
  - Bug 7: get_targets vectorized with scatter (no Python loop)
  - Uses same log-space representation contract as train_ctm_v2.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import seaborn as sns

print("==================================================")
print(" CTM v2 A-TENSOR TRAINING (VALIDATING TRANSITIVITY)")
print("==================================================")

ARTIFACT_DIR = "ctm_artifacts/processed"
NUM_RELATIONS = 11  # WN18RR

# 1. Load Data
print("Loading datasets...")
paths = torch.load(f"{ARTIFACT_DIR}/wn18rr_2hop_paths.pt")
true_prob = torch.load(f"{ARTIFACT_DIR}/wn18rr_composition_prob.pt")

# Split Train / Val (80/20)
torch.manual_seed(42)
N = len(paths)
indices = torch.randperm(N)
train_idx, val_idx = indices[:int(0.8*N)], indices[int(0.8*N):]
train_paths = paths[train_idx]
val_paths = paths[val_idx]

print(f"Train samples: {len(train_paths)}")
print(f"Val samples:   {len(val_paths)}")

# 2. Model Definition
class ATensor(nn.Module):
    def __init__(self, num_rel):
        super().__init__()
        # Initialize with negative logits so default sigmoid prob is low (e.g. 0.05)
        self.A = nn.Parameter(torch.ones(num_rel, num_rel, num_rel) * -3.0)

    def forward(self, r1, r2):
        return self.A[r1, r2, :]

model = ATensor(NUM_RELATIONS)
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.BCEWithLogitsLoss()


def get_targets(labels, r3s):
    """
    Vectorized target generation (Bug 7 fix).
    Given an array of labels and r3 connections:
    If label=1, target is 1 at index r3, 0 elsewhere.
    If label=0, target is 0 everywhere.
    """
    B = labels.shape[0]
    targets = torch.zeros(B, NUM_RELATIONS)
    mask = labels.bool()
    if mask.any():
        targets[mask] = targets[mask].scatter(
            1, r3s[mask].unsqueeze(1).long(), 1.0
        )
    return targets


# 3. Training Loop
EPOCHS = 150
BATCH_SIZE = 256

print("\nStarting Training Loop...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    perm = torch.randperm(len(train_paths))
    shuffled_paths = train_paths[perm]

    for i in range(0, len(shuffled_paths), BATCH_SIZE):
        batch = shuffled_paths[i:i+BATCH_SIZE]
        r1, r2 = batch[:, 1], batch[:, 3]
        labels, r3s = batch[:, 5], batch[:, 6]

        targets = get_targets(labels, r3s)

        optimizer.zero_grad()
        logits = model(r1, r2)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        # Validation
        model.eval()
        with torch.no_grad():
            r1_val, r2_val = val_paths[:, 1], val_paths[:, 3]
            labels_val, r3_val = val_paths[:, 5], val_paths[:, 6]
            targets_val = get_targets(labels_val, r3_val)
            logits_val = model(r1_val, r2_val)
            val_loss = criterion(logits_val, targets_val).item()

            # Compute correlation with true discrete probabilities
            pred_probs = torch.sigmoid(model.A)
            corr = torch.corrcoef(torch.stack([pred_probs.flatten(), true_prob.flatten()]))[0, 1].item()

        print(f"Epoch {epoch:03d} | Train Loss: {total_loss/(len(train_paths)/BATCH_SIZE):.4f} | Val Loss: {val_loss:.4f} | GT Correlation: {corr:.4f}")

# 4. Visualization
print("\nTraining complete! Generating visualizations...")
model.eval()
learned_probs = torch.sigmoid(model.A).detach()

# Get max transitivity scores for a 2D heatmap summary
learned_max, _ = learned_probs.max(dim=2)
true_max, _ = true_prob.max(dim=2)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(true_max.numpy(), ax=axes[0], cmap='YlOrRd', vmin=0, vmax=1)
axes[0].set_title("Discrete Ground Truth (Empirical Probabilities)")
axes[0].set_xlabel("R2 (Second Relation)")
axes[0].set_ylabel("R1 (First Relation)")

sns.heatmap(learned_max.numpy(), ax=axes[1], cmap='YlOrRd', vmin=0, vmax=1)
axes[1].set_title("Learned Continuous A-Tensor (Gradient Descent)")
axes[1].set_xlabel("R2 (Second Relation)")
axes[1].set_ylabel("R1 (First Relation)")

plt.suptitle("Validation of Transitivity Emergence through Geometric Optimization", fontsize=16)

os.makedirs("ctm_artifacts/visualizations", exist_ok=True)
plt.savefig("ctm_artifacts/visualizations/a_tensor_v2_validation.png", bbox_inches='tight')
print("Saved visualization to: ctm_artifacts/visualizations/a_tensor_v2_validation.png")
