import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

print("==================================================")
print(" PHASE 3/4: FULL CTM INTEGRATION & BENCHMARK ")
print("==================================================")

ARTIFACT_DIR = "ctm_artifacts/processed"
NUM_RELS = 22  # Max kinship index + 1
MAX_N = 15     # Max people in a story

class CLUTRRDataset(Dataset):
    def __init__(self, json_path):
        print(f"Loading {json_path}...")
        with open(json_path) as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} stories")
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        story = self.data[idx]
        N = story['n_concepts']
        
        # Initialize Belief State W0: (MAX_N, MAX_N, NUM_RELS)
        W0 = torch.zeros(MAX_N, MAX_N, NUM_RELS)
        
        # Populate known edges
        rel_matrix = story['rel_matrix']
        hard_values = story['hard_values']
        
        for i in range(N):
            for j in range(N):
                r = rel_matrix[i][j]
                if r != -1:
                    W0[i, j, r] = hard_values[i][j]
                    
        qa, qb = story['query']
        target = int(story['target_rel_name'])
        
        return W0, qa, qb, target

train_dataset = CLUTRRDataset(f"{ARTIFACT_DIR}/clutrr_train.json")
test_dataset = CLUTRRDataset(f"{ARTIFACT_DIR}/clutrr_test.json")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

class ConstraintTopologyMachine(nn.Module):
    def __init__(self, num_rels=22):
        super().__init__()
        # The Reasoning Engine (A-Tensor)
        # Initialized to -3.0 so default prob is low.
        self.A = nn.Parameter(torch.ones(num_rels, num_rels, num_rels) * -3.0)
        
        # Scaling factor to convert bounded probabilities to raw logits for CrossEntropy
        self.scale = nn.Parameter(torch.tensor(10.0))
        
    def forward(self, W0, qa, qb, steps=3):
        W = W0
        P = torch.sigmoid(self.A)
        
        for _ in range(steps):
            # The core geometric routing equation (Tropical Max-Product Algebra)
            # U = max_{k, r1, r2} (W_{i, k, r1} * W_{k, j, r2} * P_{r1, r2, r3})
            
            U = torch.zeros_like(W)
            
            # W_left: (B, i, 1, k, r1, 1)
            W_left = W.unsqueeze(-1).unsqueeze(2)
            # W_right: (B, 1, j, k, 1, r2)
            W_right = W.transpose(1, 2).unsqueeze(-2).unsqueeze(1)
            
            for r3 in range(NUM_RELS):
                P_r3 = P[:, :, r3].view(1, 1, 1, 1, NUM_RELS, NUM_RELS)
                
                # Combine relationships to find the strongest logic path
                V_pairs = W_left * W_right * P_r3
                
                # Reduce over r1, r2, k
                max_r1, _ = V_pairs.max(dim=-2) # (B, i, j, k, r2)
                max_r2, _ = max_r1.max(dim=-1)  # (B, i, j, k)
                max_k, _ = max_r2.max(dim=-1)   # (B, i, j)
                
                U[:, :, :, r3] = max_k
            
            # Bounded topological union (Max-Plus)
            W = torch.max(W, U)
            
        B = W.shape[0]
        # Extract the relation distribution specifically for the query entities
        preds = W[torch.arange(B), qa, qb, :]
        return preds * self.scale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConstraintTopologyMachine(NUM_RELS).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

# --- TRAINING LOOP (Depth <= 3) ---
EPOCHS = 2
print("\n[Phase 1] Training on Short Stories (Depth <= 3)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (W0, qa, qb, target) in enumerate(train_loader):
        W0, qa, qb, target = W0.to(device), qa.to(device), qb.to(device), target.to(device)
        
        optimizer.zero_grad()
        # Train with T=3 steps
        logits = model(W0, qa, qb, steps=3)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {total_loss/(batch_idx+1):.4f}")
            
    acc = 100.0 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.2f}%")

# --- ZERO-SHOT GENERALIZATION BENCHMARK (Depth >= 5) ---
print("\n[Phase 2] Zero-Shot Generalization Benchmark (Depth >= 5)...")
model.eval()

# We test with different routing steps to see if the manifold dynamically traverses the gap
for steps in [3, 5, 8, 10]:
    correct = 0
    total = 0
    with torch.no_grad():
        for W0, qa, qb, target in test_loader:
            W0, qa, qb, target = W0.to(device), qa.to(device), qb.to(device), target.to(device)
            logits = model(W0, qa, qb, steps=steps)
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            
    acc = 100.0 * correct / total
    print(f"Test Accuracy (T={steps} steps): {acc:.2f}%")

print("\nCTM Training & Evaluation Complete!")
