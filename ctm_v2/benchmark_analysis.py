"""
CTM v2 Benchmark Analysis — Paper Figures
==========================================
Generates the figures needed for the research paper:
  Figure 1: Per-hop accuracy curve (Depth 1→10) — CTM Oracle vs Llama-3
  Figure 2: Composition table coverage heatmap
  Figure 3: Ablation results summary

Run this AFTER running train_ctm_v2.py and llama3_ctm_integration_v2.py.
It reads integration_v2_results.json.
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

print("==================================================")
print(" CTM v2 BENCHMARK ANALYSIS — PAPER FIGURES       ")
print("==================================================")


# ----------------------------------------------------------------
# FIGURE 1: Per-Depth Accuracy Curve
# ----------------------------------------------------------------
def plot_per_depth_accuracy(results_path="integration_v2_results.json"):
    """Generate per-hop accuracy curve from integration results."""
    if not os.path.exists(results_path):
        print(f"[SKIP] {results_path} not found. Run integration script first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    depth_llama = defaultdict(lambda: [0, 0])
    depth_oracle = defaultdict(lambda: [0, 0])
    depth_pipe = defaultdict(lambda: [0, 0])

    for r in results:
        d = r["depth"]
        depth_llama[d][0] += int(r["llama_correct"])
        depth_llama[d][1] += 1
        depth_oracle[d][0] += int(r["ctm_oracle_correct"])
        depth_oracle[d][1] += 1
        depth_pipe[d][0] += int(r["ctm_pipeline_correct"])
        depth_pipe[d][1] += 1

    depths = sorted(set(list(depth_llama.keys()) + list(depth_oracle.keys())))
    llama_accs = [100 * depth_llama[d][0] / depth_llama[d][1] if depth_llama[d][1] > 0 else 0 for d in depths]
    oracle_accs = [100 * depth_oracle[d][0] / depth_oracle[d][1] if depth_oracle[d][1] > 0 else 0 for d in depths]
    pipe_accs = [100 * depth_pipe[d][0] / depth_pipe[d][1] if depth_pipe[d][1] > 0 else 0 for d in depths]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(depths, llama_accs, 'o-', color='#ff6b6b', linewidth=2, markersize=8, label='Llama-3-8B (Direct)')
    ax.plot(depths, oracle_accs, 's-', color='#4ecdc4', linewidth=2, markersize=8, label='CTM v2 (Oracle Graph)')
    ax.plot(depths, pipe_accs, '^-', color='#45b7d1', linewidth=2, markersize=8, label='Llama-3 + CTM v2 (Pipeline)')

    ax.set_xlabel('Reasoning Depth (Number of Hops)', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Zero-Shot Reasoning Accuracy vs. Chain Depth', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figure1_per_depth_accuracy.png', dpi=300, bbox_inches='tight')
    print("Saved: figure1_per_depth_accuracy.png")

    # Print table for paper
    print("\n  Per-Depth Accuracy Table (for paper):")
    print(f"  {'Depth':<8} {'Llama-3':<12} {'CTM Oracle':<14} {'Pipeline':<12} {'N':<6}")
    print(f"  {'-'*52}")
    for d in depths:
        l = f"{100*depth_llama[d][0]/depth_llama[d][1]:.1f}%" if depth_llama[d][1] else "N/A"
        o = f"{100*depth_oracle[d][0]/depth_oracle[d][1]:.1f}%" if depth_oracle[d][1] else "N/A"
        p = f"{100*depth_pipe[d][0]/depth_pipe[d][1]:.1f}%" if depth_pipe[d][1] else "N/A"
        n = depth_llama[d][1]
        print(f"  {d:<8} {l:<12} {o:<14} {p:<12} {n:<6}")


# ----------------------------------------------------------------
# FIGURE 2: Composition Table Coverage Heatmap
# ----------------------------------------------------------------
def plot_composition_coverage():
    """Generate composition table coverage heatmap."""
    KINSHIP_RELATIONS = [
        "father", "mother", "son", "daughter", "grandfather", "grandmother",
        "grandson", "granddaughter", "uncle", "aunt", "nephew", "niece",
        "brother", "sister", "husband", "wife", "son-in-law", "daughter-in-law",
        "father-in-law", "mother-in-law", "brother-in-law", "sister-in-law"
    ]
    KIN2ID = {k: i for i, k in enumerate(KINSHIP_RELATIONS)}

    # Import the composition table from train_ctm_v2
    # (Inline the closure algorithm here for standalone use)
    BASE_RULES = {
        ("father","father"):"grandfather", ("father","mother"):"grandmother",
        ("mother","father"):"grandfather", ("mother","mother"):"grandmother",
        ("father","brother"):"uncle", ("father","sister"):"aunt",
        ("mother","brother"):"uncle", ("mother","sister"):"aunt",
        ("father","son"):"brother", ("father","daughter"):"sister",
        ("mother","son"):"brother", ("mother","daughter"):"sister",
        ("grandfather","son"):"uncle", ("grandfather","daughter"):"aunt",
        ("grandmother","son"):"uncle", ("grandmother","daughter"):"aunt",
        ("uncle","son"):"nephew", ("aunt","son"):"nephew",
        ("uncle","daughter"):"niece", ("aunt","daughter"):"niece",
        ("brother","son"):"nephew", ("sister","son"):"nephew",
        ("brother","daughter"):"niece", ("sister","daughter"):"niece",
        ("husband","father"):"father-in-law", ("husband","mother"):"mother-in-law",
        ("wife","father"):"father-in-law", ("wife","mother"):"mother-in-law",
        ("father","husband"):"son-in-law", ("mother","husband"):"son-in-law",
        ("father","wife"):"daughter-in-law", ("mother","wife"):"daughter-in-law",
        ("grandfather","father"):"grandfather", ("grandmother","mother"):"grandmother",
        ("grandfather","brother"):"uncle", ("grandmother","sister"):"aunt",
        ("nephew","father"):"brother", ("niece","mother"):"sister",
        ("nephew","mother"):"sister", ("niece","father"):"brother",
        ("son","son"):"grandson", ("son","daughter"):"granddaughter",
        ("daughter","son"):"grandson", ("daughter","daughter"):"granddaughter",
        ("son","brother"):"son", ("son","sister"):"daughter",
        ("daughter","brother"):"son", ("daughter","sister"):"daughter",
        ("brother","father"):"father", ("sister","father"):"father",
        ("brother","mother"):"mother", ("sister","mother"):"mother",
        ("brother","brother"):"brother", ("sister","sister"):"sister",
        ("brother","sister"):"sister", ("sister","brother"):"brother",
        ("grandson","father"):"son", ("granddaughter","father"):"son",
        ("grandson","mother"):"daughter", ("granddaughter","mother"):"daughter",
    }

    # Closure
    full_rules = dict(BASE_RULES)
    for _ in range(10):
        new = {}
        for (r1, r2), r3 in list(full_rules.items()):
            for (r3_, r4), r5 in list(full_rules.items()):
                if r3 == r3_ and (r1, r4) not in full_rules and r5 in KIN2ID:
                    new[(r1, r4)] = r5
        if not new:
            break
        full_rules.update(new)

    N = len(KINSHIP_RELATIONS)
    coverage = np.zeros((N, N))
    for (r1, r2), r3 in full_rules.items():
        if r1 in KIN2ID and r2 in KIN2ID:
            coverage[KIN2ID[r1], KIN2ID[r2]] = 1.0

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(coverage, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(KINSHIP_RELATIONS, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(KINSHIP_RELATIONS, fontsize=9)
    ax.set_xlabel('R2 (Second Relation)', fontsize=12)
    ax.set_ylabel('R1 (First Relation)', fontsize=12)
    ax.set_title(f'Kinship Composition Table Coverage ({len(full_rules)} rules from {len(BASE_RULES)} base)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Defined (1) / Undefined (0)')
    plt.tight_layout()
    plt.savefig('figure2_composition_coverage.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: figure2_composition_coverage.png")
    print(f"  Base rules: {len(BASE_RULES)}")
    print(f"  After closure: {len(full_rules)}")
    print(f"  Coverage: {len(full_rules)}/{N*N} = {100*len(full_rules)/(N*N):.1f}%")


# ----------------------------------------------------------------
# RUN ALL
# ----------------------------------------------------------------
if __name__ == "__main__":
    plot_per_depth_accuracy()
    plot_composition_coverage()
    print("\nAll figures generated.")
